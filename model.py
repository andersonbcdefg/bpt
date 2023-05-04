import yaml
import warnings
import math
import bitsandbytes as bnb
from rotary_embedding_torch import RotaryEmbedding
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from dataclasses import dataclass

def check_nan(tensor, label):
    if torch.isnan(tensor).any():
      print(f"{label} contains {torch.mean(torch.isnan(tensor).float())} NaN values")

@dataclass
class GPTConfig:
  vocab_size: int
  seq_len: int
  d_model: int
  d_ffn: int
  d_qkv: int
  n_heads: int
  n_layers: int
  dropout: float
  tie_weights: bool
  rotary_emb: bool = True
  stable_embedding: bool = False
  rotary_pct: float = 0.25

  def __post_init__(self):
    if self.d_qkv != self.d_model // self.n_heads:
      warnings.warn("d_qkv is not equal to d_model // n_heads. This is ok, but unusual.")

  @classmethod
  def from_yaml(cls, path):
    with open(path, 'r') as f:
      return cls(**yaml.safe_load(f))

  def __repr__(self):
    return yaml.dump(self.__dict__)

  def __str__(self):
    return yaml.dump(self.__dict__)

class RMSNorm(nn.Module):
  def __init__(self, dim, eps = 1e-8):
    super().__init__()
    self.scale = dim ** -0.5
    self.eps = eps
    self.g = nn.Parameter(torch.ones(dim))

  def forward(self, x):
    norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
    return x / norm.clamp(min = self.eps) * self.g


class CausalAttention(nn.Module):
  def __init__(self, config, rotary_emb=None):
    super().__init__()
    self.W_qkv = nn.Linear(config.d_model, config.n_heads * config.d_qkv * 3, bias=False)
    self.rotary_emb = rotary_emb
    self.dropout_p = config.dropout
    self.out_proj = nn.Linear(config.d_qkv * config.n_heads, config.d_model, bias=False)
    self.d_qkv = config.d_qkv

  def forward(self, X):
    Q, K, V = rearrange(self.W_qkv(X), "b l (h ddd) -> b h l ddd", ddd=(3 * self.d_qkv)).chunk(3, dim=-1) # b, h, l, d_attn
    # Q, K = self.rotary_emb.rotate_queries_and_keys(Q, K) if self.rotary_emb is not None else (Q, K)
    if self.rotary_emb is not None:
      Q = self.rotary_emb.rotate_queries_or_keys(Q)
      K = self.rotary_emb.rotate_queries_or_keys(K)
    # check_nan(Q, "Q")
    # check_nan(K, "K")
    attn_out = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout_p, is_causal=True).transpose(1, 2) # b, l, h, d_attn
    return self.out_proj(attn_out.flatten(2, 3))

# use SwiGLU gated unit
class FFN(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.fc1 = nn.Linear(config.d_model, config.d_ffn * 2)
    self.fc2 = nn.Linear(config.d_ffn, config.d_model)

  def forward(self, X):
    a, b = self.fc1(X).chunk(2, dim=-1)
    return self.fc2(a * F.silu(b))

class ParallelTransformerBlock(nn.Module):
  def __init__(self, config, rotary_emb=None):
    super().__init__()
    self.ln1 = RMSNorm(config.d_model)
    self.attn = CausalAttention(config, rotary_emb=rotary_emb)
    self.ln2 = RMSNorm(config.d_model)
    self.ffn = FFN(config)

  def forward(self, X):
    attn_out = self.attn(self.ln1(X))
    # check_nan(attn_out, "attn_out")
    ffn_out = self.ffn(self.ln2(X))
    # check_nan(ffn_out, "ffn_out")
    return X + attn_out + ffn_out
  
class FusedParallelTransformerBlock(nn.Module):
  def __init__(self, config, rotary_emb = None):
    super().__init__()
    self.ln = RMSNorm(config.d_model)
    self.dense = nn.Linear(config.d_model, config.d_qkv * config.n_heads * 3 + config.d_ffn * 2)
    self.dropout_p = config.dropout
    self.attn_out_proj = nn.Linear(config.d_qkv * config.n_heads, config.d_model, bias=False)
    self.ffn_out_proj = nn.Linear(config.d_ffn, config.d_model, bias=False)

  def forward(self, x):
    normed = self.ln(x)
    # apply dense layer to x then split into q, k, v, a, b (not all the same size!)
    # this is the "fused" bit that might save time, idk
    q, k, v, a, b = self.dense(normed).split([
      config.d_qkv * config.n_heads, 
      config.d_qkv * config.n_heads, 
      config.d_qkv * config.n_heads, 
      config.d_ffn, 
      config.d_ffn
    ], dim=-1)
    # split heads
    q, k, v = map(lambda t: rearrange(t, "b l (h d) -> b h l d", h=config.n_heads), (q, k, v))
    # apply rotary embedding to q and k
    if self.rotary_emb is not None:
      q = self.rotary_emb.rotate_queries_or_keys(q)
      k = self.rotary_emb.rotate_queries_or_keys(k)
    # apply attention
    attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p, is_causal=True).transpose(1, 2) # b, l, h, d_attn
    
    return x + self.attn_out_proj(attn_out.flatten(2, 3)) + self.ffn_out_proj(a * F.silu(b))






class GPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.d_model = config.d_model
    if config.stable_embedding:
      self.token_emb = bnb.nn.StableEmbedding(config.vocab_size, config.d_model)
    else:
      self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
    if config.rotary_emb:
      self.rotary_emb = RotaryEmbedding(dim = int(config.d_qkv * config.rotary_pct), use_xpos = False)
    else:
      self.rotary_emb = None
    self.transformer = nn.Sequential(*[ParallelTransformerBlock(config, self.rotary_emb) for _ in range(config.n_layers)])
    self.norm = RMSNorm(config.d_model)
    self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    if config.tie_weights:
      self.lm_head.weight = self.token_emb.weight

    n_params = (sum(p.numel() for p in self.token_emb.parameters()) +
                    # self.pos_emb.numel() +
                    sum(p.numel() for p in self.transformer.parameters()) +
                    sum(p.numel() for p in self.norm.parameters()) +
                    sum(p.numel() for p in self.lm_head.parameters())
    )
    if self.rotary_emb is not None:
      n_params += sum(p.numel() for p in self.rotary_emb.parameters())
    if config.tie_weights:
      n_params -= self.lm_head.weight.numel()
    print("Number of parameters: ~%.0fM" % (n_params/1e6,))

    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      nn.init.xavier_normal_(module.weight)
    elif isinstance(module, nn.Embedding):
      nn.init.xavier_normal_(module.weight, gain=0.5)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()

  def forward(self, X, targets=None):
    X = self.token_emb(X)
    # check_nan(X, "Token embeddings")

    for i, layer in enumerate(self.transformer):
        X = layer(X)
        # check_nan(X, f"Transformer layer {i+1}")

    X = self.norm(X)
    # check_nan(X, "Normalization")

    logits = self.lm_head(X)
    # check_nan(logits, "Logits")

    if targets is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
      # check_nan(loss, "Loss")
      return loss

    return logits


  # def forward(self, X, targets=None):
  #   X = self.token_emb(X)
  #   X = self.transformer(X)
  #   X = self.norm(X)
  #   logits =  self.lm_head(X)
  #   if targets is not None:
  #     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
  #     return loss
  #   return logits   
  
if __name__ == "__main__":
  config = GPTConfig.from_yaml("config/medium.yaml")
  model = GPT(config)
  # print(model)
  X = torch.randint(0, config.vocab_size, (2, 128))
  loss = model(X, targets=X)
  print(loss)