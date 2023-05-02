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
    """Root Mean Square Layer Normalization.
    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed

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
    Q, K = self.rotary_emb.rotate_queries_and_keys(Q, K) if self.rotary_emb is not None else (Q, K)
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
    return X + self.ffn(self.ln2(X)) + self.attn(self.ln1(X))

class GPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.d_model = config.d_model
    if config.stable_embedding:
      self.token_emb = bnb.nn.StableEmbedding(config.vocab_size, config.d_model)
    else:
      self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
    if config.rotary_emb:
      self.rotary_emb = RotaryEmbedding(dim = int(math.floor(config.d_qkv * config.rotary_pct)), use_xpos = True)
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

  def forward(self, X, targets=None):
    X = self.token_emb(X)
    X = self.transformer(X)
    X = self.norm(X)
    logits =  self.lm_head(X)
    if targets is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
      return loss
    return logits   
  
if __name__ == "__main__":
  config = GPTConfig.from_yaml("config/medium.yaml")
  model = GPT(config)
  print(model)
  X = torch.randint(0, config.vocab_size, (2, 128))
  print(model(X).shape)