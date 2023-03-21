import yaml
import warnings
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
  attn_dropout: float
  dropout: float
  tie_weights: bool
  emb_scale_factor: float = 1.0
  weight_init_scale: float = 0.02

  def __post_init__(self):
    if self.d_qkv != self.d_model // self.n_heads:
      warnings.warn("d_qkv is not equal to d_model // n_heads. This is ok, but unusual.")


  @classmethod
  def from_yaml(cls, path):
    with open(path, 'r') as f:
      return cls(**yaml.load(f, Loader=yaml.FullLoader))

  def __repr__(self):
    return yaml.dump(self.__dict__)

  def __str__(self):
    return yaml.dump(self.__dict__)


# Adapted from RMSNorm repository
# License: https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE
# no option to use bias because i'm making the executive decision that it's dumb
# option to turn off weight for speed w/o substantially decreasing parameters.
# but not sure if performance will suffer.
class RMSNorm(nn.Module):
  def __init__(self, d_model, eps=1e-8, use_weight=False):
    super().__init__()
    self.d_model = d_model
    self.eps = eps
    self.weight = None
    if use_weight:
      self.weight = nn.Parameter(torch.ones(d_model))
      self.register_parameter("weight", self.weight)

  def forward(self, x):
    norm_x = x.norm(2, dim=-1, keepdim=True) # "length" of x
    rms_x = norm_x * self.d_model ** (-1. / 2) # rms of x
    x_normed = x / (rms_x + self.eps) # normalize x by rms

    if self.weight is not None:
      return self.weight * x_normed

    return x_normed

class PreNormAddDropout(nn.Module):
  def __init__(self, d_model, sublayer, dropout):
    super().__init__()
    self.norm = RMSNorm(d_model, use_weight=False)
    self.sublayer = sublayer
    self.dropout = nn.Dropout(dropout)

  def forward(self, X):
    return X + self.dropout(self.sublayer(self.norm(X)))

class CausalAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.W_qkv = nn.Linear(config.d_model, config.n_heads * config.d_qkv * 3, bias=False)
    self.attn_dropout_p = config.attn_dropout
    self.out_proj = nn.Linear(config.d_qkv * config.n_heads, config.d_model, bias=False)
    self.d_qkv = config.d_qkv

  def forward(self, X):
    Q, K, V = rearrange(self.W_qkv(X), "b l (h ddd) -> b h l ddd", ddd=(3 * self.d_qkv)).chunk(3, dim=-1) # b, h, l, d_attn
    attn_out = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.attn_dropout_p, is_causal=True).transpose(1, 2) # b, l, h, d_attn
    return self.out_proj(attn_out.flatten(2, 3))

# use SWIGLU gated unit
class FFN(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.fc1 = nn.Linear(config.d_model, config.d_ffn * 2)
    self.fc2 = nn.Linear(config.d_ffn, config.d_model)

  def forward(self, X):
    a, b = self.fc1(X).chunk(2, dim=-1)
    return self.fc2(a * F.silu(b))

class TransformerBlock(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.attn = PreNormAddDropout(config.d_model, CausalAttention(config), config.dropout)
    self.ffn = PreNormAddDropout(config.d_model, FFN(config), config.dropout)

  def forward(self, X):
    return self.ffn(self.attn(X))

def get_position_embeddings(seq_len, d_model, scale_factor=1.0):
  position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
  div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
  pos_embeddings = torch.zeros(seq_len, d_model)
  pos_embeddings[:, 0::2] = torch.sin(position * div_term)
  pos_embeddings[:, 1::2] = torch.cos(position * div_term)
  return pos_embeddings.unsqueeze(0) * scale_factor

class GPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.d_model = config.d_model
    self.seq_len = config.seq_len # this is the maximum seq len, will only have this many position embeddings
    self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
    self.pos_emb = nn.Parameter(get_position_embeddings(config.seq_len, config.d_model, scale_factor=config.emb_scale_factor), requires_grad=True)
    self.transformer = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layers)])
    self.norm = RMSNorm(config.d_model)
    self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    if config.tie_weights:
      self.lm_head.weight = self.token_emb.weight

    n_params = (sum(p.numel() for p in self.token_emb.parameters()) +
                    self.pos_emb.numel() +
                    sum(p.numel() for p in self.transformer.parameters()) +
                    sum(p.numel() for p in self.norm.parameters()) +
                    sum(p.numel() for p in self.lm_head.parameters())
    )
    if config.tie_weights:
      n_params -= self.lm.weight.numel()
    print("Number of parameters: ~%.0fM" % (n_params/1e6,))

  

  def forward(self, X, targets=None):
    B, L = X.shape
    X = self.token_emb(X) + self.pos_emb[:, :L, :] # can work with any seq len <= self.seq_len
    X = self.transformer(X)
    X = self.norm(X)
    logits =  self.lm_head(X)
    if targets is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
      return loss
    return logits   