import yaml
import warnings
import bitsandbytes as bnb
import torch
from torch.utils.checkpoint import checkpoint_sequential
from torch import nn
from torch.nn import functional as F
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
  fused_transformer_block: bool = True
  rotary_emb: bool = True
  rotary_pct: float = 0.25
  use_xpos: bool = True
  xpos_scale_base: int = 512
  stable_embedding: bool = False
  checkpointing: bool = True
  

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

## Rotary embedding adapted from Phil Wang (lucidrains) Github repo `palm_rlhf_pytorch`.
# I modify the apply_rotary_pos_emb function to allow rotating only a fraction of the input tensor.


@torch.jit.script
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

@torch.jit.script
def apply_rotary_pos_emb(pos, t, scale):
    num_rot, d_qkv = pos.shape[-1], t.shape[-1]
    t_rot, t_keep = t.split((num_rot, d_qkv - num_rot), dim=-1)
    rotated = (t_rot * pos.cos() * scale) + (rotate_half(t_rot) * pos.sin() * scale)
    return torch.cat((rotated, t_keep), dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, scale_base = 512, use_xpos = True):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.use_xpos = use_xpos
        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device = device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        if not self.use_xpos:
            return freqs, torch.ones(1, device = device)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** power.unsqueeze(-1)
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale

class CausalAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.W_qkv = nn.Linear(config.d_model, config.n_heads * config.d_qkv * 3, bias=False)
    self.rotary_emb = RotaryEmbedding(int(config.d_qkv * config.rotary_pct), scale_base=config.xpos_scale_base, use_xpos=config.use_xpos)
    self.dropout_p = config.dropout
    self.out_proj = nn.Linear(config.d_qkv * config.n_heads, config.d_model, bias=False)
    self.d_qkv = config.d_qkv
    self.n_heads = config.n_heads

    self.register_buffer("pos_emb", None, persistent=False)
    self.register_buffer("pos_emb_scale", None, persistent=False)

  def get_rotary_embedding(self, n, device):
    if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
      return self.pos_emb[:n], self.pos_emb_scale[:n]

    pos_emb, scale = self.rotary_emb(n, device=device)
    self.register_buffer("pos_emb", pos_emb, persistent=False)
    self.register_buffer("pos_emb_scale", scale, persistent=False)
    return pos_emb, scale
  
  def forward(self, x):
    seq_len, device = x.shape[1], X.device
    qkv = self.W_qkv(x) # b, l, d_qkv * n_heads * 3
    new_qkv_shape = qkv.size()[:-1] + (self.n_heads, 3 * self.d_qkv) # b, l, h, d_qkv * 3
    q, k, v = qkv.view(*new_qkv_shape).permute(0, 2, 1, 3).chunk(3, dim=-1)

    positions, scale = self.get_rotary_embedding(seq_len, device)

    q = apply_rotary_pos_emb(positions, q, scale)
    k = apply_rotary_pos_emb(positions, k, scale ** -1)
    attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p, is_causal=True).transpose(1, 2) # b, l, h, d_attn
    return self.out_proj(attn_out.flatten(2, 3))

# use SwiGLU gated unit
class FFN(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.fc1 = nn.Linear(config.d_model, config.d_ffn * 2, bias=False)
    self.fc2 = nn.Linear(config.d_ffn, config.d_model, bias=False)

  def forward(self, X):
    a, b = self.fc1(X).chunk(2, dim=-1)
    return self.fc2(a * F.silu(b))

class ParallelTransformerBlock(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.ln1 = RMSNorm(config.d_model)
    self.attn = CausalAttention(config)
    self.ln2 = RMSNorm(config.d_model)
    self.ffn = FFN(config)

  def forward(self, x):
    attn_out = self.attn(self.ln1(x))
    ffn_out = self.ffn(self.ln2(x))
    return x + attn_out + ffn_out
  
class FusedParallelTransformerBlock(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.ln = RMSNorm(config.d_model)
    self.dense = nn.Linear(config.d_model, config.d_qkv * config.n_heads * 3 + config.d_ffn * 2, bias=False)
    if config.rotary_emb:
      self.rotary_emb = RotaryEmbedding(int(config.d_qkv * config.rotary_pct), scale_base=config.xpos_scale_base, use_xpos=config.use_xpos)
      self.register_buffer("pos_emb", None, persistent=False)
      self.register_buffer("pos_emb_scale", None, persistent=False)
    else:
      self.rotary_emb = None
    self.dropout_p = config.dropout
    self.attn_out_proj = nn.Linear(config.d_qkv * config.n_heads, config.d_model, bias=False)
    self.ffn_out_proj = nn.Linear(config.d_ffn, config.d_model, bias=False)

  def get_rotary_embedding(self, n, device):
    if self.rotary_emb is None:
      return
    if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
      return self.pos_emb[:n], self.pos_emb_scale[:n]

    pos_emb, scale = self.rotary_emb(n, device=device)
    self.register_buffer("pos_emb", pos_emb, persistent=False)
    self.register_buffer("pos_emb_scale", scale, persistent=False)
    return pos_emb, scale

  def forward(self, x):
    seq_len, device = x.shape[1], x.device
    normed = self.ln(x)
    # apply dense layer to x then split into q, k, v, a, b
    qkv, a, b = self.dense(normed).split([
      3 * self.config.d_qkv * self.config.n_heads,  
      self.config.d_ffn, 
      self.config.d_ffn
    ], dim=-1)
    new_qkv_shape = qkv.size()[:-1] + (self.config.n_heads, 3 * self.config.d_qkv) # b, l, h, d_qkv * 3
    q, k, v = qkv.view(*new_qkv_shape).permute(0, 2, 1, 3).chunk(3, dim=-1)

    # apply rotary embedding, if applicable
    if self.rotary_emb:
      positions, scale = self.get_rotary_embedding(seq_len, device)
      q = apply_rotary_pos_emb(positions, q, scale)
      k = apply_rotary_pos_emb(positions, k, scale ** -1)
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
    if config.fused_transformer_block:
      self.transformer = nn.Sequential(*[FusedParallelTransformerBlock(config) for _ in range(config.n_layers)])
    else:
      self.transformer = nn.Sequential(*[ParallelTransformerBlock(config) for _ in range(config.n_layers)])
    self.norm = RMSNorm(config.d_model)
    self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    if config.tie_weights:
      self.lm_head.weight = self.token_emb.weight
    self.checkpointing = config.checkpointing

    n_params = (sum(p.numel() for p in self.token_emb.parameters()) +
                    # self.pos_emb.numel() +
                    sum(p.numel() for p in self.transformer.parameters()) +
                    sum(p.numel() for p in self.norm.parameters()) +
                    sum(p.numel() for p in self.lm_head.parameters())
    )
    if config.tie_weights:
      n_params -= self.lm_head.weight.numel()
    print("Number of parameters: ~%.0fM" % (n_params/1e6,))

    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      nn.init.xavier_normal_(module.weight)
    elif isinstance(module, nn.Embedding):
      nn.init.xavier_normal_(module.weight, gain=0.5)
    if isinstance(module, nn.Linear) and module.bias is not None and module.bias is not False:
      module.bias.data.zero_()

  def forward(self, x, labels=None):
    x = self.token_emb(x)

    if self.checkpointing:
      x = checkpoint_sequential(self.transformer, len(self.transformer) // 2, x)
    else:
      x = self.transformer(x)

    x = self.norm(x)
    logits = self.lm_head(x)
    if labels is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
      return loss
    return logits 
  
if __name__ == "__main__":
  config = GPTConfig.from_yaml("config/medium.yaml")
  config.fused_transformer_block = True
  model = GPT(config)
  # print(model)
  X = torch.randint(0, config.vocab_size, (2, 128))
  loss = model(X, targets=X)
  print(loss)
  loss.backward()
  print(model.token_emb.weight.grad)