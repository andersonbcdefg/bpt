import bitsandbytes as bnb
import torch
import config
from torch.utils.checkpoint import checkpoint_sequential
from torch import nn
from torch.nn import functional as F

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
    self.transformer = nn.Sequential(*[FusedParallelTransformerBlock(config) for _ in range(config.n_layers)])
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
  config = config.GPTConfig.from_yaml("config/3b-default.yaml")
  model = GPT(config)
  # print(model)
  X = torch.randint(0, config.vocab_size, (2, 128))
  loss = model(X, labels=X)
  print(loss)
  loss.backward()
  print(model.token_emb.weight.grad)