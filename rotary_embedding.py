## Adapted from Phil Wang (lucidrains) Github repo `palm_rlhf_pytorch`.
# I modify the apply_rotary_pos_emb function to allow rotating only a fraction of the input tensor.

import torch
import torch.nn as nn
from einops import rearrange

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
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale