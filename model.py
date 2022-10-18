import yaml
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

class PreNormAndAdd(nn.Module):
  def __init__(self, d_model, sublayer):
    super().__init__()
    self.norm = nn.LayerNorm(d_model)
    self.sublayer = sublayer

  def forward(self, X):
    return X + self.sublayer(self.norm(X))

class CausalAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.W_qkv = nn.Linear(config['d_model'], config['n_heads'] *config['d_attn'] * 3)
    self.d_attn = config['d_attn']
    self.register_buffer("causal_mask", torch.tril(torch.ones(config['seq_len'], config['seq_len'])).unsqueeze(0).unsqueeze(0))
    self.attn_dropout = nn.Dropout(config['attn_dropout'])
    self.out_proj = nn.Linear(config['d_attn'] * config['n_heads'], config['d_model'])
    self.resid_dropout = nn.Dropout(config['resid_dropout'])

  def forward(self, X):
    Q, K, V = rearrange(self.W_qkv(X), "b l (h ddd) -> b l h ddd", ddd=(3 * self.d_attn)).chunk(3, dim=-1) # b, l, h, d_attn
    attn_weights = torch.einsum("blhd, bkhd -> bhlk", Q, K) / self.d_attn **0.5
    attn_weights.masked_fill_(self.causal_mask == 0, float("-inf"))
    attn_weights = self.attn_dropout(F.softmax(attn_weights, dim=-1))
    attn_out = torch.einsum("bhlk, bkhd -> blhd", attn_weights, V)
    return self.resid_dropout(self.out_proj(attn_out.flatten(2, 3)))

class FFN(nn.Module):
  def __init__(self, config):
    super().__init__()
    d_model = config['d_model']
    self.net = nn.Sequential(
        nn.Linear(d_model, 4 * d_model),
        nn.GELU(),
        nn.Linear(4 * d_model, d_model),
        nn.Dropout(config['resid_dropout'])
    )

  def forward(self, X):
    return self.net(X)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = PreNormAndAdd(config['d_model'], CausalAttention(config))
        self.ffn = PreNormAndAdd(config['d_model'], FFN(config))
    
    def forward(self, X):
        return self.ffn(self.attn(X))

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config["vocab_size"] is not None
        assert config["seq_len"] is not None

        self.token_emb = nn.Embedding(config['vocab_size'], config['d_model'])
        self.pos_emb = nn.Parameter(torch.randn(config['seq_len'], config['d_model']))
        self.transformer = nn.Sequential(*[TransformerBlock(config) for _ in range(config['n_layers'])])
        self.norm = nn.LayerNorm(config['d_model'])
        self.lm_head = nn.Linear(config['d_model'], config['vocab_size'])
        if "tie_weights" in config and config['tie_weights']:
            self.lm_head.weight = self.token_emb.weight

    def forward(self, X):
        X = self.token_emb(X) + self.pos_emb
        X = self.transformer(X)
        X = self.norm(X)
        return self.lm_head(X)
        