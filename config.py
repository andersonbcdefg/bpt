import yaml
import warnings
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
  rotary_pct: float = 0.25
  use_xpos: bool = True
  xpos_scale_base: int = 512
  stable_embedding: bool = False
  checkpointing: bool = False

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