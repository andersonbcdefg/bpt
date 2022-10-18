import torch
from model import GPT

config = {
    "vocab_size": 50257, 
    "seq_len": 512, 
    "d_model": 512, 
    "n_layers":6, 
    "n_heads":8, 
    "d_attn":64, 
    "attn_dropout":0.1, 
    "resid_dropout":0.1
}

model = GPT(config)

example_input = torch.randint(0, config["vocab_size"], (1, config["seq_len"]))
out = model(example_input)
print(out.shape)
