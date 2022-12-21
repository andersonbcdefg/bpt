import torch
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

class TextDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens - block_size - 1)

    def __getitem__(self, idx):
        X = torch.tensor(self.tokens[idx:idx + self.block_size])
        Y = torch.tensor(self.tokens[idx + 1:idx + self.block_size + 1])
        return X, Y

text = open("../..//Downloads/shakespeare.txt").read()
tokens = tokenizer.encode(text)
dataset = TextDataset(tokens, 512)

X, Y = dataset[100]
print(X.shape, Y.shape)



    