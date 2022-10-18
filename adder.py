import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import GPT
from einops import rearrange

def construct_dataset(size, max_val):
    max_len = len(f"{max_val}+{max_val}={max_val * 2}.")
    dataset = []
    for iter in range(size):
        int1 = np.random.randint(0, max_val)
        int2 = np.random.randint(0, max_val)
        sum = int1 + int2
        seq = f"{int1}+{int2}={sum}."
        dataset.append(seq)
    return max_len, dataset

def construct_test(size, max_val):
    max_len = len(f"{max_val}+{max_val}={max_val * 2}.")
    X = []
    y = []
    for iter in range(size):
        int1 = np.random.randint(0, max_val)
        int2 = np.random.randint(0, max_val)
        sum = int1 + int2
        seq = f"{int1}+{int2}="
        X.append(seq)
        y.append(str(sum))
    return max_len, X, y

def get_vocab():
    chr_to_i = {}
    chr_to_i[" "] = 0
    chr_to_i["."] = 1
    chr_to_i["="] = 2
    chr_to_i["+"] = 3
    for i in range(10):
        chr_to_i[str(i)] = 4 + i
    i_to_chr = {v: k for k, v in chr_to_i.items()}
    return chr_to_i, i_to_chr

def tokenize(seq, chr_to_i, max_len, pad=True):
    chars = list(seq)[-max_len:]
    tokens = [chr_to_i[c] for c in chars]
    if pad:
        tokens = tokens + [0] * (max_len - len(tokens))
    return tokens

def detokenize(tokens, i_to_chr):
    return "".join([i_to_chr[t] for t in tokens]).replace(" ", "")

def generate(model, input, chr_to_i, i_to_chr, max_len):
    result = tokenize(input, chr_to_i, max_len, pad=False)
    while True:
        idx = len(result)
        padded = result + [0] * (max_len - idx)
        output = model(torch.tensor(padded).unsqueeze(0))
        output_char = torch.argmax(output, dim=-1)[0, idx - 1].item()
        if output_char == 0 or output_char == 1:
            break
        result += [output_char]
    return detokenize(result, i_to_chr)

class AdderDataset(Dataset):
    def __init__(self, size, max_val):
        self.size = size
        self.max_len, raw_dataset = construct_dataset(size, max_val)
        self.chr_to_i, self.i_to_chr = get_vocab()
        self.X = [tokenize(seq, self.chr_to_i, self.max_len) for seq in raw_dataset]
        self.Y = [seq[1:] + [0] for seq in self.X]
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])


if __name__ == "__main__":
    train_dataset = AdderDataset(100000, 10000)
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    config = {
        "vocab_size": len(train_dataset.chr_to_i),
        "seq_len": train_dataset.max_len,
        "d_model": 48,
        "n_layers": 4,
        "n_heads": 8,
        "d_attn": 6,
        "embedding_dropout": 0.1,
        "attn_dropout": 0.1,
        "resid_dropout": 0.1
    }
    model = GPT(config)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    max_steps = 1750
    running_loss = 0.0
    steps = 0
    while True:
        for i, (X, Y) in enumerate(train_dataloader):
            steps += 1
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(rearrange(out, "b l d -> (b l) d"), Y.flatten())
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if steps % 50 == 0:
                print(f"Step {steps} | Loss: {running_loss / 50}")
                running_loss = 0.0
            if steps >= max_steps:
                break
        
        if steps >= max_steps:
            break

    print("Training complete.")
    max_len, X_test, y_test = construct_test(1000, 10000)
    preds = []
    for test_idx in range(len(X_test)):
        X = X_test[test_idx]
        pred = generate(model, X, train_dataset.chr_to_i, train_dataset.i_to_chr, max_len)[len(X):]
        preds.append(pred)

    accuracy = np.mean([pred == y_test[i] for i, pred in enumerate(preds)])
    print(f"Accuracy: {accuracy}")
    torch.save(model.state_dict(), "model.pth")
    
