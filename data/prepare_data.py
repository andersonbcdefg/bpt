import os
import fire
import json
import functools
import torch
from tqdm.auto import tqdm
from datasets import load_dataset, interleave_datasets, concatenate_datasets
from transformers import LlamaTokenizer
from torch.utils.data import IterableDataset, DataLoader

def apply_tokenizer_to_batch(batch, tokenizer):
    ids = tokenizer(batch["text"], padding=False, truncation=False).input_ids
    return {"tokens": [[item for sublist in ids for item in sublist]]} # flatten

class TextDataset(IterableDataset):
    def __init__(self, streaming_dataset, tokenizer, seq_len):
        self.tokenize = functools.partial(apply_tokenizer_to_batch, tokenizer=tokenizer)
        self.dataset = streaming_dataset.map(self.tokenize, batched=True, batch_size=32, remove_columns=["id", "title", "text", "reddit_scores", "timestamp", "url"])
        self.seq_len = seq_len

    def __iter__(self):
        current_tokens = []
        for doc in self.dataset:
            current_tokens.extend(doc["tokens"])
            while len(current_tokens) > self.seq_len:
                x, y = current_tokens[:self.seq_len], current_tokens[1:self.seq_len + 1]
                yield (
                    torch.tensor(x, dtype=torch.long),
                    torch.tensor(y, dtype=torch.long)
                )


"""
Get dataset that samples from a variety of English language datasets.
"""
def get_dataloader(seq_len=512, bsz=32):
    tokenizer = LlamaTokenizer.from_pretrained(".")

    # load datasets
    wiki = load_dataset("andersonbcdefg/combined_en_wikipedia", split="train", streaming=True)
    books1 = load_dataset("bookcorpusopen", split="train", streaming=True) # key: "text"
    books3 = load_dataset("the_pile_books3", split="train", streaming=True) # key: "text"
    webtext = load_dataset("the_pile_openwebtext2", split="train", streaming=True) # key: "text"
    c4 = load_dataset("c4", "en", split="train", streaming=True) # key: "text"
    news = load_dataset("c4", "realnewslike", split="train", streaming=True) # key: "text"

    # interleave datasets
    streaming_probabilities = [0.09, 0.01, 0.15, 0.2, 0.3, 0.25]
    streaming_dataset = interleave_datasets([wiki, books1, books3, webtext, c4, news], probabilities=streaming_probabilities)

    # wrap in iterable dataset
    dataset = TextDataset(streaming_dataset, tokenizer, seq_len)

    return DataLoader(dataset, batch_size=bsz, num_workers=0, pin_memory=True)