import regex as re
import numpy as np
import json
from itertools import chain

# Simple one-to-one mapping of bytes 0-255 into Unicode characters
# that "look nice" (no '\xc3' type of characters).
def bytes_to_chars():
  already_nice = list(chain(range(33, 127), range(161, 173), range(174, 256)))
  to_map = [x for x in range(0, 256) if x not in already_nice]
  mapped = [ix + 256 for ix in range(len(to_map))]
  byte_to_chr_dict = {b:chr(b) for b in already_nice}
  byte_to_chr_dict.update({to_map[i]:chr(mapped[i]) for i in range(len(to_map))})
  return byte_to_chr_dict

# Break tuple 'word' into set of bigrams
def get_pairs(word):
    return set(zip(word[:-1], word[1:]))

  # Similar to OpenAI class, but with ability to learn merges from corpus, or load vocab/encoder from file.
class BytePairEncoder:
  def __init__(self):
    self.pretok_re = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    self.byte_encoder = bytes_to_chars()
    self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
    
    # Unlike OpenAI GPT or minGPT, we either explicitly load or learn these
    self.encoder = None # Dictionary of <merged_token> : <int_encoding>
    self.decoder = None # Inverse dictionary of encoder: <int_encoding> : <merged_token>
    self.bpe_ranks = None # Dictionary of <pair_to_merge> : <rank>

    # Cache for efficiency
    self.cache = {}

  def _pretokenize(self, text):
    raw_words = re.findall(self.pretok_re, text)
    as_bytes = [tok.encode('utf-8') for tok in raw_words]
    as_readable_bytes = ["".join(self.byte_encoder[b] for b in bytestring) for bytestring in as_bytes]
    return as_readable_bytes

  def _update_stats(self):
    self.pairs = {}
    for word, freq in self.vocab.items():
      for i in range(len(word) - 1):
        self.pairs[(word[i], word[i + 1])] = self.pairs.get((word[i], word[i + 1]), 0) + freq 

  def _get_best_pair(self, candidate_pairs, ranks, how="min"):
    if how == "min":
      pair_ranks = [(pair, ranks.get(pair, float('inf'))) for pair in candidate_pairs]
      best_pair = min(pair_ranks, key=lambda x: x[1])
    elif how == "max":
      print(len(candidate_pairs))
      pair_ranks = [(pair, ranks.get(pair, float('-inf'))) for pair in candidate_pairs]
      best_pair = max(pair_ranks, key=lambda x: x[1])
    else:
      raise "Invalid 'how' argument--must be 'min' or 'max'"
    pair, rank = best_pair
    return pair, rank


  # Given pair of BPE tokens to merge, and word (as tuple), merge
  # all occurrences of the pair and return the new word.
  def _merge_bytes(self, word, pair):
      t1, t2 = pair
      new_word = []
      i = 0
      while True:
        if i == len(word) - 1:
          new_word.append(word[i])
          break
        if i >= len(word):
          break
        if word[i] == t1 and word[i + 1] == t2:
          new_word.append(t1 + t2)
          i += 2
        else:
          new_word.append(word[i])
          i += 1
      return tuple(new_word)

  # Corpus must be list of strings
  def train(self, corpus, n_merges=100, method="bpe"): 
    # in the future, add wordpiece. for now, only bpe supported
    if method != "bpe":
      raise "Method not supported."

    # Count words
    word_counter = {}
    for doc in corpus:
        words = self._pretokenize(doc)
        for word in words:
          word_counter[word] = word_counter.get(word, 0) + 1
    self.vocab = {tuple(word):count for word, count in word_counter.items()}

    # For m in n_merges, update stats, do merge
    for iter in range(n_merges):
      self._update_stats()
      if len(self.pairs.keys()) == 0:
        print("no more merges")
        break
      best_pair, count = self._get_best_pair(self.pairs.keys(), self.pairs, how="max")
      if count < 0:
        print("no more merges")
        break
      print("=== Merge", iter, "===")
      print(best_pair, "=>", "".join(best_pair))
      self.vocab = {self._merge_bytes(word, best_pair):count for word, count in self.vocab.items()}
    
    # Save resulting vocab, encoder to the class
    final_vocab = set(chain(*self.vocab.keys()))
    print(final_vocab)

  def save_model(self, merge_file, encoder_file):
    with open(encoder_file, "w") as f1:
      json.dump(self.encoder, f1)
    inverse_ranks = {v:k for k, v in self.bpe_ranks.items()}
    with open(merge_file, "w") as f2:
      for i in range(len(inverse_ranks)):
        f2.write(f"{inverse_ranks[i][0]} {inverse_ranks[i][1]}\n")   

  def load_pretrained(self, merge_file, encoder_file):
    with open(encoder_file, "r") as f1:
      self.encoder = json.load(f1)
    with open(merge_file, "r") as f2:
      bpe_merges = [tuple(x.strip().split(' ')) for x in f2.readlines()[1:]]
      self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

  # Convert byte-encoded word into space-delimited string of merged BPE tokens
  def _merge_word_bytes(self, token):
    if len(token) < 2: return token # nothing to merge

    word = tuple(token)
    pairs = self._get_pairs(word)

    # As long as merge can be performed, do merge on highest-ranking pair.
    while True:
      pair, rank = self._get_best_pair(pairs, self.bpe_ranks, how="min")
      if pair not in self.bpe_ranks:
        break # No more merges to perform

      # Update word to new word with merges
      self._merge_bytes(word, pair)
      if len(word) == 1:
          break
      else:
          pairs = self._get_pairs(word)
    
    # Separate BPE tokens with space (not used in vocab) and return
    merged = " ".join(word)
    self.cache[token] = merged
    return merged

  # Convert input string to list of integer indexes into BPE vocabulary
  def encode_string(self, text):
    as_readable_byte_words = self._pretokenize(text)
    as_bpe_tokens = [self.merge_word_bytes(tok).split(' ') for tok in as_readable_byte_words]
    integers = [self.encoder[subtok] for tok in as_bpe_tokens for subtok in tok]
    return integers


  # Convert list of integer indexes to BPE vocab back into string
  def decode_string(self, idxs):
    as_readable_bytestring = "".join([self.decoder[idx] for idx in idxs])
    as_bytes = bytearray([self.byte_decoder[b] for b in as_readable_bytestring])
    as_string = as_bytes.decode("utf-8", errors="replace")
    return as_string