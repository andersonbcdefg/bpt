# BPT – Ben's Pretrained Transformer ✨
This repository contains my implementation an autoregressive transformer language model in PyTorch, which I am working on as a personal project to familiarize myself with recent advances in deep learning. Because I'm making my own model from scratch, I was empowered to pick all my favorite transformer bells and whistles, which are detailed below. So far, I've been experimenting by training a 1.3B variant of this model on Jean Kaddour's [MiniPile](https://arxiv.org/abs/2304.08442), which contains ~1.5B tokens from The Pile. Rumor has it this dataset is not the best, but it's a convenient size for experiments! Once I've figured out a training setup I'm happy with, I'll train it on something bigger, like C4 or OpenWebText.

# References

### Relevant Papers
* [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
* [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf)

### Code References
* Andrej Karpathy's [`mingpt`](https://github.com/karpathy/minGPT): Referenced for some tricks related to implementation of multi-head attention. Also for BPE, borrowed pre-tokenization regex and mapping from bytes to characters (which in turn are borrowed from the [OpenAI implementation](https://github.com/openai/gpt-2)).
* [Einops Documentation](https://einops.rocks/pytorch-examples.html): Referenced for more tricks related to multi-head attention, namely, Einstein notation.
* Phil Wang's [ViT repository](https://github.com/lucidrains/vit-pytorch): Referenced for more attention tricks (wrapping the attention and FFN blocks in a "PreNorm" layer, which results in a much cleaner transformer block implementation).
