# GPT Implementation in PyTorch
This repository contains my implementation a GPT-like model in PyTorch, which I am working on as a personal project to familiarize myself with recent advances in deep learning. As of now, I have trained the model on a toy task (adding numbers), but will soon attempt to train it on text corpora (e.g. the complete works of Shakespeare, Wikitext, Webtext). I am also planning to implement Byte-Pair Encoding (BPE) from scratch.

# References

### Relevant Papers
* [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
* [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf)

### Code References
* Andrej Karpathy's [`mingpt`](https://github.com/karpathy/minGPT): Referenced for some tricks related to implementation of multi-head attention. Also for BPE, borrowed pre-tokenization regex and mapping from bytes to characters (which in turn are borrowed from the [OpenAI implementation](https://github.com/openai/gpt-2)).
* [Einops Documentation](https://einops.rocks/pytorch-examples.html): Referenced for more tricks related to multi-head attention, namely, Einstein notation.
* Phil Wang's [ViT repository](https://github.com/lucidrains/vit-pytorch): Referenced for more attention tricks (wrapping the attention and FFN blocks in a "PreNorm" layer, which results in a much cleaner transformer block implementation).
