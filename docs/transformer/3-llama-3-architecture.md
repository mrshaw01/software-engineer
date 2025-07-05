# Llama 3 Architecture

- https://pub.towardsai.net/build-your-own-llama-3-architecture-from-scratch-using-pytorch-2ce1ecaa901c

<div align="center">
    <img src="images/Llama3.webp" alt="Llama 3" title="Llama 3"/>
    <p><em>Llama 3</em></p>
</div>

## 1. The Input Block

The input block has 3 components: Texts/ Prompts, Tokenizer and Embeddings.

<div align="center">
    <img src="images/InputBlock.webp" alt="Input Block" title="Input Block"/>
    <p><em>Input Block</em></p>
</div>

First, a single text or a batch of prompts (e.g., “Hello World”) is passed to the model. Since the model processes numerical inputs, a tokenizer converts the text into token IDs, representing each token’s index in the vocabulary. We will build this vocabulary and tokenizer ourselves using the Tiny Shakespeare dataset to understand encoding, decoding, and implementation details fully.

While Llama 3 uses TikToken, a subword tokenizer, we will implement a character-level tokenizer to gain complete insight and control. Each token ID is then mapped to an embedding vector of dimension 128 (4096 in Llama 3 8B) before being fed into the Decoder Block.

Implementation: [input_block.py](input_block.py)
