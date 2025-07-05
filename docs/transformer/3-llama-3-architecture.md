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

## 2. The Decoder Block

### 2.1 RMSNorm (Root Mean Square Normalization)

#### Why RMSNorm?

As shown in the diagram, the embedding vector output from the input block passes through an RMSNorm block. Embedding vectors have high dimensionality (e.g. 4096 in Llama 3-8B), resulting in values with varying scales. Without normalization, this can lead to gradient explosion or vanishing, causing slow convergence or divergence during training. RMSNorm scales these values to a consistent range, stabilizing gradients and improving training speed.

#### How does RMSNorm work?

RMSNorm operates similarly to LayerNorm but with simplified computation:

- It normalizes along the embedding dimension.
- For each token embedding (e.g. $X_1$ with dimensions $x_{11}, x_{12}, x_{13}$), each value is divided by the root mean square (RMS) of all dimensions plus a small epsilon $\epsilon$ for numerical stability:

$$RMS(x) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2 + \epsilon}$$

- The normalized value is then scaled by a learnable parameter $\gamma$, initialized to 1:

$$RMSNorm(x_i) = \frac{x_i}{RMS(x)} \times \gamma_i$$

where each dimension has its own $\gamma_i$.

<div align="center">
    <img src="images/RMSNorm.webp" alt="RMSNorm" title="RMSNorm"/>
    <p><em>RMSNorm</em></p>
</div>

The diagram illustrates embeddings before and after applying RMSNorm, showing reduced value ranges post-normalization.

#### Why choose RMSNorm over LayerNorm?

Unlike LayerNorm, RMSNorm does not compute the mean, reducing computational overhead. According to its authors, RMSNorm achieves similar accuracy while being more efficient.

Implementation: [RMSNorm.py](RMSNorm.py)
