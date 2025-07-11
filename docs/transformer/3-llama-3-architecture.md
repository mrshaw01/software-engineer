# Llama 3 Architecture

- https://pub.towardsai.net/build-your-own-llama-3-architecture-from-scratch-using-pytorch-2ce1ecaa901c

<div align="center">
    <img src="images/Llama3.webp"/>
    <p><em>Llama 3</em></p>
</div>

## 1. The Input Block

The input block has 3 components: Texts/ Prompts, Tokenizer and Embeddings.

<div align="center">
    <img src="images/InputBlock.webp"/>
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
    <img src="images/RMSNorm.webp"/>
    <p><em>RMSNorm</em></p>
</div>

The diagram illustrates embeddings before and after applying RMSNorm, showing reduced value ranges post-normalization.

#### Why choose RMSNorm over LayerNorm?

Unlike LayerNorm, RMSNorm does not compute the mean, reducing computational overhead. According to its authors, RMSNorm achieves similar accuracy while being more efficient.

Implementation: [RMSNorm.py](RMSNorm.py)

Here is your **professional, concise, and LaTeX-enhanced Markdown rewrite**:

### 2.2 Rotary Positional Encoding (RoPE)

#### Why do we need RoPE?

So far, we have converted input texts into embeddings and applied RMSNorm. However, without positional encoding, sentences like “I love apple” and “apple love I” would appear identical to the model, as embeddings alone lack order information. For language models, understanding token order is crucial.

In Llama 3, **Rotary Positional Encoding (RoPE)** is used to embed positional information, ensuring the model captures both **absolute** and **relative** token positions within a sequence.

#### What is RoPE and how does it work?

RoPE encodes token positions by **rotating embedding vectors** using a rotation matrix. This approach:

- Adds absolute positional information.
- Encodes relative position relationships between tokens.

Mathematically, RoPE applies a rotation transformation to each embedding dimension pair:

$$x' = R(\theta) x$$

where:

- $x$ is the embedding vector.
- $R(\theta)$ is the rotation matrix parameterized by $\theta$, derived based on token position.

<div align="center">
    <img src="images/RoPE.png"/>
    <p><em>RoPE</em></p>
</div>

- Q1 $(x_1, y_1)$: A 2D Query vector with magnitude $d$ and angle $\theta_1$ relative to the x-axis.
- Q2 $(x_2, y_2)$: Formed by rotating Q1 clockwise by $\theta_2$, resulting in a new coordinate $(x_2, y_2)$ with the same magnitude $d$.

By expressing both Q1 and Q2 in polar form and applying standard derivations, we obtain the final result as shown below.

<div align="center">
    <img src="images/RotationMatrix.png"/>
    <p><em>Rotation Matrix</em></p>
</div>

In practice:

1. **Embeddings are converted to complex form** for rotation.
2. **Rotation is applied**, encoding position information.
3. **Rotated embeddings are converted back to real form** before attention operations.

<div align="center">
    <img src="images/Rotation.png"/>
    <p><em>Rotation</em></p>
</div>

- θ = Rotation angle, this is different for each pair of embedding dimension. Total number is equal to dim/2
- m = Token's position number in sentence. For "apple", it is 1, for "is" it is 2. it will be remain same for each embedding.
- Dim = 6, this means each embedding rotate 3 times (6 is 3 pair)

**Note:** Even though the working principle is same, for better computation efficiency, the author has advised to perform calculation using the expression below.

<div align="center">
    <img src="images/RoPECalculation.png"/>
    <p><em>RoPE Calculation</em></p>
</div>

**Note:**

- RoPE is applied only to **Query** and **Key** embeddings, not to **Value** embeddings.
- While the diagram shows a 2D example, in Llama 3 RoPE operates over high-dimensional embeddings (e.g. 4096 dimensions).

This efficient positional encoding enables the model to understand token order and relationships without the overhead of traditional positional encodings.

Implementation: [RoPE.py](RoPE.py)

### 2.3 KV Cache (Inference Only)

In Llama 3, **KV Cache** stores previously generated tokens as **Key** and **Value** caches during inference. Only keys and values are cached; **queries are not cached**, hence the name KV Cache.

<div align="center">
    <img src="images/KVCache.webp"/>
    <p><em>KV Cache</em></p>
</div>

#### (A) Without KV Cache

- When generating output token 3, the model recomputes outputs for tokens 1 and 2 unnecessarily.
- This requires computing attention for all tokens, resulting in expensive matrix multiplications (e.g. $3 \times 3$ attention scores).

#### (B) With KV Cache

- Previously generated tokens are stored in the cache.
- For each new token, only its **query embedding** is processed, using cached keys and values to compute attention efficiently.
- This reduces computation from a **full sequence attention** ($3 \times 3$) to **single-step attention** ($1 \times 3$) – approximately **66% reduction** in matrix multiplications.

#### Key Benefits

- Significant computation savings, especially for long sequences and large batch sizes.
- Faster generation, as only the **latest output token** is computed at each step.

### 2.4 Grouped Query Attention

Grouped Query Attention is similar to **Multi-Head Attention (MHA)** used in models like Llama 1, with a key difference:

- MHA uses the **same number of heads** for Queries (Q), Keys (K), and Values (V).
- Grouped Query Attention uses **more heads for Queries** compared to Keys and Values.

<div align="center">
    <img src="images/GroupedQueryAttention.png"/>
    <p><em>Grouped Query Attention</em></p>
</div>

For example, in the diagram:

- **Multi-Head Attention:**

  $$n_{heads}^{Q} = n_{heads}^{K} = n_{heads}^{V} = 8$$

- **Grouped Query Attention:**

  $$n_{heads}^{Q} = 8, \quad n_{heads}^{K} = n_{heads}^{V} = 4$$

This means Query has twice as many heads as Key/Value.

While Multi-Head Attention is effective, it scales up memory usage significantly when combined with **KV Cache** (which stores all previous Key and Value tokens). As sequence length grows, so does the memory footprint.

**Grouped Query Attention addresses this by:**

- **Reducing the number of K/V heads**, thus lowering the total cached parameters and memory requirements.
- Maintaining similar accuracy, as validated by empirical results.

Implementation: [Attention.py](Attention.py)

### 2.5 FeedForward Network (SwiGLU Activation)

After attention outputs are normalized by RMSNorm, they are passed into the **FeedForward Network (FFN)**. This network:

- Expands the embeddings to higher dimensions through its hidden layers.
- Learns more **complex token features** to enhance model representation capacity.

<div align="center">
    <img src="images/SwiGLU.png"/>
    <p><em>SwiGLU</em></p>
</div>

The diagram shows the difference between ReLU and SwiGLU activations:

- **ReLU:** Outputs zero for negative inputs, potentially losing negative information.
- **SwiGLU:** Outputs small negative values for negative inputs, enabling richer learning.

$$FFN_{swiGLU}(x, W, V, W_2, \beta) = (Swish(xW) \otimes xV) W_2$$

where:

$$Swish(x) = x \cdot \sigma(\beta x) = \frac{x}{1 + e^{-\beta x}}$$

If $\beta = 1$, Swish simplifies to **SiLU**:

$$SiLU(x) = \frac{x}{1 + e^{-x}}$$

Hence, **SiLU is used in practice** for computational efficiency.

#### **Key takeaway**

SwiGLU enhances the model by allowing negative outputs, improving learning dynamics compared to ReLU, and has been empirically shown to perform better in transformer architectures like Llama 3.

Implementation: [FeedForward.py](FeedForward.py)

### 2.6 Decoder Block

As shown in the architecture diagram, the **decoder block** consists of several sub-components described in Sections 2.1–2.5. The operations within each decoder block are as follows:

1. **Input embedding** is passed through the **Attention-RMSNorm block**, then fed into the **Group Query Attention** block.
2. The **attention output** is **added to the original input embedding** (residual connection).
3. The result is then passed through **FeedForward-RMSNorm**, followed by the **FeedForward network** block.
4. The **FeedForward output** is **added to the attention output** (residual connection).
5. This final result is the **Decoder Output**.

The **Decoder Output** becomes the input to the next decoder block. This process repeats across all **32 decoder blocks**, with the final output from the last block passed to the **Output block** for prediction.

Implementation: [TransformerBlock.py](TransformerBlock.py)

## 3. The Output Block

The **decoder output from the final decoder block** feeds into the output block. The process is:

1. The output is first passed through **RMSNorm**.
2. It is then fed into a **Linear Layer**, producing logits.

- **Inference Mode**

<div align="center">
    <img src="images/Inference.png"/>
    <p><em>Inference</em></p>
</div>

- The logits are used to calculate **top-p probabilities**.
- The next token is generated based on these probabilities.
- Generation continues until either the maximum length is reached or an end-of-sentence token is produced.

- **Training Mode**

<div align="center">
    <img src="images/Training.png"/>
    <p><em>Training</em></p>
</div>

- The logits are compared with **target labels** to compute the loss using a cross-entropy function.
- Training repeats for the defined number of epochs.

Finally, combining the **Input Block**, **Decoder Blocks**, and **Output Block** forms the complete Llama 3 model.

Implementation: [Transformer.py](Transformer.py)

Here is your **professional, concise rewrite**:

## 4. Training the Llama 3 Model

The training flow is illustrated in the **Output Block flow diagram (Step 3)**. Refer to it for a clear understanding before proceeding.

Implementation: [training.py](training.py)

<div align="center">
    <img src="images/LossGraph.webp"/>
    <p><em>Loss Graph</em></p>
</div>

The graph above shows the training and validation loss over 2500 epochs. Training was completed in approximately 10 minutes on Google Colab using the default GPU and RAM, which is notably fast. The final validation loss reached 2.19, which is acceptable given the limited training data and number of epochs. To further reduce the loss, we would need a larger dataset, more training epochs, and greater computational resources.

## 5. Inference the Llama 3 Model

Implementation: [inference.py](inference.py)

Our Llama 3 model can now perform inference and generate text for new prompts. While the output quality is limited due to the small training dataset and few epochs, using larger datasets and longer training will significantly improve accuracy.
