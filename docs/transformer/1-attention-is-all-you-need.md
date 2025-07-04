# Attention Is All You Need

## Abstract

In "Attention Is All You Need," the authors point out that most sequence transduction models—like those for machine translation—have relied on complex recurrent or convolutional neural networks, typically with both an encoder and a decoder. The best-performing models before this work used an attention mechanism to connect these two components.

The key contribution of the paper is the **Transformer** architecture, which relies entirely on attention mechanisms, removing the need for recurrence or convolutions. According to the authors, this design leads to models that are not only higher in quality, but also more parallelizable and significantly faster to train.

Their experiments on two machine translation tasks show that the Transformer achieves strong results:

- 28.4 BLEU on WMT 2014 English-to-German, surpassing previous state-of-the-art (even ensembles) by more than 2 BLEU.
- 41.8 BLEU on WMT 2014 English-to-French, setting a new single-model record with just 3.5 days of training on eight GPUs—much less compute than earlier approaches.

The authors also report that the Transformer generalizes well to other tasks, such as English constituency parsing, with both large and small datasets.

## Introduction

The authors start by noting that recurrent neural networks (RNNs), particularly LSTMs and GRUs, have been the dominant approach for sequence modeling and transduction problems like language modeling and machine translation. Many advances have been made by improving these recurrent models and encoder-decoder frameworks.

However, RNNs process sequences one step at a time, which makes parallelization difficult—especially for long sequences. While some recent work has improved efficiency through various tricks and optimizations, the sequential computation bottleneck remains.

Attention mechanisms have become an important part of modern sequence models, since they allow the model to directly relate different positions in the input and output, regardless of distance. However, most attention-based models still rely on a recurrent backbone.

In this paper, the authors introduce the **Transformer**: a model architecture that eliminates recurrence and relies entirely on attention mechanisms to model dependencies between inputs and outputs. This design enables much greater parallelization and, according to their experiments, can reach state-of-the-art translation quality after as little as twelve hours of training on eight P100 GPUs.

## 2. Background

The authors emphasize that reducing sequential computation has been a major goal in previous sequence modeling research. They highlight several earlier models—**Extended Neural GPU**, **ByteNet**, and **ConvS2S**—which all use convolutional neural networks to compute hidden representations in parallel across all input and output positions.

However, in these convolutional models, the number of operations needed to connect information between any two positions still grows with the distance between them (linearly in ConvS2S, logarithmically in ByteNet). This scaling makes it harder for the model to learn dependencies between distant positions. The Transformer addresses this by reducing the number of operations needed to connect any two positions to a constant, regardless of distance. This improvement comes with a potential trade-off: the effective resolution can be reduced due to the averaging effect of attention over many positions, but the authors address this limitation using Multi-Head Attention (explained in section 3.2 of the paper).

The authors also discuss **self-attention** (or intra-attention), which relates different positions within a single sequence to compute its representation. Self-attention has already been successfully applied to tasks like reading comprehension, abstractive summarization, textual entailment, and learning general-purpose sentence representations.

Another related architecture is the end-to-end memory network, which uses a recurrent attention mechanism (rather than standard sequence-aligned recurrence) and has performed well in question answering and language modeling.

According to the authors, the Transformer is the first transduction model to rely entirely on self-attention for computing input and output representations, without using any sequence-aligned RNNs or convolutions. In the following sections, they go on to describe the Transformer architecture in detail, explain the motivation behind self-attention, and discuss the advantages of this approach over previous models like ConvS2S and ByteNet.

## 3. Model Architecture

The authors explain that most competitive sequence transduction models use an encoder-decoder structure. In this setup, the encoder takes an input sequence of symbols $(x_1, ..., x_n)$ and maps it to a sequence of continuous representations $\mathbf{z} = (z_1, ..., z_n)$. The decoder then generates an output sequence $(y_1, ..., y_m)$, producing one symbol at a time. The process is auto-regressive: at each decoding step, the model conditions on all previously generated outputs.

The Transformer adopts this encoder-decoder framework, but replaces recurrence and convolutions with **stacked self-attention** and **point-wise, fully connected layers** for both the encoder and decoder. The figure below, reproduced from the paper, illustrates the overall architecture:

<div align="center">
    <img src="images/transformer.png" alt="Transformer Model Architecture" title="Transformer Model Architecture"/>
    <p><em>Transformer Model Architecture</em></p>
</div>

The left half of the diagram shows the encoder, while the right half shows the decoder. Both are composed of multiple layers. The encoder uses self-attention and feed-forward layers, while the decoder includes masked self-attention, encoder-decoder attention, and feed-forward layers. Positional encoding is added to the input embeddings to inject order information. The final output is passed through a linear layer and a softmax to produce probabilities over the target vocabulary.
