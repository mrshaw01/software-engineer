# What Is Inference in LLMs?

Inference in large language models (LLMs) refers to the process of generating output text given an input prompt. It is the runtime phase where a trained model is used to produce predictions, typically in the form of tokens, using its learned parameters. This is distinct from training, where model weights are updated.

## Inference Workflow

Inference can be broken down into the following high-level stages:

1. **Tokenization**
   The input string is tokenized into integer IDs using the model's tokenizer (e.g., Byte Pair Encoding, SentencePiece). These token IDs represent the context to condition the model.

2. **Prefill (Context Embedding)**
   The initial input tokens are embedded and passed through all transformer layers in a single forward pass. The key/value (KV) pairs from each attention layer are cached to avoid recomputation.

3. **Decoding (Autoregressive Generation)**
   The model generates one token at a time, feeding its own output as the next input. Each decoding step uses the cached KV pairs from previous steps to compute the next token efficiently.

4. **Detokenization**
   The output token IDs are converted back into text using the tokenizer's vocabulary.

## Decoding Strategies

Inference performance and quality are influenced by the decoding method:

- **Greedy decoding**: Selects the most probable token at each step.
- **Sampling**: Draws tokens based on probability distribution (with or without temperature/top-k/top-p).
- **Beam search**: Explores multiple hypotheses and selects the best sequence.
- **Speculative decoding**: Uses a smaller draft model to propose multiple tokens and verifies them using the full model.

## Goals of Inference

- **Latency**: Reduce time per generated token (e.g., milliseconds/token).
- **Throughput**: Maximize tokens/second across concurrent requests.
- **Scalability**: Serve multiple users/models efficiently.
- **Correctness**: Ensure deterministic output when required (e.g., for testing or reproducibility).

## Practical Considerations

- Inference is compute-bound and memory-intensive, especially for long contexts.
- Optimizations such as KV cache reuse, quantization, and efficient batching are critical for production deployment.
- Token generation is inherently sequential, but system-level parallelism (e.g., across requests) is essential to reach high throughput.

## Summary

Inference in LLMs is an autoregressive, sequential process that transforms input prompts into token outputs using a trained transformer model. Understanding the inference lifecycle—from tokenization to output generation—is foundational for optimizing performance and building scalable LLM-serving systems.
