# 8. Speculative Decoding

Speculative decoding is an inference-time optimization technique that reduces latency by generating multiple tokens in parallel using a lightweight draft model and then verifying them with a larger, more accurate target model. It is particularly useful for accelerating autoregressive large language model (LLM) inference without modifying the target model’s architecture or training.

## Motivation

Autoregressive LLMs decode one token at a time, leading to high latency—especially for large models like GPT-3/4. However, many of the next tokens can be predicted with high confidence by smaller models. Speculative decoding leverages this property to increase throughput while maintaining generation quality.

## Core Components

### 1. **Draft Model**

- A small, fast model (e.g., distilled version of the target model).
- Generates multiple tokens (e.g., 4–16) in one shot.
- Does not require training modifications if it approximates the target well.

### 2. **Target Model**

- The large, accurate model whose quality you want to retain.
- Verifies the tokens proposed by the draft model.
- If verification fails, it rolls back to the last verified token.

## Algorithm Steps

1. **Prefill**:

   - Both models receive the same initial context.
   - The target model computes and caches KV pairs.

2. **Drafting**:

   - The draft model generates `n` tokens using greedy or sampling strategy.
   - These tokens are proposed as a speculative sequence.

3. **Verification**:

   - The target model verifies the entire draft in parallel by computing the logits for each draft token position.
   - If the top-1 token from the target model matches the draft token, it is accepted.

4. **Fallback / Rollback**:

   - If the target model disagrees at token `t`, decoding is rolled back to token `t-1`.
   - The verified tokens are committed, and the next draft starts from there.

5. **Repeat**:

   - The process continues until `max_tokens` or an end-of-sequence token is reached.

## Pseudocode

```python
def speculative_decode(draft_model, target_model, context, max_tokens):
    generated = context[:]
    while len(generated) < max_tokens:
        # Draft step
        draft_tokens = draft_model.generate(generated, num_tokens=n)

        # Verification step
        verified = []
        logits = target_model.forward(generated + draft_tokens[:-1])
        for i, token in enumerate(draft_tokens):
            if argmax(logits[i]) == token:
                verified.append(token)
            else:
                break  # Rollback if disagreement occurs

        if not verified:
            # Target generates one token as fallback
            next_token = argmax(target_model.forward(generated)[-1])
            verified = [next_token]

        generated.extend(verified)
    return generated
```

## Benefits

- **Latency Reduction**: Generates multiple tokens per verification step.
- **Throughput Increase**: Particularly effective in batch or streaming settings.
- **Plug-and-Play**: Does not require changing the target model or retraining.

## Design Trade-offs

| Factor           | Draft Model     | Target Model  |
| ---------------- | --------------- | ------------- |
| Speed            | Fast            | Slow          |
| Accuracy         | Lower           | High          |
| Compute Cost     | Low             | High          |
| Role in Decoding | Generate drafts | Verify drafts |

- **Batch size** and **number of draft tokens** must be tuned.
- More draft tokens = higher risk of rollback.
- Too few = low speedup.

## Implementation Notes

- Requires both models to support batched generation and verification.
- KV caching must be managed independently for both models.
- Works best when the draft model is architecturally similar to the target.

## Example: vLLM Integration

In vLLM:

- The draft model uses FlashAttention2 and grouped KV cache.
- Target model verifies with CUDA graphs for fast re-execution.
- Token streaming is preserved even with speculative decoding.

## References

- [Chen et al., 2023. Accelerating LLMs with Speculative Decoding](https://arxiv.org/abs/2302.01318)
- [OpenAI's Draft-and-Verify](https://openai.com/research/accelerating-gpts-with-speculative-decoding)
- [vLLM Implementation](https://github.com/vllm-project/vllm)
