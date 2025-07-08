# DeepSeek-V3: Contributions and Evaluation Results

## Main Contributions

### 1. Architecture: Innovative Load Balancing and Training Objective

- **Auxiliary-Loss-Free Load Balancing**
  - Pioneered strategy minimizes performance degradation typically caused by load balancing efforts.
- **Multi-Token Prediction (MTP) Objective**
  - Enhances model performance.
  - Enables speculative decoding for faster inference.

### 2. Pre-Training: Ultimate Training Efficiency

- **FP8 Mixed Precision Training Framework**
  - First large-scale validation of FP8 training on extremely large models.
- **Algorithm-Framework-Hardware Co-Design**
  - Achieves near-full computation-communication overlap.
  - Overcomes cross-node MoE training communication bottleneck.
  - Enables further model scaling without additional overhead.
- **Training Cost Efficiency**
  - **Pre-training**: 2.664M H800 GPU hours on 14.8T tokens.
  - Produces the strongest open-source base model.
  - **Subsequent stages** (context extension + post-training): Only ~0.1M GPU hours.

### 3. Post-Training: Knowledge Distillation from DeepSeek-R1

- **Methodology**:
  - Distills reasoning capabilities from long Chain-of-Thought (CoT) models (DeepSeek-R1 series) into DeepSeek-V3.
  - Incorporates R1’s verification and reflection patterns.
  - Enhances reasoning performance while maintaining output style and length control.

## Summary of Core Evaluation Results

### Knowledge

1. **Educational Benchmarks**:
   - **MMLU**: 88.5
   - **MMLU-Pro**: 75.9
   - **GPQA**: 59.1
   - Outperforms all open-source models.
   - Comparable to GPT-4o and Claude-Sonnet-3.5, narrowing the open-source vs closed-source gap.
2. **Factuality Benchmarks**:
   - Best open-source performance on **SimpleQA** and **Chinese SimpleQA**.
   - Slightly behind GPT-4o and Claude-Sonnet-3.5 in **English factual knowledge** (SimpleQA).
   - **Surpasses** these models in **Chinese factual knowledge** (Chinese SimpleQA).

### Code, Math, and Reasoning

1. **Math-Related Benchmarks**:
   - State-of-the-art among non-long-CoT models (open-source and closed-source).
   - Outperforms o1-preview on benchmarks like **MATH-500**, showcasing strong mathematical reasoning.
2. **Coding Tasks**:
   - Top-performing model on **coding competition benchmarks** (e.g. LiveCodeBench).
   - Leading model in coding domains.
3. **Engineering Tasks**:
   - Slightly below Claude-Sonnet-3.5.
   - Outpaces all other models by a significant margin.

> **Key Takeaway**: DeepSeek-V3 combines **innovative architectural strategies, efficient FP8 training, and advanced knowledge distillation**, resulting in **state-of-the-art open-source performance across knowledge, code, math, and reasoning benchmarks**.
