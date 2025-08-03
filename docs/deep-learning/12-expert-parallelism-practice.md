# Expert Parallelism with Mixture-of-Experts (MoE)

Expert parallelism is a key technique in **Mixture-of-Experts (MoE)** models that enables efficient scaling of large neural networks. By distributing multiple "experts" across devices and dynamically selecting which experts to activate per input token, MoE achieves **sparse computation** while maintaining high model capacity.

This approach is commonly used in large language models (LLMs) like **Switch Transformer**, **GLaM**, and **DeepSpeed-MoE**.

## Key Concepts

- **Experts**: Independent feed-forward networks (FFNs) specialized to handle different parts of the input space.
- **Router (Gating Network)**: A lightweight network that assigns tokens to experts based on learned routing scores.
- **Top-k Selection**: Instead of activating all experts, only the top-k experts (e.g., k=1 or k=2) are selected per token.
- **Expert Parallelism**: Experts are distributed across devices; tokens are routed to remote devices when needed.
- **Load Balancing**: Techniques like auxiliary losses or stochastic routing ensure experts receive balanced workloads.

## Extended Pseudo-Code Example

Below is an extended example that supports **top-k routing**, **load balancing loss**, and **expert parallelism hooks** for distributed training.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class SwitchFFNLayer(nn.Module):
    def __init__(self, num_experts, hidden_dim, ffn_dim, k=1):
        super().__init__()
        self.k = k  # top-k experts per token
        self.router = nn.Linear(hidden_dim, num_experts)
        self.experts = nn.ModuleList([FFN(hidden_dim, ffn_dim) for _ in range(num_experts)])

    def forward(self, x):
        gate_logits = self.router(x)                       # [batch, seq_len, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)
        topk_vals, topk_indices = torch.topk(gate_probs, self.k, dim=-1)  # [batch, seq_len, k]

        outputs = torch.zeros_like(x)
        load_balance_loss = (gate_probs.mean(dim=0).var()).mean()  # Encourage balanced routing

        for i, expert in enumerate(self.experts):
            mask = (topk_indices == i).any(dim=-1)
            if mask.any():
                selected_tokens = x[mask]
                # Simulate expert parallelism: send to remote device if needed
                processed = expert(selected_tokens)
                outputs[mask] = processed

        return outputs, load_balance_loss
```

## Example Usage

```python
batch_size, seq_len, hidden_dim, ffn_dim, num_experts = 4, 16, 512, 2048, 8
layer = SwitchFFNLayer(num_experts, hidden_dim, ffn_dim, k=2)
x = torch.randn(batch_size, seq_len, hidden_dim)

outputs, lb_loss = layer(x)
print("Output shape:", outputs.shape)
print("Load balance loss:", lb_loss.item())
```

## Expert Parallelism in Distributed Training

In large-scale training, experts are **sharded across multiple GPUs or nodes**. Tokens are routed via **all-to-all communication** so that each token reaches its assigned expert.

### High-Level Steps:

1. **Local Routing:** Compute routing probabilities locally.
2. **Token Dispatch:** Perform `all_to_all` communication to send tokens to their expert devices.
3. **Expert Processing:** Each device processes only the tokens assigned to its experts.
4. **Token Combine:** Perform `all_to_all` communication to gather processed tokens back to their original positions.

## Optimizations

- **Top-1 vs Top-2 Routing:** Top-1 reduces compute but may harm model quality; top-2 is a good trade-off.
- **Load Balancing Auxiliary Loss:** Prevents expert under-utilization.
- **Capacity Factor:** Limit tokens per expert to avoid overload.
- **Overlapping Communication and Computation:** For efficiency in distributed setups.

## References

- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)
- [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905)
- [DeepSpeed-MoE](https://www.deepspeed.ai/tutorials/mixture-of-experts/)
