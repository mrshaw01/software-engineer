# Mixed-Precision Training Scenarios on an NPU (BF16 + FP32)

Modern NPUs natively accelerate both **bfloat16 (BF16)** and **float32 (FP32)**.
The sweet spot for speed, memory, and numerical stability is usually a _hybrid_ approach: run most math in BF16, but keep a few critical quantities in FP32.

## 1 Common Scenarios

| #     | Scenario                                       | Runs in **BF16**                           | Remains in **FP32**                                                             | When to Use                                                    | Key Points                                                    |
| ----- | ---------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------- |
| **1** | **Pure BF16 compute + FP32 master weights**    | All forward/backward ops, weight gradients | Master parameters for the optimizer, global reductions (loss, batch-norm stats) | _Default_ for CNNs/Transformers                                | Fastest path; usually no loss scaling needed.                 |
| **2** | **BF16 compute with FP32 accumulation**        | MatMul operands and partial products       | Dot-product accumulations (implicit widening)                                   | Attention/MLP blocks on hardware that auto-accumulates to FP32 | Accuracy close to FP32 while keeping BF16 memory footprint.   |
| **3** | **BF16 everywhere except last layers**         | Early & middle layers                      | Final classifier / logits / SoftMax                                             | Output logits are tiny → risk of underflow                     | Toggle final `n` layers to FP32 only.                         |
| **4** | **BF16 activations, FP32 weights**             | Activations, intermediate tensors          | Weights, optimizer states                                                       | Activation-heavy models where RAM is the bottleneck            | Cuts activation memory in half without touching weight maths. |
| **5** | **BF16 compute, FP32 for normalization stats** | Convs / MLPs / attention                   | Mean & variance (BatchNorm / LayerNorm)                                         | Vision models with heavy normalization                         | Stats are tiny; FP32 cost negligible, stability gained.       |
| **6** | **Dynamic loss scaling (rare with BF16)**      | Same as Scenario 1                         | Same as Scenario 1 plus scaling factor `S`                                      | Extremely sparse or tiny gradients                             | BF16 usually avoids scaling, but keep a hook ready.           |

## 2 Practical Guidelines

1. **Start with Scenario 1**
   Enable BF16 tensors and kernels; keep an FP32 master copy of weights. Most NPUs are tuned for this pattern.

2. **Watch for NaN/Inf**
   _No NaNs → stick with pure BF16._
   NaNs only in the last layer? Switch to Scenario 3.
   Drift in batch-norm stats? Adopt Scenario 5.

3. **Memory Budgeting**
   _Activation-bound_ models ⇒ Scenario 4
   _Weight-bound_ models ⇒ Scenario 1

4. **Optimizer States**
   Moment buffers (Adam, Lion, etc.) are best stored in FP32—small overhead, big stability win.

5. **Reductions & Norms**
   Always accumulate large reductions (e.g., global mean) in FP32. Frameworks usually expose a `dtype="float32"` flag.

6. **Loss Scaling**
   Almost exclusive to FP16. Use only if BF16 gradients still underflow.

## 3 Decision Flow

```text
Does training diverge when everything is BF16?
│
├─ No → Use Scenario 1 and enjoy the speed-up.
│
└─ Yes
    ├─ Only final logits unstable → Scenario 3
    ├─ BatchNorm stats drift     → Scenario 5
    └─ Gradients too small       → Enable dynamic loss scaling (Scenario 6)
```

## 4 Takeaways

- **BF16 + FP32 master weights** is the simplest and fastest starting point.
- Promote _only_ those tensors that empirically need higher precision.
- Dynamic loss scaling is a last-resort tool; BF16’s wide range usually makes it unnecessary.

Adopt, test, and iterate—mixed precision on NPUs can deliver _FP32-level accuracy_ with **significant** gains in speed and memory efficiency.
