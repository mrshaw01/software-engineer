# Part 3: Training with Mixed Precision Using FP16

## 3.1. When to Consider Mixed Precision with FP16

When your deep learning framework supports Tensor Core operations, simply enabling mixed precision often accelerates training with minimal effort. Many networks can be trained faster by switching tensor data types and layer computations to FP16, without changing hyperparameters from the original FP32 training setup.

However, some models—especially those with small gradient magnitudes—may require scaling to map those values into FP16’s limited representable range. Without this, the gradients may underflow to zero, leading to unstable or failed training.

### Histogram Example

In the Multibox SSD network (VGG-D backbone), histograms of gradient magnitudes collected during FP32 training show:

- **66.8% of gradient values are 0**
- **4% lie between 2⁻³² and 2⁻³⁰**
- **Converting to FP16 results in 31% of values becoming 0**, leaving only 5.3% as non-zero, which caused training to diverge

### Solution: Gradient Shifting

To address this:

- Multiply gradients by 8 (shift by 3 exponent values): sufficient to match FP32 accuracy
- Multiply by 32K (shift by 15): recovers all but 0.1% of zeros without overflow

**Key Insight**: FP16 has sufficient dynamic range for training, but gradients must often be scaled into that range.

## 3.2. Loss Scaling to Preserve Small Gradients

To prevent small gradients from vanishing in FP16:

- **Scale the loss value** during the forward pass by a factor **S**
- Due to the chain rule, all gradients are automatically scaled by **S**
- After backpropagation, **unscale the gradients** (multiply by 1/S) before updating weights

### Recommended Training Procedure

1. **Maintain FP32 master copy of weights**
2. For each iteration:
   - Cast weights to FP16
   - Forward pass with FP16 weights and activations
   - Multiply loss by scaling factor **S**
   - Backward pass with scaled gradients in FP16
   - Unscale gradients (multiply by **1/S**)
   - Apply gradient clipping, weight decay, etc.
   - Update FP32 weights

### Note on Reductions and Batch Norm

Some operations require higher precision:

- Keep **statistics** (e.g., mean/variance in batch norm) in FP32
- Compute **reductions** like SoftMax and normalization in FP32
- Inputs/outputs can remain in FP16 to save bandwidth

## 3.3. Choosing a Scaling Factor

### Constant Loss Scaling

You can use a fixed scaling factor **S**, selected via:

- **Trial and error**
- **Gradient statistics** (e.g., ensure `max_gradient × S < 65,504`)

Trained models have used values from **8 to 32,000**, depending on architecture and batch size.

### Dynamic Loss Scaling

A more robust method dynamically adjusts **S** during training:

#### Procedure:

1. Initialize scaling factor **S** to a large value
2. For each iteration:
   - Perform forward and backward passes with scaled loss
   - Check for NaNs/Infs in gradients
   - If overflow is detected:
     - Reduce **S**
     - Skip weight update
   - If no overflow for **N** iterations:
     - Increase **S**
   - Otherwise, unscale gradients and apply weight update

#### Example Parameters:

- `N = 2000`
- Increase S by ×2 after N successful iterations
- Decrease S by ×0.5 on overflow

This adaptive method avoids frequent overflows while preserving training stability and accuracy.

## Summary

Training with FP16 offers large performance and memory advantages, but requires:

- Proper **loss scaling** to prevent gradient underflow
- Strategic use of **FP32 for certain operations**
- Either **manual** or **dynamic scaling** of gradients
