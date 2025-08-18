# Weight-Only Quantization (W8A16, W4A16)

This means:

- **Weights are quantized** (INT8 or INT4, symmetric/asymmetric).
- **Activations remain FP16/FP32** (no quantization/dequant overhead in forward pass).
- **Accumulation stays FP16/BF16/FP32** (for stability).

## 1. Preparation / Calibration (common for CPU & GPU)

1. **Collect representative data**

   - Run 128–512 calibration samples (short prompts or task-specific data).
   - Collect per-channel or per-group statistics of weights.

2. **Compute scales & zero-points**

   - Symmetric (common for matmuls):

     $$q_w = \text{round}\left(\frac{w}{s}\right)$$

     where $s = \frac{\max(|w|)}{Q_{\max}}$.

   - Asymmetric (if distribution skewed):

     $$q_w = \text{round}\left(\frac{w - z}{s}\right)$$

     with zero-point $z$.

3. **Quantize weights offline**

   - Store quantized weights (INT8/INT4) + scales (FP16/FP32).
   - Save in a quantized checkpoint.

4. **Choose granularity**

   - Per-tensor (fast, worse quality).
   - Per-channel / per-group (better accuracy, slightly more storage for scales).

## 2. Inference Execution (CPU vs GPU)

### A) CPU Execution Path

- **Dequantize on the fly** or fuse dequant into GEMM:

  $$y = (q_w \cdot s) \times x$$

  - `q_w`: quantized weight matrix (INT8/INT4).
  - `s`: scale (per-channel/group).
  - `x`: FP16/FP32 activation.
  - `y`: FP32/FP16 output.

- **Implementation**:

  - Use INT8/INT4 → FP16 kernels in BLAS (e.g., FBGEMM, oneDNN).
  - Typical flow:

    1. Load quantized weights (INT8).
    2. Convert to FP16 via scale \* int8.
    3. Multiply with FP16 activations.
    4. Accumulate in FP16/FP32.

✅ Benefit: 2–4× lower memory footprint, CPU cache-friendly.
⚠️ Overhead: Dequant adds a few % runtime cost, but bandwidth savings dominate.

### B) GPU Execution Path

- **Same math** as CPU, but fused inside CUDA/HIP kernels.
- Avoid explicit dequant → instead, multiply int8 + scale \* activation directly inside GEMM.

**Minimal GPU kernel pseudocode (INT8 weight, FP16 activation):**

```cpp
// Each thread computes a dot product between
// int8 weight row and fp16 activation vector
__global__ void gemm_w8a16(
    const int8_t* __restrict__ Wq,  // quantized weights
    const half*  __restrict__ X,    // activations
    const float* __restrict__ scales, // per-channel scale
    half*        __restrict__ Y,    // output
    int M, int K, int N)            // GEMM sizes
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // output row
    int col = blockIdx.x * blockDim.x + threadIdx.x; // output col

    if (row < M && col < N) {
        float acc = 0.0f;
        float scale = scales[row];  // per-output-channel scale

        for (int k = 0; k < K; k++) {
            int8_t qw = Wq[row * K + k];
            half   x  = X[k * N + col];
            acc += (float)qw * __half2float(x) * scale;
        }
        Y[row * N + col] = __float2half(acc);
    }
}
```

- **Optimized kernels** (real systems):

  - Use tensor cores with INT8 MMA instructions (NVIDIA: `mma.sync`, AMD: MFMA).
  - Pack 4×INT4 or 2×INT8 per register.
  - Use shared memory tiling, warp-level MMA.
  - Scales are often pre-multiplied into the dequant path.

✅ Benefit: Minimal overhead vs FP16 GEMM.
⚠️ Need tuned kernels for high throughput.

## 3. Runtime Considerations

- **CPU**:

  - Rely on vectorized INT8 GEMM libraries (oneDNN, FBGEMM).
  - Best when batch size small/medium (low concurrency).

- **GPU**:

  - Rely on fused INT8 Tensor Core kernels (cuBLASLt, CUTLASS, Triton, ROCm).
  - Best when batch size large (maximize parallelism).

- **Fallbacks**: keep fragile layers (e.g., first, last, layernorms) in FP16.

## 4. Validation

- Run perplexity eval + task probes (MMLU, LAMBADA).
- Compare vs FP16 baseline.
- Expect <0.5 pt perplexity delta (INT8) and \~1–2 pt (INT4).

**Summary**:

- **On CPU**: Offline weight quant → load INT8/INT4 + scales → BLAS dequant-fused GEMM.
- **On GPU**: Same principle but inside Tensor Core kernels; INT8 weight loads, fused scale multiply, FP16 accumulation.
- **Kernel**: Dot product of INT8 × FP16 fused with scale (see pseudocode above).
