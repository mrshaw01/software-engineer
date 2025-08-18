# Weights + Activations quantized (fast prefill)** with **INT8 matmuls

# 0) Goals & when to use B

- **Target:** W8A8 GEMMs with per-channel/group scales and outlier handling (SmoothQuant/AWQ-style).
- **Why:** **1.3–1.8× prefill speedup** (QKV/MLP matmuls run in INT8) with small quality loss when calibrated; decode can optionally keep A16 to avoid extra Q/DQ overhead.

# 1) Scope: what to quantize vs keep in FP16/BF16

Quantize to INT8:

- Attention projections: **Q, K, V, O** linears
- MLP **gate/up/down** linears
- (Optional) Embedding matmul in prefill if kernel supports it

Keep in FP16/BF16:

- **LayerNorm/RMSNorm**, **rotary** phases, **softmax**, **residual adds**
- **lm_head** often FP16 (or W8) for quality
- Final logits scaling/softmax

# 2) Calibration & model transform (shared CPU/GPU)

## 2.1 Collect activation stats (128–512 short prompts)

- Hook after norms and before each quantized linear’s input.
- Record per-channel/group statistics (e.g., max/percentile or MSE fit).

```python
# PyTorch-ish sketch
acts = {name: []}
def hook(name):
    def _f(x):
        with torch.no_grad():
            # per-channel over feature dim
            a = x.detach().float()
            # Example: take max over batch*seq; keep per-channel (dim=-1)
            m = a.abs().amax(dim=(0,1))
            acts[name].append(m)
        return x
    return _f
```

## 2.2 SmoothQuant equalization (shift outliers from activations into weights)

For each linear `y = x W^T + b` (in_features=C, out_features=F), choose a diagonal scaling `D = diag(d_j)` on the input channels to “flatten” activation ranges:

- Offline transform: `W' = W · D^{-1}`, `b' = b`
- Runtime: `x' = x · D`
- Then quantize both `W'` and `x'` to INT8.

One simple recipe (there are many):

- From calib stats, get per-channel activation scale `a_j = percentile(|x_j|, 99.9)`
- Optionally also weight per-channel scale `w_j = percentile(|W[:,j]|, 99.9)`
- Choose `d_j = (a_j ** α) / (w_j ** (1-α))` with `α ∈ [0.5, 0.9]` (search per layer).
- Clip `d_j` to a reasonable range (e.g., \[1/32, 32]) to avoid extremes.

## 2.3 Choose quantization granularity

- **Weights (W8):** per-output-channel or small **group-wise** (e.g., 64) scales (symmetric, zp=0).
- **Activations (A8):** per-input-channel or group-wise scales (symmetric usually; asymmetric helps skewed dists).

Quantize/dequant math:

- Quantization: `q = clamp(round(x / s) + zp, -128, 127)`
- Dequantization in GEMM epilogue: `x̂ = (q - zp) * s`
- For symmetric we use `zp=0`.

**Fused rescale:** If `C = A_int8 * B_int8^T` with int32 accumulation,

```
Y_fp = (sA_row ⊗ sB_col) ⊙ C_int32  + bias
```

where `sA_row` is per-activation row scale and `sB_col` is per-weight column scale; fold them into the epilogue.

## 2.4 Weight rounding (optional AWQ flavor)

Improve low-bit rounding by preserving important channels (activation-aware rounding), especially for W4; for W8 it’s often near-lossless but still helps with outliers.

## 2.5 Export artifacts

Per layer, persist:

- Packed **W_int8** with its layout (see §3/§4), plus **sW** (per-channel/group)
- **D** equalization vector (folded into runtime activation scales)
- Activation scale policy (per-channel/group) and how to compute **sA** at runtime
- Bias (FP16/BF16)
- JSON/YAML manifest describing the scheme (sym/asym, groupsize, α, clipping)

# 3) CPU execution path (oneDNN / FBGEMM style)

## 3.1 Prepack & layout

- Use **row-major A (activations)** and **column-major B (weights)**, or library-preferred blocked layouts.
- Prepack B: `PackB(W_int8, col_block=64)` for cache-friendly access.

## 3.2 Runtime (prefill)

1. Apply equalization: `x' = x @ D` (or multiply by `d_j` per channel; fuse into preceding op if possible).
2. Compute per-row/group **sA** from runtime activations; typically `sA[row,group] = max(|x'|)` or an EMA from calibration.
3. Quantize A → `A_int8` with `sA` (symmetric).
4. **INT8×INT8→INT32 GEMM** via oneDNN/FBGEMM (uses AVX2/AVX512-VNNI).
5. **Epilogue (fused):** `Y = C_int32 * (sA ⊗ sW) + bias`, cast to FP16/BF16.
6. Continue pipeline (GELU/SiLU in FP16/BF16, etc.).

### Minimal C++ (FBGEMM-like sketch; not compile-ready)

```cpp
// Pseudocode illustrating the flow, not full API calls.
packB = PrepackWint8ColMajor(W_int8, sW);     // offline
for (batch of tokens):
  sA = calc_row_group_scales(x_eq);           // runtime scales
  A_int8 = quantize_symmetric(x_eq, sA);
  C_int32 = gemm_i8i8i32(A_int8, packB);      // vectorized VNNI
  Y = requant_fp16(C_int32, sA, sW, bias);    // fuse scales + bias
```

# 4) GPU execution path

## 4.1 Library-first approach (recommended)

- **NVIDIA:** `cublasLtMatmul()` with `CUDA_R_8I` inputs and `CUDA_R_32I` accum, epilogue to apply `alpha_col`/`alpha_row`, bias, and cast to FP16/BF16. Use col/row scale vectors via epilogue “per-column/row” scale features (or fold into B beforehand).
- **AMD (ROCm):** `rocblas_gemm_ex()` with `rocblas_datatype_i8_r` inputs and `i32` accum; epilogue scaling via custom kernel or Tensile epilogues; cast to FP16/BF16.

**Layout & packing**

- Ensure **K is a multiple of 4** for dp4a/IMMA paths; pad if needed.
- Prefer tensor-core paths (IMMA/WMMA int8) when available; fallback to dp4a.

## 4.2 Minimal CUDA kernel (INT8 dp4a core)

This shows the essence: pack INT8 in `char4`, use `__dp4a` to accumulate 4-wide dot-products into `int32`, then fuse dequant scales and bias in the epilogue, writing FP16.

```cpp
// nvcc -arch=sm_75+ (dp4a supported). For real use, prefer cuBLASLt/CUTLASS kernels.
// C = A(MxK, int8) * B(NxK, int8)^T  -> Y(MxN, fp16)
// Assumes K % 4 == 0, A row-major, B row-major (we access B by K-major / transpose logic).

#include <cuda_fp16.h>
#include <stdint.h>

__global__ void gemm_i8_dp4a_epilogue_fp16(
    const int8_t* __restrict__ A,       // [M, K]
    const int8_t* __restrict__ B,       // [N, K]
    const float*  __restrict__ sA_row,  // [M]  (or [M, G] if group-wise; simplify to [M] here)
    const float*  __restrict__ sB_col,  // [N]
    const __half* __restrict__ bias,    // [N]  (optional; can be nullptr)
    __half*       __restrict__ Y,       // [M, N]
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // m
    int col = blockIdx.x * blockDim.x + threadIdx.x; // n
    if (row >= M || col >= N) return;

    const int8_t* a_ptr = A + row * K;
    // Access B as row-major [N,K], but we need B[col, :] for the dot with A[row, :]
    const int8_t* b_ptr = B + col * K;

    int acc = 0;
    // process 4 int8 at a time
    const int4* a4 = reinterpret_cast<const int4*>(a_ptr);
    const int4* b4 = reinterpret_cast<const int4*>(b_ptr);
    int K4 = K >> 2;

    #pragma unroll 1
    for (int k4 = 0; k4 < K4; ++k4) {
        // Each int4 holds 4 int8 values
        int a_packed = reinterpret_cast<const int*>(a4)[k4];
        int b_packed = reinterpret_cast<const int*>(b4)[k4];
        // dp4a: acc += sum_{i=0..3} a_i * b_i  (int8*int8 -> int32)
        acc = __dp4a(a_packed, b_packed, acc);
    }

    // Epilogue: dequant + bias + cast
    float scale = sA_row[row] * sB_col[col];
    float y = static_cast<float>(acc) * scale;
    if (bias) y += __half2float(bias[col]);
    Y[row * N + col] = __float2half(y);
}

// launcher sketch
void launch_int8_gemm(
    const int8_t* A, const int8_t* B,
    const float* sA_row, const float* sB_col,
    const __half* bias, __half* Y,
    int M, int N, int K, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1)/block.x, (M + block.y - 1)/block.y);
    gemm_i8_dp4a_epilogue_fp16<<<grid, block, 0, stream>>>(
        A, B, sA_row, sB_col, bias, Y, M, N, K);
}
```

**Notes on making this production-ready**

- Add shared-memory tiling, vectorized global loads/stores, and double-buffering.
- Use **Tensor Cores IMMA** (mma.sync for s8) when available for much higher throughput.
- Handle K tail (K%4) and different layouts (col-major B or pre-transposed B_T).
- For **group-wise** activation scales, index `sA_row_group[row, g]` per K-tile and multiply partial results or pre-scale A.

## 4.3 GPU runtime steps (prefill)

1. Equalize: `x' = x · D` (fuse wherever possible).
2. Compute per-row/group **sA** for the current mini-batch (or reuse EMA from calibration to avoid per-token cost).
3. Quantize A → `A_int8` (symmetric, zp=0).

   - You can fuse `x' / sA` + clamp + cast in a small kernel.

4. GEMM INT8×INT8→INT32 (cuBLASLt/rocBLAS or your kernel).
5. Epilogue: apply `(sA ⊗ sW)` and bias; write FP16/BF16.
6. Continue with non-quant ops (GELU/SiLU/softmax) in FP16/BF16.

### Decode path option

- To avoid quantize-A overhead at tiny batch sizes, keep **A16** during decode (W8A16) and use W8 with FP16 activations for the projections; still preserves memory benefit of W8 and stability, while prefill keeps the A8 speedup.

# 5) Practical knobs that matter

- **Granularity:** per-channel or small groups (32/64) for both W and A.
- **Symmetric vs asymmetric:** symmetric (zp=0) is fastest; if activations are skewed, asymmetric helps (adds subtractions in inner loop).
- **Outliers:** SmoothQuant equalization; percentile clipping at 99.9% for A; per-layer `α` search (grid over 0.5–0.9).
- **Accumulation:** keep **int32 accum**, dequant to FP16/BF16 in epilogue.
- **Fallbacks:** allow FP16 for a few fragile blocks (first/last MLP, attention output) if you see quality dips.

# 6) Validation & expected results

- **Offline:** perplexity delta on validation, a few task probes (e.g., MMLU subset) vs FP16 baseline.
- **Online:** measure **prefill tokens/s** and CUDA/ROCm kernel mix; confirm INT8 paths dominate prefill FLOPs.
- Typical: **1.3–1.8× prefill throughput**; minimal quality loss with good calibration.

# 7) File/asset checklist for deployment

- `model.int8/`:

  - `layer_i/W_int8.bin` (+ layout)
  - `layer_i/sW.{fp32}.bin` (per-channel/group)
  - `layer_i/D.{fp32}.bin` or folded into export
  - `layer_i/bias.fp16.bin`

- `quant_manifest.yaml` (scheme, groupsize, α, clip, symmetric/asymmetric, library layout)
- Integration tests comparing logits vs FP16 on a small prompt set.
