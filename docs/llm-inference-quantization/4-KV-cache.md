# KV Cache Quantization

Quantize K/V tensors stored in the cache (post-projection, after RoPE) to shrink memory and bandwidth: **FP16 → INT8 ≈ 2×**, **FP16 → INT4 ≈ 4×**. Prefer **per-head, per-group (32/64)** symmetric scales; dequantize on-the-fly inside attention.

# Design choices you should lock in

- **Bit-width:**

  - **INT8** = safest, near-lossless for long contexts.
  - **INT4** = aggressive; great for capacity but watch very long contexts.
  - **FP8 (E4M3/E5M2)** on H100/MI300 with TE: stable and simple if your stack already supports FP8 casting.

- **Granularity:** per-head + per-group along `head_dim` (e.g., group_size=64). Store **separate scales for K and V**.
- **Symmetry:** symmetric (no zero-point) for fast matmuls.
- **Where to quantize:** **on write** (when you append the new token’s KV).
- **Where to dequantize:** **inside attention kernels** (fused with QK^T and AV), never as a separate pass.
- **Layout:** keep your existing paged cache; attach a small **scale table per (layer, head, page, group)**.
  Example KV memory: `[layer][head][page][token_in_page][head_dim_qbits]`.
  Example scales: `[layer][head][page][groups]` (FP16/BF16).

# Calibration (offline or warm-up)

1. Run **128–512 short prompts**.
2. For each `(layer, head)`, collect K and V activations at the cache **write** point (after RoPE).
3. For each **group** over `head_dim`, compute scale:

   ```
   s = max(|x|) / qmax          # max or 99.9th percentile
   # qmax = 127 for INT8, 7 for INT4
   ```

   (Optionally SmoothQuant/percentile to handle outliers.)

4. Persist scale stats. If using **paged dynamic scales**, compute per-page stats during prefill and carry them with the page.

**Rule of thumb:** INT8, per-head, group_size=64, symmetric + percentile(99.9%) is a strong default.

# Runtime pipeline

## CPU path (reference & production)

### Write path (append token t)

1. Compute `K_t, V_t` in FP16/BF16.
2. Quantize per `(head, group)` using stored scales.
3. Pack to INT8 (or INT4) and write into paged KV.
4. Write/update scales for the current page (if page-dynamic).

**Reference quantize (C++ – scalar, easy to SIMD):**

```cpp
// x: FP16/BF16 input (promote to float for math), y: int8 output
// scale[g] is the per-group scale for this head (float/FP16)
inline int8_t q8(float x, float inv_s) {
    float r = x * inv_s;
    r = std::round(std::min(127.f, std::max(-127.f, r)));
    return static_cast<int8_t>(r);
}

void quantize_kv_int8(const float* __restrict x, // [head_dim]
                      int8_t* __restrict y,      // [head_dim]
                      const float* __restrict scale, // [groups]
                      int head_dim, int group_size)
{
    int G = (head_dim + group_size - 1) / group_size;
    for (int g = 0; g < G; ++g) {
        float inv_s = 127.f / std::max(1e-8f, scale[g]);
        int start = g * group_size;
        int end = std::min(start + group_size, head_dim);
        for (int i = start; i < end; ++i) y[i] = q8(x[i], inv_s);
    }
}
```

**SIMD / libraries:**

- For **INT8**: dequant in-place to FP16 and use your normal GEMM (BLAS); or use **oneDNN int8 matmul** (s8 × s8 → s32, dequant epilogue).
- For **INT4**: pack two nibbles per byte; SIMD with AVX-512 VNNI/AMX for matmul, or dequant to FP16 before GEMM.

### Read path (attention at step t)

- **Fuse dequant into matmuls**:

  - QK^T: load `K_q`, multiply by `scale[g]/127`, accumulate in FP16/BF16.
  - AV: same for `V_q`.

## GPU path (production)

### Memory & metadata

- Allocate KV in INT8/INT4 buffers (global mem), page-aligned.
- Allocate **scale tables**: FP16/bfloat16, `[L][H][pages][groups]`.

### Write path (append token t)

Launch a light **quantize-and-pack** kernel after the K/V projection (and RoPE for K). One warp per `(layer, head, token)` or tile over `head_dim`.

**Minimal CUDA/HIP-friendly INT8 quantize kernel**

```cpp
// Works with CUDA or HIP via preprocessor guards
#include <cuda_fp16.h>
extern "C" __global__
void kv_quantize_int8_kernel(const half* __restrict x,   // [tokens*heads*head_dim]
                             int8_t* __restrict y,       // same shape
                             const half* __restrict scale, // [heads*groups]
                             int head_dim, int group_size,
                             int heads, int tokens)
{
    int tok = blockIdx.y;      // token index within the launch
    int h   = blockIdx.x;      // head index
    int lane = threadIdx.x;    // 0..(warpSize-1)

    const int G = (head_dim + group_size - 1) / group_size;

    const half* x_base = x + (tok*heads + h)*head_dim;
    int8_t*     y_base = y + (tok*heads + h)*head_dim;
    const half* s_base = scale + h*G;

    for (int g = 0; g < G; ++g) {
        float s = __half2float(s_base[g]);
        float inv_s = 127.f / fmaxf(1e-8f, s);

        int start = g * group_size + lane;
        for (int i = start; i < min((g+1)*group_size, head_dim); i += warpSize) {
            float xf = __half2float(x_base[i]);
            float r  = nearbyintf(fminf(127.f, fmaxf(-127.f, xf * inv_s)));
            y_base[i] = static_cast<int8_t>(r);
        }
    }
}
```

**Launch config (example):**

```cpp
dim3 grid(heads, tokens);
dim3 block(32); // one warp
kv_quantize_int8_kernel<<<grid, block>>>(x, y, scale, head_dim, 64, heads, tokens);
```

**INT4 variant tips:** quantize to int, clamp to \[-7,7], then pack two 4-bit values per byte:

```cpp
uint8_t hi = (uint8_t)((q_hi & 0xF) << 4);
uint8_t lo = (uint8_t)(q_lo & 0xF);
out[i >> 1] = hi | lo;
```

(Keep an INT4 version of the kernel with packing logic; group/scale logic identical.)

### Read path (fused dequant inside attention)

**Option A (common):** Keep matmuls FP16/BF16 and **dequant on load**.
**Option B:** Use **INT8 GEMMs** if you already run Q in INT8 (more plumbing).

#### Minimal QK^T inner-product loop with on-the-fly dequant (CUDA fragment)

```cpp
// Within a tiled attention kernel, for one (q_row, k_col) dot:
float acc = 0.f;
for (int g = 0; g < G; ++g) {
    float s = __half2float(scale_k[g]);    // scale for this group
    float dq = s / 127.f;

    // Load a tile (vectorized if possible)
    #pragma unroll
    for (int t = 0; t < TILE_ELEMS; ++t) {
        // k_q: int8, q: half
        int8_t kq = k_tile[g_offset + t];
        float  kf = (float)kq * dq;
        float  qf = __half2float(q_tile[g_offset + t]);
        acc += qf * kf;
    }
}
// write acc to shared or register fragment; continue softmax later
```

#### Minimal AV (values) multiply with on-the-fly dequant (CUDA fragment)

```cpp
// For one output feature dim j in V:
float out = 0.f;
for (int g = 0; g < G; ++g) {
    float s = __half2float(scale_v[g]);
    float dq = s / 127.f;
    #pragma unroll
    for (int t = 0; t < TILE_ELEMS; ++t) {
        int8_t vq = v_tile[g_offset + t];
        float  vf = (float)vq * dq;
        float  p  = attn_prob[t]; // from softmax row
        out += p * vf;
    }
}
// store out (FP16/BF16)
```

**FP8 path (H100 / MI300):** If you already use Transformer Engine, replace manual int8 path with FP8 casting ops for KV (E4M3 for V, E5M2 for K often works), accumulate in BF16. The flow stays the same: cast on write, fused cast on read.

# Integration with paged-KV

- **When you allocate pages**, also allocate page-local scale buffers (`[head][groups]`).
- **During prefill**, compute scales for the page (if dynamic) before writing many tokens; otherwise use calibrated static scales.
- **Carry scales** with the page during swaps/evictions.

# Validation & guardrails

- **Numerics checks:** cosine sim of attention logits vs FP16; end-to-end perplexity delta (≤ +0.2 typical for INT8).
- **Stress tests:** very long contexts; repeated entities; cross-head saturation counts.
- **Profiling:** ensure dequant happens **inside** the matmul tiles (no extra global writes).
- **Fallbacks:** keep first/last layer heads in FP16 if you see quality cliffs; raise `group_size` (32→64) before changing bit-width.

# Troubleshooting quick hits

- Saturation spikes → switch max→percentile or increase group size; split outlier channels.
- Long-context drift (INT4) → try INT8 for K only; keep V INT4.
- Kernel stalls → vectorize loads (e.g., `int4`/`char4`), coalesce by head_dim, prefetch scales to shared memory.

# Recommended defaults (works well in practice)

| Component   | Setting                                                                                       |
| ----------- | --------------------------------------------------------------------------------------------- |
| Bit-width   | **INT8** for both K and V                                                                     |
| Granularity | Per-head, **group_size=64**                                                                   |
| Scales      | Symmetric, FP16 storage                                                                       |
| Dequant     | Fused in QK^T and AV kernels                                                                  |
| Paged-KV    | Page-local scales if you want dynamic per-page calibration; otherwise static per-(L,H) scales |
| Fallbacks   | Keep a few fragile heads/layers FP16 if needed                                                |
