# 0) Scope & expectations

- **Goal:** Push size/latency down hard (edge/embedded, tiny VRAM/DRAM, low bandwidth).
- **Common configs:**

  - **W3A8** or **W2A8** for safer accuracy.
  - **W2A4**/**W3A4** when you must shrink bandwidth further (riskier).
  - Optional: **KV INT2/INT4** (KV INT2 is very aggressive; test long-context drift).

- **Key techniques:** **group- or channel-wise scales**, **non-uniform codebooks** (k-means/Lloyd-Max/log-quant), **bit-packing**, **bit-serial matmuls**, **small LoRA/QAT finetune**, **distillation**.
- **Keep in FP16/BF16:** Norm scales, rotary phase, logits (often), a few fragile layers.

# 1) End-to-end (offline) quantization pipeline

## 1.1 Data & calibration

1. Select **128–512 short, representative prompts** (matching your domain).
2. Collect **activation stats** at inputs of attention/MLP (post-norm), per layer.
3. Choose **group size** (e.g., 32 or 64 channels) → better SNR than per-tensor.

## 1.2 Weight codebook & scaling (W3/W2)

You can do **uniform/symmetric** or **non-uniform** (log-quant/Lloyd-Max/k-means):

**Ternary (W3):** codes {−1, 0, +1} with per-group scale `s`.

- Threshold `t` from percentile/clipping; map weight `w` in a group to:

  ```
  q =  0,      if |w| < t
     +1,      if  w ≥ t
     −1,      if  w ≤ −t
  dequant ≈ s * q
  ```

- Solve `s = argmin ||W - s * Q||_F` per group (closed-form via projection).

**Binary (W2, ±1) or 2-bit (0..3 / −2..1):**

- **Binary ±1:** `q ∈ {−1, +1}`, per-group scale `s = mean(|W|)` or least-squares.
- **2-bit signed:** map to {−2, −1, 0, +1} (or optimized codebook via Lloyd-Max). Store `s` and (optionally) a **zero-point** `z` for asymmetric codes.

## 1.3 Activation quant (optional, lightweight)

- Prefer **A8** (symmetric per-channel or per-group; SmoothQuant-style if needed).
- For **A4**, use per-channel scales + percentile clipping (e.g., 99.9%). Keep accum in int32/fp16.

## 1.4 Packing formats (weights)

- **2-bit:** pack **16 values into one 32-bit word** (or 32 values into 64-bit).

  - Layout: `word[j]` stores channel-major or K-major order with group boundaries aligned.

- **Ternary:** store **two bitplanes** per value:

  - `mask_nonzero` (1 if ≠0), `mask_sign` (1 for +1, 0 for −1). Scale `s` per group.
  - Packs like binary but with a parallel mask.

- Save **per-group `scale` (and optional `zero-point` or 2-4-entry codebook)**.

## 1.5 Checkpoint emit

- Emit: packed weights, group scales (and codebooks), per-tensor metadata (group size, layout).
- Keep a **fallback list** of fragile layers in FP16/BF16 (lm_head, first/last MLP/attn often).

## 1.6 Loss-recovery finetune (fast)

Use **post-quant LoRA** or **1–3h QAT** (small budget) with **distillation**:

- Freeze quantization choice; **learn per-group scales and small LoRA adapters**.
- Loss: `L = λ_KL * KL(student, teacher) + λ_ce * CE + λ_reg * ||Δ||`.
- Mix real data + synthetic prompts. Use **low LR**, early stop by **val perplexity** & a few probes.

# 2) Inference path (common considerations)

- **Accumulate in int32/FP16/BF16**; dequant late.
- **Compute in low-bit domain** when possible (bit-serial or table-expand), to avoid dequant bandwidth.
- **Prefill:** activation quant helps matmul throughput.
- **Decode:** weight/KV compression helps memory/BW; keep dequant overhead minimal.

# 3) On CPU: implementation details

## 3.1 Operators

### A) 2-bit weights × A8 activations (int32 accum)

Two strategies:

**(i) Bit-serial (no unpack):**

- Decompose 2-bit `w = w0 + 2*w1` with bitplanes `w0,w1 ∈ {0,1}` (adjust for signed).
- Compute `dot = (a · w0) + 2*(a · w1)` using **bitmask popcount tricks** if `a` is binary;
  else multiply `a8` by `(w0 + 2*w1 − offset)` via masked adds after extracting bits.

**(ii) Table expand (SIMD-friendly):**

- Expand 2-bit nibbles to s8 via **PSHUFB** (SSSE3) or **AVX-512 VBMI** byte-permutes.
- Then use u8×s8 → s16 via **VPMADDUBSW**, accumulate to s32 via **VPMADDWD**.
- Apply per-group scale `s` (and optional `z`) at epilogue.

### B) Ternary weights (±1/0) × A8

- Keep two bitmasks per group: `nonzero`, `sign`.
- Expand on the fly: `w = s * ((sign? +1 : −1) & nonzero)`.
- Multiply with `a8` (vector), accumulate s32; **mask out zeros** cheaply.

## 3.2 Micro-kernel sketch (AVX2/AVX-512)

- Tile K in chunks of 128–256.
- Load packed words; expand to s8 vectors using byte-permute LUT;
- Use **VPMADDUBSW** / **VPDPBUSD (AVX-512 VNNI)** for int8 dot-products;
- Per-group epilogue: convert s32 → fp16/bf16 with scale multiply and optional bias.

## 3.3 KV-cache INT2 (aggressive)

- Quantize per head/channel with **double-quant** (per-block secondary scale to compress scales).
- Store compact: 2-bit K/V + per-block scales.
- **On read:** expand bitplanes or use bit-serial; watch drift at **very long** T (refresh if needed).

# 4) On GPU: implementation details

## 4.1 Kernels & tiling

- Tile M×N into **CTA** tiles (e.g., 128×64), K in strips (e.g., 256).
- **Shared memory**: stage **A** (activations, often int8) and **packed W**.
- Each warp handles a C tile; iterate K:

  - Load packed W (2-bit or ternary) → **decode to s8 in registers** or compute bit-serial.
  - Use **int32 accumulators**; apply per-group scale in epilogue (fp16 output).

- No native INT2 tensor cores; use **bit-serial** or **nibble-unpack→INT4 path**, then int8 MMA/DP4A.

## 4.2 Minimal CUDA kernels

### 4.2.1 Binary (±1) × Binary (±1) dot (XNOR-popcount)

This is the classic fast path when both are binarized (or for ternary’s sign part). It shows the idea; you’ll wrap it inside a tiled GEMM.

```cuda
// Compute dot(x, w) where x,w ∈ {−1,+1}, packed 32 per uint32.
// Return int32 dot. Scale with per-group s in epilogue.
__device__ inline int dot_xnor_popc(const uint32_t* __restrict__ xb,
                                    const uint32_t* __restrict__ wb,
                                    int nwords) {
    int acc = 0;
    #pragma unroll
    for (int i = 0; i < nwords; ++i) {
        // XNOR: matching bits => +1, else −1
        uint32_t x = ~(xb[i] ^ wb[i]);
        // popcnt gives count of +1 matches in this 32-bit chunk
        int m = __popc(x);
        // (+1)*m + (−1)*(32−m) = 2*m − 32
        acc += (m << 1) - 32;
    }
    return acc;
}
```

### 4.2.2 Ternary weights (±1/0) × A8 activations (bitmask + masked add)

- Store **two bitmasks** per 32 positions: `mask_nz`, `mask_sign`.
- We consume 32 activations at a time and apply masked adds/subs; multiply later by `scale`.

```cuda
// a: int8 activations (not bitpacked), len K
// For each 32-lane block we have:
//   mask_nz: bit=1 if weight != 0
//   mask_sign: bit=1 for +1, bit=0 for −1 (only meaningful where nz=1)
// scale s is applied in epilogue (float/bf16).
template<int KWORDS>  // e.g., KWORDS = K/32
__device__ int dot_ternary_a8_block(const int8_t* __restrict__ a,
                                    const uint32_t* __restrict__ mask_nz,
                                    const uint32_t* __restrict__ mask_sign) {
    int acc = 0;
    #pragma unroll
    for (int w = 0; w < KWORDS; ++w) {
        uint32_t nz = mask_nz[w];
        uint32_t sg = mask_sign[w];

        // Process 32 activations in this word.
        #pragma unroll
        for (int b = 0; b < 32; ++b) {
            int8_t av = a[w * 32 + b];
            uint32_t bit = 1u << b;
            if (nz & bit) {
                // +av if sign=1, −av if sign=0
                acc += (sg & bit) ? (int)av : -(int)av;
            }
        }
    }
    return acc; // multiply by scale s and add bias in epilogue
}
```

> Notes:
>
> - For speed, replace the inner loop with warp-wide ballots and use `__vadd*` intrinsics on packed bytes; or pre-expand to ±1 masks using `__byte_perm`/`prmt` and do `dp4a` on gathered 4-tuples.
> - With **2-bit** weights, either unpack nibbles with bitfield extracts (`bfe`) to s8 registers or do **bit-serial**: `dot = (a · w0) + 2*(a · w1)` (adjust for signed).

### 4.2.3 Epilogue (per-group scale)

Apply `y = y_int32 * s_group + bias` and convert to fp16/bf16. If weights had an asymmetric zero-point, fold it into the bias once per group.

## 4.3 KV-cache INT2 on GPU

- Quantize per head/chunk when writing KV: pack 2-bit + per-block scale.
- On read, either **bit-serial** (two passes) or **nibble-expand** to s8 in registers, then use dp4a/IMMA-like paths.
- Consider **periodic refresh** beyond certain sequence lengths to bound drift.

# 5) Putting it together (CPU / GPU pipelines)

## CPU pipeline (W2A8 example)

1. Load packed W2 + scales; map layer-wise fragile exceptions to FP16 path.
2. For each GEMM tile:

   - Prefetch `A8` tile, stream packed W2 tile.
   - **Expand** 2-bit → s8 (PSHUFB/AVX-512 VBMI), or **bit-serial** across two bitplanes.
   - Do u8×s8 int32 dot (VPMADDUBSW/VPDPBUSD).
   - Epilogue: multiply by per-group `s`, add bias, cast to fp16/bf16.

3. KV: write/read INT2 with per-block scales.

## GPU pipeline (W3A8 example)

1. Tile M×N×K; stage A8/packed W3 into shared memory.
2. Warp: fetch packed masks (`nz`, `sign`), load A8 fragment.
3. Compute masked adds/subs (ternary) or unpack 2-bit → s8; accumulate int32.
4. Epilogue: `y = acc * s_group + bias` → fp16/bf16; write global.
5. KV caching with INT2 read/write kernels (bit-serial or expand-in-register).

# 6) Finetune/distillation recipe (quick)

- **Teacher:** FP16/BF16 original. **Student:** Quantized (frozen codebooks), LoRA adapters on attention/MLP.
- **Loss mix (typical):** `λ_KL=0.7, λ_ce=0.3`.
- **LR:** 5e-5→1e-4 for LoRA/scales, cosine decay, 1–3 epochs over 50–200k tokens.
- **Eval:** ppl on calibration set + 3–5 probes (e.g., short QA, arithmetic, code) and your internal prompts.
- **Fallback:** keep lm_head / 1st & last blocks FP16 if Δppl or Δtask too high.

# 7) Validation, metrics, and guardrails

- Track **VRAM/DRAM footprint**, **prefill/decoding throughput**, **end-to-end latency**, **PPL**, and **task deltas**.
- For KV INT2, test **long contexts** (e.g., 16k–64k) and enable **paged-KV**.
- Use **per-layer fallback**: promote outlier-sensitive layers to W8/FP16 if needed.
- Ensure **accumulation** int32/fp16; never accumulate in 2/3 bits.

# 8) Minimal reference: 2-bit packing & decode

**Packing (host):**

```cpp
// Pack 16 signed 2-bit values (v in {-2,-1,0,1} encoded as 0..3) into one uint32.
uint32_t pack16_2bit(const int8_t* vals) {
    uint32_t w = 0;
    for (int i = 0; i < 16; ++i) {
        uint32_t code = uint32_t(vals[i] & 0x3); // assume pre-encoded
        w |= (code << (2*i));
    }
    return w;
}
```

**Decode (device):**

```cuda
__device__ inline int8_t decode2(uint32_t word, int idx) {
    uint32_t c = (word >> (2*idx)) & 0x3u;  // 0..3
    // Map to signed {-2,-1,0,1} (example); better: use LUT/codebook per group.
    static __constant__ int8_t map[4] = {-2, -1, 0, 1};
    return map[c];
}
```

## Practical defaults (good first try)

- **Format:** W3A8, group size 64, per-group symmetric scale; KV INT4 (only switch to INT2 if memory forces it).
- **Calibration:** 256 prompts, 99.9% clip for A8 if needed.
- **Fallback layers:** lm_head + first/last block FP16.
- **Finetune:** LoRA rank 8–16 on attention/MLP, 1–2 epochs with distillation.
- **GPU kernel:** ternary masked-add path (above), upgrade to warp-level bit-packing and dp4a when stabilizing.
