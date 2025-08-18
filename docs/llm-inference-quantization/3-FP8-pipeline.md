# FP8 pipeline

- **Data formats:** FP8 **E4M3** and **E5M2** (8-bit floats). Typical choice: **E4M3** for weights & activations (more precision), **E5M2** for gradients (more range) when training; for inference you’ll usually use **E4M3** everywhere the kernels support it. ([NVIDIA Developer][1], [Lambda][2], [VLLM Docs][3])
- **Compute:** **accumulate in BF16/FP16**, not in FP8. Tensor cores (H100) and MFMA equivalents (MI300) do FP8 input → BF16/FP16 accumulate → optional BF16/FP16 output. ([NVIDIA Developer][4])
- **Library layer:** NVIDIA **Transformer Engine** + **cuBLASLt** on H100; AMD **ROCm TransformerEngine** + **hipBLASLt** on MI300. ([NVIDIA Docs][5], [GitHub][6], [Radeon Open Compute Documentation][7])

# End-to-end on CPU (prep, packing, calibration)

You won’t _compute_ matmuls in FP8 on CPU, but you’ll do all the prep so the GPU kernels stay fast and stable.

1. **Pick formats & scaling policy**

- Default: **E4M3** for W/A, **BF16** accumulation, per-tensor or **group-wise** (e.g., 64-channel) scales.
- Keep **LayerNorm/RMSNorm, rotary phase, logits** in BF16/FP16. ([arXiv][8])

2. **Collect calibration stats (activation amax)**

- Run 128–512 short, representative prompts in BF16 and record **amax** after layer norms and at GEMM inputs (QKV proj, MLP in/out).
- If you’ll use TE’s dynamic scaling on GPU, you can still pre-warm with CPU stats to get initial scales.

3. **Derive scales**

- For format **E4M3**, the largest finite is \~**448**. Set `scale = amax / 448` (per tensor/group/channel as chosen). Store **`scale_inv = 1/scale`** for fast on-GPU dequant. (Use 57344 for E5M2 if you do gradients; for inference you likely won’t.) ([VLLM Docs][3])

4. **Pack weights to FP8 offline**

- Convert FP16/BF16 weights to FP8 bytes + scales (and optionally per-group scale tables) so GPU doesn’t pay this cost at load time.
- Keep a BF16 “master copy” only if you expect to re-tune scales; otherwise FP8+scale is enough for inference.

5. **Export metadata**

- Persist: tensor name → {format(E4M3/E5M2), layout, group size, `scale_inv`, amax history seed}.
- Your runtime then loads FP8 arrays and scale vectors directly into device memory.

_Why this works:_ FP8 uses floating exponents, so it’s more tolerant than INT8 to distribution shifts. You still need good **amax** and group granularity. ([Baseten][9])

# End-to-end on GPU (runtime & kernels)

Two good paths: **(A)** use Transformer Engine modules, or **(B)** call the FP8 GEMM libraries (cuBLASLt / hipBLASLt) yourself.

## A) High-level: Transformer Engine (in PyTorch)

### NVIDIA H100 (CUDA 12+, TE ≥ 2.x)

- Use `transformer_engine.pytorch` layers under an `fp8_autocast` context; TE tracks amax and updates scales automatically each step (you can freeze after warm-up for inference). ([NVIDIA Docs][5])

```python
# NVIDIA Hopper (H100): FP8 inference with TE
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format

model = ...                   # your HF/vLLM module converted to te.Linear, te.LayerNorm, etc.
model.eval().cuda()
recipe = te.DelayedScaling(   # TE handles amax/scale history internally
    margin=0, interval=1,
    fp8_format=Format.E4M3
)

with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    with torch.inference_mode():
        y = model(x.cuda())
```

- Under the hood TE dispatches FP8 GEMMs through **cuBLASLt** and fuses de/quant, bias, gelu, etc., with BF16 accumulation. ([NVIDIA Developer][10])

### AMD MI300 (ROCm ≥ 6.2, hipBLASLt backend)

- Use **ROCm TransformerEngine**; the PyTorch API mirrors NVIDIA’s, routing GEMMs to **hipBLASLt**. Install ROCm TE and set FP8 recipe similarly (E4M3). ([GitHub][11], [Radeon Open Compute Documentation][7])

> Notes for both: warm up a few dozen forward passes to stabilize amax; then optionally freeze scales for deterministic inference.

## B) Low-level: call the FP8 GEMM yourself

### Minimal cuBLASLt FP8 matmul (H100)

This is adapted from NVIDIA’s official FP8 sample; inputs A/B are FP8 **E4M3** with per-tensor scales; compute in FP16/BF16. ([GitHub][12])

```cpp
// H100 FP8 GEMM: C = alpha * A * B + beta * C
// A: [m,k] FP8 E4M3, B: [k,n] FP8 E4M3, C: [m,n] BF16
cublasLtHandle_t lt;
cublasLtCreate(&lt);

cublasOperation_t opN = CUBLAS_OP_N;
cublasLtMatmulDesc_t opDesc;
cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;

cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_16F, CUDA_R_16F); // BF16/FP16 compute
cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, m, k, m);
cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, k, n, k);
cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF,   m, n, m);

// Optional: attach per-tensor amax/scale pointers via matmul desc attributes
// (see sample_cublasLt_LtFp8Matmul.cu)

float alpha = 1.f, beta = 0.f;
cublasLtMatmul(lt, opDesc,
               &alpha, A_fp8, Adesc,
                        B_fp8, Bdesc,
               &beta,  C_bf16, Cdesc,
                      C_bf16, Cdesc,
               /*algo, workspace, stream… see sample*/);
```

- The **official sample** shows the required attributes for scales/amax histories and the recommended heuristics/algo selection; use it as your reference implementation. ([GitHub][12])
- Keep CUDA ≥ 12.6 to pick up FP8 fixes and perf in cuBLASLt. ([NVIDIA Docs][13])

### Minimal hipBLASLt FP8 matmul (MI300)

hipBLASLt exposes the same idea: FP8 inputs + BF16/FP16 accumulate. Types and allowed combinations are documented in the API reference. ([Radeon Open Compute Documentation][7])

```cpp
// MI300 FP8 GEMM (hipBLASLt)
hipblasLtHandle_t lt;
hipblasLtCreate(&lt);

hipblasOperation_t opN = HIPBLAS_OP_N;
hipblasLtMatmulDesc_t opDesc;
hipblasLtMatrixLayout_t Adesc, Bdesc, Cdesc;

hipblasLtMatmulDescCreate(&opDesc, HIPBLASLT_COMPUTE_BF16, HIP_R_16BF);
hipblasLtMatrixLayoutCreate(&Adesc, HIP_R_8F_E4M3, m, k, m);
hipblasLtMatrixLayoutCreate(&Bdesc, HIP_R_8F_E4M3, k, n, k);
hipblasLtMatrixLayoutCreate(&Cdesc, HIP_R_16BF,   m, n, m);

float alpha = 1.f, beta = 0.f;
hipblasLtMatmul(lt, opDesc,
                &alpha, A_fp8, Adesc,
                         B_fp8, Bdesc,
                &beta,   C_bf16, Cdesc,
                         C_bf16, Cdesc,
                /*algo, workspace, stream…*/);
```

- Use **ROCm ≥ 6.2** for FP8 enablement on MI300; newer releases keep improving kernels & tooling. ([Datacenter Dynamics][14], [Radeon Open Compute Documentation][15])

# Layer-by-layer mapping (Transformers)

- **Embeddings / final logits:** keep BF16/FP16.
- **QKV, O proj, MLP in/out:** FP8 matmul (E4M3) + BF16 accumulate; de/quant fused in epilogue.
- **Attention softmax/RoPE:** compute in BF16; cast inputs/outputs as needed.
- **Norms:** BF16/FP16.
- **KV cache:** can be FP8 or INT8 to halve memory; many teams use INT8 per-head scaling for safer long-context; FP8 works on FP8-capable GPUs too. (Your mileage varies; test.) ([NVIDIA Developer][4])

# Practical guardrails

- **Warm-up & freeze:** Run 50–200 inferences with amax tracking, then freeze scales for stability (TE supports this). ([NVIDIA Docs][5])
- **Group size:** 32–64 usually balances quality vs overhead; per-tensor is fastest but least robust.
- **E4M3 saturation:** watch for amax spikes; clamp or increase group granularity if 99.9-percentile/amax nudges the scale too high. ([arXiv][8])
- **Heuristics/algo IDs:** Don’t hard-code cuBLASLt/hipBLASLt algos; query heuristics per shape. NVIDIA has patched several FP8 issues in recent CUDA 12.x; stay current. ([NVIDIA Docs][13])
- **H100 extras:** Leverage **TMA** (Tensor Memory Accelerator) and WGMMA-based kernels if you custom-code; TE/cuBLASLt already do. ([PyTorch][16])

# Tiny “pack on CPU, run on GPU” example

**CPU pack (conceptual):**

```python
# x: torch.float16 or bfloat16 tensor on CPU
# returns: uint8 payload (fp8), scale_inv for dequant on GPU
def pack_to_fp8_e4m3(x, group=64):
    # compute per-group amax and scales
    xg = x.view(-1, group)
    amax = xg.abs().amax(dim=1, keepdim=True)
    scale = torch.clamp(amax / 448.0, min=1e-8)                # E4M3
    scale_inv = (1.0/scale).repeat_interleave(group, dim=1).reshape_as(x)
    # quantize to fp8 bytes via a library or table (omitted for brevity)
    # store payload bytes + one scale per group
    return fp8_bytes, scale.squeeze()
```

At load time you `cudaMemcpy` `fp8_bytes` and `scale_inv` (or `scale`) to device and call cuBLASLt/hipBLASLt as shown.

**GPU call:** use the minimal cuBLASLt or hipBLASLt snippets above.

# When to choose FP8 vs INT8

- If you have H100/MI300 and can use **Transformer Engine + (cu|hip)BLASLt**, FP8 usually gives **1.5–2× prefill** speedups with little quality loss and less integration pain than INT8 activation quant. ([NVIDIA Developer][4])
- INT8 can still win for strict memory budgets or older GPUs; for Hopper/MI300, FP8 is the smoothest high-throughput path.

## TL;DR recipes

- **On CPU:** choose E4M3, gather amax on a small corpus, compute per-group scales, pack weights to FP8 bytes + scales, emit metadata.
- **On GPU (NVIDIA):** TE `fp8_autocast` or direct **cuBLASLt FP8**; BF16 accumulate; keep norms/logits in BF16; warm-up then freeze scales. ([NVIDIA Docs][5], [GitHub][12])
- **On GPU (AMD):** ROCm TE with **hipBLASLt FP8** GEMMs (ROCm ≥ 6.2); same layer policy and warm-up. ([GitHub][11], [Radeon Open Compute Documentation][7])
