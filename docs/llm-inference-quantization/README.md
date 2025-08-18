# Core scenarios (recipes)

**A) Weight-only quantization (most common, widest HW support)**

- **W8A16** (INT8 weights, FP16 activations): big VRAM save, minimal accuracy loss, little kernel risk. Great default for latency-sensitive decode when you don’t want activation quantization overhead (e.g., GPTQ, AWQ W8).
- **W4A16** (INT4 weights, FP16 activations): \~2× weight memory reduction vs FP16; excellent for fitting bigger models or batch sizes (e.g., GPTQ/AWQ/EXL2 W4). Prefill speedups depend on having good INT4 kernels; decode often benefits from lower memory bandwidth.

**B) Weights + activations quantized (fast prefill)**

- **W8A8 (INT8 matmuls)** with per-channel/group scales and outlier handling (SmoothQuant/AWQ-style). Typically **1.3–1.8× prefill** speedup with small quality loss when calibrated well. Good when you have strong INT8 GEMM kernels.
- **W4A8**: possible but riskier for quality; use per-channel + careful clipping; better for smaller models or when accuracy headroom exists.

**C) FP8 pipelines (H100 / MI300 class with Transformer Engine)**

- **W8(A8) FP8 + BF16/FP16 accumulation**: strong **prefill + decode** gains (often \~1.5–2× prefill) with high accuracy if calibrated. Lower integration friction than INT4 and often more stable than INT8 A-quant for outlier layers.

**D) KV-cache quantization (huge memory & bandwidth win)**

- **KV INT8 (per-channel/head)**: \~2× KV memory reduction, minimal quality impact in typical lengths; great for higher batch/longer contexts.
- **KV FP8**: similar memory to INT8 with good numeric stability on FP8-capable GPUs.
- **KV INT4**: \~4× KV memory reduction; use per-head/group scaling or “double-quant” to control drift; expect some quality drop for very long contexts—best when memory is the bottleneck.

**E) Extreme compression (specialized / edge)**

- **W3/W2 (ternary/binary) or log-quant** with distillation or small finetunes to claw back accuracy. Niche, but useful for tiny devices or aggressive cost targets.

# Granularity & calibration (what actually matters)

- **Scaling granularity:** per-channel or small **group-wise (e.g., 32/64)** almost always beats per-tensor.
- **Symmetric vs asymmetric:** symmetric is faster/simpler for matmuls; asymmetric (zero-points) helps when distributions are skewed.
- **Outliers:** handle via SmoothQuant (shift activation scale into weights), outlier channel splitting, or percentile clipping (e.g., 99.9%).
- **Calibration:** use 128–512 short, representative prompts; collect activation stats after layernorms and MLP/attention inputs; tune group size and clipping.

# What not to (or carefully) quantize

- **LayerNorm/RMSNorm scales, rotary embeddings phases, and final logits computation** often best left in FP16/BF16.
- **lm_head** can be sensitive at very low bit-widths; consider W8 or keep FP16 if quality is paramount.

# Choosing a scenario (quick picker)

- **Need max throughput on H100/MI300 and can calibrate?** FP8 (W8/A8, accumulate in BF16) + **KV FP8/INT8**.
- **Small-batch, latency-critical decode and you want minimal risk?** **W8A16** (or **W4A16** if VRAM is tight) + **KV INT8**.
- **Memory-constrained, very long context / high concurrency?** **W4A16 + KV INT4/INT8**, plus paged-KV.
- **CPU/offload or older GPUs with solid INT8 kernels?** **W8A8** (SmoothQuant/AWQ-style) for prefill speed; decode may still run A16 to avoid dequant overhead.

# Typical savings & effects (rules of thumb)

- **Weights:** FP16→INT8 ≈ 2× smaller; FP16→INT4 ≈ 4× smaller.
- **KV-cache:** FP16→INT8 ≈ 2×; →INT4 ≈ 4×; directly lifts batch/throughput and long-context capacity.
- **Quality:** INT8 done right ≈ near-lossless; INT4 generally good with per-channel/group + calibration; KV INT4 shows more drift on very long sequences.

# Integration tips

- Use **per-layer fallback** (keep a few fragile layers in FP16).
- Keep **accumulation** in FP16/BF16 for stability.
- Benchmark **prefill vs decode** separately—activation quant helps prefill most; weight/KV quant helps decode and memory footprint.
- Validate with **perplexity + task probes** (e.g., MMLU subset, LAMBADA) and a few internal prompts you care about.
