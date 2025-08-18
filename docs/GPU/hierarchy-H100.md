# H100 at a glance (SXM vs PCIe)

- **SMs:** 132 (SXM5) or 114 (PCIe). Full GH100 has 144 SMs on silicon; shipping parts enable fewer. **L2:** 50 MB unified. **Compute capability:** 9.0. ([NVIDIA Developer][1])
- **Memory:** SXM5 uses **HBM3, \~3 TB/s**; PCIe uses **HBM2e, \~2 TB/s** (both typically 80 GB). ([NVIDIA Developer][1], [Advanced Clustering Technologies][2])
- **NVLink:** Gen-4 up to **900 GB/s** GPU↔GPU on SXM; PCIe cards can bridge two GPUs to **900 GB/s** with NVLink bridges. ([NVIDIA][3])

# Execution hierarchy (software → hardware)

```
Kernel
└─ Grid
   └─ Block / CTA (stays on one SM)
      └─ Warp (32 threads, lockstep)
         └─ Thread
```

H100 caps (per SM): **64 warps**, **2,048 threads**, **32 CTAs**. Compute capability **9.0**. ([Advanced Clustering Technologies][2])

# Inside an H100 SM (SM90)

- **Cores:** **128 FP32 CUDA cores / SM**, **4 Tensor Cores / SM**; FP8‐capable “Transformer Engine.” ([Advanced Clustering Technologies][2], [NVIDIA Developer][1])
- **Registers:** **64 K 32-bit regs / SM** (256 KB); **≤255 regs/thread**. ([NVIDIA Docs][4])
- **L1 + Shared (unified):** **256 KB per SM**, **shared mem carve-out up to 228 KB** (runtime-selectable). ([Advanced Clustering Technologies][2], [NVIDIA Docs][5])
- **Other functional units:** LD/ST units with coalescers; SFUs; texture/sampler path (also used in compute). _(Inherited from prior gens; NVIDIA doesn’t list every lane in public specs.)_

# On-chip caches & fabric

- **Constant & texture caches:** per-SM (read-optimized paths).
- **L2 cache:** **50 MB**, shared by all SMs, sliced per memory partition. ([NVIDIA Developer][1])
- **NoC / crossbar:** links SMs ↔ L2 ↔ memory controllers (HBM stacks).

# Off-chip & interconnects

- **HBM:** SXM5 (HBM3 \~3 TB/s), PCIe (HBM2e \~2 TB/s). ([NVIDIA Developer][1], [Advanced Clustering Technologies][2])
- **NVLink 4:** **900 GB/s** GPU↔GPU (SXM); PCIe pair **900 GB/s** via bridges. **PCIe Gen5** for host links. ([NVIDIA][3])

# Hopper-specific features you’ll actually use

- **Thread-Block Clusters (TBC):** schedule CTAs as a gang across multiple SMs (up to **16 CTAs per cluster**), enabling **Distributed Shared Memory (DSM)**—CTAs can directly load/store/atomic into each other’s shared memory. ([Advanced Clustering Technologies][2], [NVIDIA Developer][1])
- **TMA (Tensor Memory Accelerator):** hardware async bulk copies GMEM↔SMEM; supports CTA-to-CTA copies inside a cluster. ([NVIDIA Developer][1])
- **DPX instructions:** dynamic-programming kernels (e.g., Smith–Waterman, Floyd–Warshall) see **\~7×** vs. A100. ([NVIDIA Blog][6])
- **Transformer Engine (FP8):** auto-mixes **FP8 (E4M3/E5M2)** with 16-bit; big speedups on LLMs. ([NVIDIA Developer][1])

# Clean mental hierarchy (H100-specific sizes in **bold**)

```
System
└─ GPU (H100 SXM5 / PCIe)
   ├─ NVLink4 (**900 GB/s**) / PCIe Gen5
   ├─ L2 cache (**50 MB**) + memory fabric
   ├─ Memory controllers → HBM (SXM: HBM3 ~**3 TB/s**; PCIe: HBM2e ~**2 TB/s**)
   └─ GPC/TPC clusters
      └─ SMs (SXM: **132** | PCIe: **114**)
         ├─ Registers (**64K x 32-bit**)
         ├─ Unified L1+Shared (**256 KB**, shared up to **228 KB**)
         ├─ Tensor Cores (**4/SM**), FP32 cores (**128/SM**)
         └─ Resident CTAs → Warps (32) → Threads
```
