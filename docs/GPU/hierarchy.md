# 1) Execution hierarchy (software → hardware mapping)

```
Kernel launch
└─ Grid (all blocks for a kernel)
   └─ Block / CTA (co-resident on one SM/CU)
      └─ Warp / Wavefront (lockstep group: 32 on NVIDIA, 32/64 on AMD)
         └─ Thread / Lane
```

- **CTA scheduler** (hardware/firmware) assigns blocks to **SMs** (NVIDIA) / **CUs** (AMD).
- A block never spans SMs/CUs; multiple blocks can reside concurrently on an SM/CU if resources fit (regs, shared/LDS, warp slots).

# 2) Compute blocks inside the core (SM on NVIDIA / CU or SIMD array on AMD)

**Per SM / CU you typically find:**

- **Warp/Wavefront schedulers** & **issue/dispatch units** (scoreboards, dependency tracking).
- **Register file** (very large, banked; partitioned per scheduler/processing block).
- **Shared memory (NVIDIA)** / **LDS (AMD)** — low-latency, software-managed scratchpad, banked.
- **Load/Store (LD/ST) units** and address coalescers.
- **ALUs** (INT32, FP32, FP64 where present).
- **Special Function Units (SFUs)** for transcendental ops.
- **Tensor/Matrix cores** (NVIDIA “Tensor Cores”, AMD “Matrix Cores/XCDNA MFMA”) for MMA operations.
- **Texture/Sampler units** (useful even in compute for filtered/normalized reads).
- **Instruction cache(s)** (L0/L1I depending on arch).
- **Data cache(s)** (often the L1D is unified with shared memory on NVIDIA; AMD has per-CU caches).

# 3) Memory hierarchy & address spaces (latency roughly increases top→bottom)

**Per-thread**

- **Registers** – fastest, allocated from SM/CU register file.
- **Local memory** – per-thread spill space; physically in global DRAM but cached by L1/L2.

**Per-block (on one SM/CU)**

- **Shared memory / LDS** – user-managed scratchpad (very low latency, banked).
- **L1 data cache** – near the SM/CU; may be unified with shared mem (configurable on many NVIDIA gens).

**Per-SM/CU or nearby**

- **Constant cache** – serves `__constant__`/read-only data, broadcast-friendly.
- **Texture cache** – for texture/sampler path and often read-only data.

**Chip-wide**

- **L2 cache** – unified, shared by all SMs/CUs; sliced by memory partitions.
- **MMU & TLBs** – per-SM/CU L1 TLBs and a larger shared L2 TLB (exact layout varies).

**Off-chip**

- **HBM/GDDR (DRAM)** – device global memory via memory controllers.
- **Peer/Host paths** – **NVLink/NVSwitch/PCIe** (NVIDIA) or **Infinity Fabric/xGMI/PCIe** (AMD) for GPU↔GPU and GPU↔CPU.

**Other memory-like spaces you’ll encounter in programming models**

- **Read-only data cache** (NVIDIA `__ldg`/RO cache path).
- **Surface/texture objects** (specialized addressing/filtering).
- **Unified/Managed memory** (driver-managed migration/paging across CPU↔GPU).

# 4) Chip organization above/beside SMs/CUs

- **SM (Streaming Multiprocessor)** (NVIDIA) ↔ **CU (Compute Unit)** / **WGP** (Work-Group Processor, AMD RDNA/CDNA grouping).
- **TPC / GPC** (NVIDIA graphics-era groupings of SMs; less visible for pure compute but still physical clusters).
- **NoC / Crossbar / Fabric** connecting SMs/CUs to **L2 slices** and **memory partitions**.
- **Memory controllers** per HBM/GDDR stack/channel.
- **Command processors**:

  - **GPC/GFX/Compute front-ends** parse/dispatch work.
  - **Copy/DMA engines** (async memcopy, H2D/D2H, peer-to-peer).
  - **Async/Tensor Memory Accelerators** (e.g., NVIDIA TMA on Hopper) for bulk tiled copies.

- **Fixed-function/aux engines** (may or may not matter to compute):

  - **NVENC/NVDEC** (video encode/decode), **JPEG engines**.
  - **RT cores** (ray tracing), **raster/ROP** (graphics pipeline).

# 5) Scheduling & concurrency concepts

- **Occupancy**: how many warps/blocks can reside on an SM/CU concurrently (bounded by registers, shared/LDS, warp slots).
- **Scoreboarding**: tracks operand readiness; interacts with **warp schedulers** to hide latency.
- **Barriers/Sync**: per-block barriers (e.g., `__syncthreads()` / `s_barrier`) and finer-grained warp sync.
- **Asynchronous copies**: e.g., `cp.async`/TMA to shared/LDS for latency hiding.

# 6) Vendor naming cheat-sheet (rough equivalence)

- **SM (NVIDIA)** ≈ **CU**/**WGP** (AMD).
- **Warp (32)** ≈ **Wavefront (32/64)**.
- **Shared memory** ≈ **LDS**.
- **Tensor Cores** ≈ **MFMA/Matrix instructions** (AMD).
- **NVLink/NVSwitch** ≈ **Infinity Fabric/xGMI** (GPU interconnects).

# 7) One-page hierarchy view

```
System
└─ CPU(s) ─ PCIe/CXL ─┐
                      └─ GPU Device
                         ├─ Interconnect (NVLink/NVSwitch | IF/xGMI)
                         ├─ L2 cache (sliced) + Memory Fabric
                         ├─ Memory Controllers → HBM/GDDR stacks
                         ├─ Command/Copy Engines (DMA/TMA/Enc/Dec)
                         └─ Arrays of Compute Clusters
                            └─ SMs / CUs
                               ├─ Warp/Wave schedulers & dispatch
                               ├─ Register file (per-SM/CU)
                               ├─ L1D / Shared(LDS) / RO / Tex caches
                               ├─ LD/ST units, ALUs, SFUs, Tensor/Matrix cores, Tex units
                               └─ Resident blocks (CTAs)
                                  └─ Warps/Wavefronts
                                     └─ Threads/Lanes
```

# 8) Quick scope table

| Scope           | Resources you tune/watch                                   |
| --------------- | ---------------------------------------------------------- |
| Per-thread      | Registers, local (spill)                                   |
| Per-warp        | Divergence, instruction mix, issue efficiency              |
| Per-block (CTA) | Shared/LDS usage, barriers, residency                      |
| Per-SM/CU       | Occupancy (regs/shared), L1 behavior, scheduler pressure   |
| Chip-wide       | L2 hit rate, memory BW, copy engines, interconnect traffic |
| Off-chip/board  | HBM/GDDR BW, PCIe/NVLink/IF bandwidth                      |
