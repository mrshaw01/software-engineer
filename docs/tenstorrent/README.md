# 1. Compute Engines and Data Flow within Tensix

https://github.com/tenstorrent/tt-metal/blob/a8856ee9851f29972222b46a2b1726d0955fa818/docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst

- Component introduction
- Dst register
- Matrix engine/FPU
- Vector engine/SFPU

# 2. Achieving FP32 Accuracy for Computation

https://github.com/tenstorrent/tt-metal/blob/a8856ee9851f29972222b46a2b1726d0955fa818/docs/source/tt-metalium/tt_metal/advanced_topics/fp32_accuracy.rst

- Host-side configuration
- Kernel-side implementation
- Distinguishing between matrix and vector engine APIs

# 3. Tiles

https://github.com/tenstorrent/tt-metal/blob/a8856ee9851f29972222b46a2b1726d0955fa818/docs/source/tt-metalium/tt_metal/advanced_topics/tiles.rst

- Internal structure of a Tile
- Conversion between tiles and row-major format

# 4. Memory from a kernel developer's perspective

https://github.com/tenstorrent/tt-metal/blob/a8856ee9851f29972222b46a2b1726d0955fa818/docs/source/tt-metalium/tt_metal/advanced_topics/memory_for_kernel_developers.rst

- Data addressing on Tenstorrent processors
  - RISC-V Address Space
  - DRAM tiles
  - Memory access via the NoC
- Tensor Layout
- Memory placement
  - Lock step allocation
  - Interleaved memory
  - SRAM buffers
  - Sharded tensor

# 5. TT Architecture and Metalium Guide

https://github.com/tenstorrent/tt-metal/blob/471840ed81a36bd3fd8156c8f3b5ab3d193d99fa/METALIUM_GUIDE.md

- Executive Summary
- Tenstorrent Architecture Overview
  - SRAM, interleaved and shared buffers
  - Native tile based computing
  - Where is the cache hierarchy
  - Tenstorrent hardware for GPU experts
  - Tenstorrent hardware for CPU experts
  - Scaling beyond one chip
- TT-Metalium
  - Running code on device
  - Register control and Data Flow within the Compute Kernels
  - Compute APIs
  - Fast dispatch
  - SPMD in Metalium
