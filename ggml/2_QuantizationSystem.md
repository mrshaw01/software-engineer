# Quantization System

This document describes GGML’s quantization system, which compresses tensor storage by representing rows in packed low-precision block formats instead of dense floating-point arrays. The system combines a type-based metadata layer, reference quantization and dequantization routines, and backend-specific execution hooks for quantized arithmetic.

## Overview

- `include/ggml.h`
- `src/ggml.c`
- `src/ggml-quants.c`
- `src/ggml-cpu/quants.c`

GGML encodes tensor storage formats through `enum ggml_type`. The type enum includes dense types such as `F32`, `F16`, `BF16`, and integer types, alongside a large family of quantized formats such as `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, `Q2_K` through `Q8_K`, several `IQ*` formats, `TQ1_0`, `TQ2_0`, `MXFP4`, and `NVFP4`. The enum value `GGML_TYPE_COUNT = 41` covers both quantized and non-quantized entries, with some removed legacy slots preserved for compatibility.

The public helper APIs `ggml_is_quantized(...)`, `ggml_blck_size(...)`, `ggml_type_size(...)`, and `ggml_row_size(...)` expose the storage properties of each type. This is the basic contract that lets the rest of GGML reason about how many logical elements are packed into one block and how many bytes each stored block occupies.

The core metadata layer is the `type_traits[GGML_TYPE_COUNT]` table in `src/ggml.c`, exposed through `ggml_get_type_traits(...)`. Each entry records properties such as `type_name`, `blck_size`, `type_size`, and `is_quantized`, and the type-traits interface also carries conversion hooks such as `to_float` and `from_float_ref` where that format supports them.

Reference quantization and dequantization live in `src/ggml-quants.c`. The file includes deterministic reference routines such as `quantize_row_q4_0_ref(...)`, along with reference implementations for other families including K-quant formats such as `quantize_row_q2_K_ref(...)`. These routines define the canonical packing behavior used for model conversion and validation.

The CPU path adds execution-oriented quantization hooks through `struct ggml_type_traits_cpu` in `include/ggml-cpu.h`. That interface provides fields such as `from_float`, `vec_dot`, `vec_dot_type`, and `nrows`, and exposes them through `ggml_get_type_traits_cpu(...)`. This is the main CPU-side dispatch layer for optimized quantized kernels and mixed-format dot products.

GGML also contains separate backend directories for CUDA, Vulkan, and Metal, so quantized tensor types are part of a backend-agnostic representation even though execution can be implemented by different device backends. The exact optimized kernel coverage depends on the backend, but the quantized storage model is shared at the GGML type level.

In practice, the quantization system provides three main capabilities:

- **Format diversity**: multiple block-quantized and importance-aware formats, from classic `Q4_*` and `Q8_*` schemes to `K`-quant, `IQ`-quant, and newer formats such as `MXFP4` and `NVFP4`.
- **Type-traits dispatch**: per-type metadata and conversion hooks exposed through the GGML type-traits interface.
- **Backend-aware execution**: reference packing in `src/ggml-quants.c`, CPU-specialized execution hooks in `include/ggml-cpu.h`, and backend-specific execution layers in the device backends.

By storing tensor rows in packed low-precision blocks rather than dense FP32 arrays, GGML reduces model memory usage substantially. The exact compression ratio depends on the selected `ggml_type`, its block size, and its stored byte size.
