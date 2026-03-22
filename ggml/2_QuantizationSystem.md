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

## Quantization Format Hierarchy

GGML organizes quantized storage formats through `enum ggml_type` in `include/ggml.h`, and the implementation maps each type to metadata in the `type_traits[GGML_TYPE_COUNT]` table in `src/ggml.c`. Each quantized type is identified by a name, block size, packed storage size, and optional conversion hooks such as `to_float` and `from_float_ref`.

A useful way to read the hierarchy is by format family:

- **Classic block quantization**: `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, and `Q8_1`. These are the traditional row-block formats, each with its own block struct such as `block_q4_0` or `block_q5_0`, and they have direct reference quantization and dequantization routines in `src/ggml-quants.c`.
- **K-quant super-block formats**: `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, and `Q8_K`. In `src/ggml-quants.c`, this family is explicitly grouped under the comment “2-6 bit quantization in super-blocks,” and in `src/ggml.c` these formats share `QK_K` as their block size.
- **IQ-family formats**: `IQ2_XXS`, `IQ2_XS`, `IQ3_XXS`, `IQ1_S`, `IQ4_NL`, `IQ3_S`, `IQ2_S`, `IQ4_XS`, and `IQ1_M`. These are separate named quantized types in the enum and type-traits table, and several of them use `QK_K`-sized blocks as well.
- **TQ-family formats**: `TQ1_0` and `TQ2_0`. These are distinct quantized types with their own block structs and reference quantizers, and they also use `QK_K`-sized blocks in the type-traits table.
- **FP4-family formats**: `MXFP4` and `NVFP4`. In the enum, `MXFP4` is annotated as “1 block” and `NVFP4` as “4 blocks, E4M3 scale,” and both have dedicated reference quantizers in `src/ggml-quants.c`.

At the metadata level, these families all plug into the same type-traits interface. For example, `Q4_0` is registered with type name `q4_0`, block size `QK4_0`, packed size `sizeof(block_q4_0)`, and both `to_float` and `from_float_ref` hooks; `Q2_K` is registered with type name `q2_K`, block size `QK_K`, packed size `sizeof(block_q2_K)`, and corresponding conversion hooks.

The IQ family follows the same metadata pattern but is not uniform in conversion support. For example, `IQ2_XXS` and `IQ2_XS` have `to_float` hooks but `from_float_ref = NULL`, while types such as `IQ3_XXS` and `IQ3_S` do provide reference quantizers.

The enum also preserves removed or deprecated slots for compatibility. `include/ggml.h` marks `Q4_2` and `Q4_3` as removed, and `src/ggml.c` keeps removed packed variants such as `Q4_0_4_4`, `Q4_0_4_8`, `Q4_0_8_8`, and several `IQ4_NL_*` entries as non-active placeholders with descriptive names.

So, in practical documentation terms, the hierarchy is:

1. classic row-block formats
2. K-quant super-block formats
3. IQ importance-oriented formats
4. TQ low-bit formats
5. FP4-style formats
6. compatibility placeholders for removed legacy encodings

## Quantization Format Details

### Classic 32-Value Block Formats

- `src/ggml-common.h`
- `src/ggml-quants.c`

The classic GGML quantized formats use fixed 32-value blocks: `QK4_0`, `QK4_1`, `QK5_0`, `QK5_1`, `QK8_0`, and `QK8_1` are all defined as 32 in `src/ggml-common.h`. Their block structs show the exact packed layout: `block_q4_0` stores one FP16 scale plus 16 bytes of packed 4-bit quants, `block_q4_1` adds an FP16 minimum, `block_q5_0` and `block_q5_1` add 4 bytes for the fifth bit plane, `block_q8_0` stores one FP16 scale plus 32 int8 values, and `block_q8_1` stores one FP16 scale plus one FP16 auxiliary sum term `s` and 32 int8 values.

Using those struct sizes, the effective storage cost and FP32 compression ratio are:

| Type   | Bits / value | Block layout                                                                         | FP32 compression ratio |
| ------ | -----------: | ------------------------------------------------------------------------------------ | ---------------------: |
| `Q4_0` |          4.5 | 1 × FP16 scale + 16 bytes packed 4-bit quants                                        |                   7.1× |
| `Q4_1` |          5.0 | 1 × FP16 scale + 1 × FP16 min + 16 bytes packed 4-bit quants                         |                   6.4× |
| `Q5_0` |          5.5 | 1 × FP16 scale + 4 bytes high bits + 16 bytes packed low 4-bit quants                |                   5.8× |
| `Q5_1` |          6.0 | 1 × FP16 scale + 1 × FP16 min + 4 bytes high bits + 16 bytes packed low 4-bit quants |                   5.3× |
| `Q8_0` |          8.5 | 1 × FP16 scale + 32 int8 quants                                                      |                   3.8× |
| `Q8_1` |          9.0 | 1 × FP16 scale + 1 × FP16 sum term + 32 int8 quants                                  |                   3.6× |

These numbers come directly from the block definitions and their `static_assert(sizeof(...))` checks in `src/ggml-common.h`. The compression ratio is the size of 32 FP32 values divided by the packed block size.

#### Format Semantics

The classic formats split into two main styles:

- **Symmetric-style formats**: `Q4_0`, `Q5_0`, and `Q8_0` store values around zero using a scale factor.
- **Affine-style formats**: `Q4_1` and `Q5_1` store both a scale and a minimum value, and dequantize with `x = q * d + m`.
- **Auxiliary-sum format**: `Q8_1` stores `d` and an extra FP16 field `s`, documented in the struct as `d * sum(qs[i])`.

You can see this directly in the dequantization code:

- `Q4_0`: `y = (q - 8) * d`
- `Q4_1`: `y = q * d + m`
- `Q5_0`: reconstruct the 5th bit from `qh`, then `y = (q - 16) * d`
- `Q5_1`: reconstruct the 5th bit from `qh`, then `y = q * d + m`
- `Q8_0`: `y = q * d`

#### Example: `Q4_0` Quantization

The reference implementation of `quantize_row_q4_0_ref(...)` in `src/ggml-quants.c` works block by block over 32 input floats:

1. scan the 32-value block and track the value with the largest absolute magnitude
2. set the scale as `d = max / -8`
3. compute `id = 1.0f / d` when `d != 0`
4. quantize each value relative to that scale
5. pack two 4-bit values into each byte of `qs[]`

A repo-aligned pseudocode version is:

```c
// For one Q4_0 block of 32 floats:
amax = 0
max  = 0

for j in 0..31:
    v = x[j]
    if abs(v) > amax:
        amax = abs(v)
        max  = v

d  = max / -8
id = (d != 0) ? 1.0f / d : 0.0f
store_fp16(y.d, d)

for j in 0..15:
    x0 = x[j]      * id
    x1 = x[j + 16] * id

    xi0 = min(15, (int8_t)(x0 + 8.5f))
    xi1 = min(15, (int8_t)(x1 + 8.5f))

    y.qs[j] = xi0 | (xi1 << 4)
```

That packing scheme matches the corresponding `dequantize_row_q4_0(...)` routine, which unpacks each nibble, subtracts 8, and multiplies by the stored scale `d`.

#### Notes

Two important details are easy to miss:

- `Q4_0` and `Q4_1` still operate on **32 logical values per block**, even though their `qs[]` arrays are only 16 bytes long because each byte stores two 4-bit quants.
- `Q5_0` and `Q5_1` do not store 32 separate 5-bit integers directly. They store the low 4 bits in `qs[]` and the extra high bit in `qh[]`, then reconstruct the full 5-bit code during dequantization.

### K-Quants

- `src/ggml-common.h`
- `src/ggml-quants.c`
- `src/ggml.c`

K-quant formats use `QK_K = 256`-value super-blocks. Instead of applying one scale to the entire block, they subdivide the super-block into smaller groups and store quantized per-group scale information. In `src/ggml-common.h`, this family is defined through `block_q2_K`, `block_q3_K`, `block_q4_K`, `block_q5_K`, `block_q6_K`, and `block_q8_K`, with comments documenting the grouping strategy and effective bits per weight. The reference quantization and dequantization routines for these formats are implemented in `src/ggml-quants.c`.

| Type   | Effective bits / weight | Super-block structure  | Notes                                                                   |
| ------ | ----------------------: | ---------------------- | ----------------------------------------------------------------------- |
| `Q2_K` |                   2.625 | 16 blocks of 16 values | affine form `x = a*q + b`; 4-bit quantized scales and mins              |
| `Q3_K` |                  3.4375 | 16 blocks of 16 values | scale-only form `x = a*q`; high-bit mask + low 2-bit storage            |
| `Q4_K` |                     4.5 | 8 blocks of 32 values  | affine form `x = a*q + b`; 6-bit quantized scales and mins              |
| `Q5_K` |                     5.5 | 8 blocks of 32 values  | affine form `x = a*q + b`; 6-bit quantized scales/mins + high-bit plane |
| `Q6_K` |                  6.5625 | 16 blocks of 16 values | scale-only form `x = a*q`; 8-bit scales                                 |
| `Q8_K` |        auxiliary format | 256 values             | used for intermediate quantization and dot products                     |

The block layouts show how each format is packed:

- `Q2_K` stores `scales[QK_K/16]`, `qs[QK_K/4]`, and two FP16 super-block coefficients `d` and `dmin`.
- `Q3_K` stores a high-bit mask, low 2-bit quants, a 12-byte scale array, and one FP16 super-block scale.
- `Q4_K` stores FP16 `d` and `dmin`, a 12-byte packed scale/min array, and `qs[QK_K/2]`.
- `Q5_K` extends `Q4_K` with an additional high-bit array `qh[QK_K/8]`.
- `Q6_K` stores lower 4 bits, upper 2 bits, 8-bit per-sub-block scales, and one FP16 super-block scale.
- `Q8_K` stores FP32 `d`, 256 int8 quants, and block sums for groups of 16.

A useful way to read this family is:

- `Q2_K`, `Q4_K`, and `Q5_K` are **affine** formats, where weights are reconstructed as `x = a*q + b`.
- `Q3_K` and `Q6_K` are **scale-only** formats, where weights are reconstructed as `x = a*q`.
- `Q8_K` is marked in the code as a helper format for **intermediate quantization and dot products**, not as a general-purpose model storage format.

### I-Quants (Importance-Oriented Quantization)

- `src/ggml-common.h`
- `src/ggml-quants.c`
- `src/ggml.c`

The IQ family uses grid-based and lookup-based encodings to push compression below the classic `Q*_K` formats. In `src/ggml-common.h`, these formats are defined as `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, and `IQ4_XS`, each with its own packed block layout. In `src/ggml-quants.c`, their quantizers use precomputed grids, lookup tables, neighbor searches, or non-linear codebooks rather than only uniform linear buckets.

| Type      | Effective bits / value | Packed structure                                                                   | Quantization style                         |
| --------- | ---------------------: | ---------------------------------------------------------------------------------- | ------------------------------------------ |
| `IQ1_S`   |                 1.5625 | `ggml_half d` + `qs[QK_K/8]` + `qh[QK_K/32]`                                       | 1-bit grid / indexed encoding              |
| `IQ1_M`   |                   1.75 | `qs[QK_K/8]` + `qh[QK_K/16]` + `scales[QK_K/32]`                                   | extended 1-bit grid encoding               |
| `IQ2_XXS` |                 2.0625 | `ggml_half d` + `uint16_t qs[QK_K/8]`                                              | compact 2-bit lookup-grid format           |
| `IQ2_XS`  |                 2.3125 | `ggml_half d` + `uint16_t qs[QK_K/8]` + `scales[QK_K/32]`                          | 2-bit lookup-grid format with extra scales |
| `IQ2_S`   |                 2.5625 | `ggml_half d` + `qs[QK_K/4]` + `qh[QK_K/32]` + `scales[QK_K/32]`                   | signed 2-bit grid format                   |
| `IQ3_XXS` |                 3.0625 | `ggml_half d` + `qs[3*QK_K/8]`                                                     | compact 3-bit lookup-grid format           |
| `IQ3_S`   |                 3.4375 | `ggml_half d` + `qs[QK_K/4]` + `qh[QK_K/32]` + `signs[QK_K/8]` + `scales[QK_K/64]` | enhanced 3-bit grid format                 |
| `IQ4_NL`  |                    4.5 | `ggml_half d` + `qs[QK4_NL/2]`                                                     | non-linear 4-bit codebook                  |
| `IQ4_XS`  |                   4.25 | `ggml_half d` + `scales_h` + `scales_l[QK_K/64]` + `qs[QK_K/2]`                    | super-block non-linear 4-bit format        |

The effective bits-per-value numbers above come directly from the packed block sizes in `src/ggml-common.h`. For example, `block_iq2_xxs` is documented as “(Almost) true 2-bit quantization” but the block layout adds one FP16 scale per 256-value block, so the effective cost is 2.0625 bits per weight rather than exactly 2.0. The same pattern appears for `IQ3_XXS`, which is documented as “(Almost) true 3-bit quantization” and packs to 3.0625 bits per weight.

#### Grid and Lookup Structure

The low-bit IQ formats are implemented around precomputed grids and maps in `src/ggml-quants.c`:

- `IQ2_XXS`, `IQ2_XS`, `IQ1_S`, `IQ1_M`, and `IQ2_S` share helpers such as `iq2_data_index(...)` and `iq2_grid_size(...)`, and select different grids such as `kgrid_2bit_256`, `kgrid_2bit_512`, `kgrid_1bit_2048`, and `kgrid_2bit_1024`.
- `IQ3_XXS` and `IQ3_S` use `iq3_data[...]` tables together with maps and neighbor tables to snap local blocks to valid 3-bit grid points.
- `IQ4_NL` and `IQ4_XS` use the non-linear value table `kvalues_iq4nl` through `quantize_row_iq4_nl_impl(...)`.

This is why the IQ family is better described as **grid-coded quantization** rather than plain linear quantization. The quantizer is not just scaling and rounding into evenly spaced buckets; it is selecting encodings from structured low-bit codebooks.

#### Importance Weights

At the GGML layer, the IQ quantizers accept an optional weighting input through the parameter:

```c
size_t quantize_iq2_xxs(
    const float * src,
    void * dst,
    int64_t nrow,
    int64_t n_per_row,
    const float * quant_weights);
```

The same `const float * quant_weights` parameter is used by `quantize_iq2_xs`, `quantize_iq2_s`, `quantize_iq3_xxs`, `quantize_iq3_s`, `quantize_iq1_s`, `quantize_iq1_m`, `quantize_iq4_nl`, and `quantize_iq4_xs`. Inside the quantizers, when `quant_weights` is present, it influences the per-element error weighting; when it is `NULL`, the routines fall back to internally derived weights such as `x[i] * x[i]` or related magnitude-based heuristics.

So, in GGML terms, IQ quantization supports **importance-weighted quantization**, but the API is phrased in terms of `quant_weights` rather than a hard-coded `imatrix` requirement. Some IQ formats also expose only dequantization in the generic type-traits table: for example, `IQ2_XXS`, `IQ2_XS`, `IQ1_S`, and `IQ1_M` have `from_float_ref = NULL`, while `IQ3_XXS`, `IQ3_S`, `IQ2_S`, `IQ4_NL`, and `IQ4_XS` register reference quantizers.

#### Notes

- `IQ4_NL` is called a “non-linear” 4-bit format, but its packed storage cost is 4.5 bits per value because each 32-value block also stores one FP16 scale.
- `IQ4_XS` uses a 256-value super-block and stores both high and low parts of the scale metadata, which is why its effective storage cost is 4.25 bits per value.
- The IQ family mixes two block granularities: most IQ formats use `QK_K = 256`, while `IQ4_NL` uses `QK4_NL = 32`.
