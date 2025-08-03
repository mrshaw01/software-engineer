# SFPI Overview

**SFPI** is a C++ interface designed for programming Tenstorrent’s **SFPU**. It is intended to emit highly optimized vector instructions with minimal overhead and provides direct access to low-level SFPU functionality.

## Key Characteristics

- **Zero-overhead abstraction**: The C++ wrapper is designed so that only SFPU instructions generate code; all other constructs compile away to nothing.
- **Target architectures**: Supports **Grayskull** and **Wormhole B0** hardware.
- **Compiler base**: Built on a modified **RISC-V GCC toolchain** with TT-specific vector extensions and intrinsics.

## Toolchain and Components

SFPI includes:

- Enhanced RISC-V GCC and binutils submodules.
- Standard newlib and qemu for testing and emulation.
- Custom tests, build scripts, and a DejaGNU-based test harness.

## Core Programming Model

### Vector Types

Located in the `sfpi` namespace:

- `vFloat`, `vInt`, `vUInt`: Strongly typed vector wrappers over `__rvtt_vec_t`.
- Conversion between types uses `reinterpret<>`.
- Operations include standard arithmetic, logical, and conditional comparisons.

### Register Access

- `dst_reg[]`: Array for destination registers.
- `l_reg[]`: Array to access specific hardware LRegs.
- `LRegs`: Enum for SFPU general-purpose vector registers.

### Control Flow

Predicated execution is enabled using macros:

- `v_if`, `v_elseif`, `v_else`, `v_endif`: Enable conditional vector execution.
- `v_block`, `v_endblock`, `v_and`: For compound predicate narrowing.
- Note: `v_endif` is mandatory to match `{` inserted by `v_if`.

### Constants

Vector constants are expressed as scalars and expanded to vector width. For example:

- Grayskull: `vConst0`, `vConst1`, `vConstTileId`, etc.
- Wormhole: Adds programmable constants like `vConstFloatPrgm0`.

## Compiler and Build

- Compiler flags:

  ```
  -m<arch> -fno-exceptions
  ```

  with `arch` as `grayskull` or `wormhole`.

- Disabling compiler features:

  ```
  -fno-rvtt-sfpu-warn
  -fno-rvtt-sfpu-combine
  -fno-rvtt-sfpu-cc
  -fno-rvtt-sfpu-replay
  ```

- Build process:

  1. Clone repo and init submodules.
  2. Use `scripts/build.sh` to compile.
  3. Tests can be run to validate generated assembly against gold standards.
  4. Releases created using `scripts/release.sh`.

## Optimization Philosophy

- SFPI emphasizes **programmer-driven performance**.
- Optimizations include:

  - Instruction combining (e.g., MUL + ADD → MAD)
  - Constant folding and implicit instruction reordering
  - CC enable/disable optimization
  - **SFPREPLAY**: Wormhole feature to replay repeated instruction sequences

## Emulator and Testing

- A fast \_\_builtin-level emulator is included for rapid development.
- DejaGNU test framework supports:

  - Full toolchain testing (`--test`)
  - Targeted GCC or binutils testing (`--test-gcc`, `--test-binutils`)
  - TT-specific test execution (`--test-tt`)

## Limitations

- **No ABI support**: All functions must be `sfpi_inline` (always inlined).
- **No register spilling**: Grayskull has 4 LRegs, Wormhole has 8.
- **Unclear error messages**: Errors may trace back to wrapper macros.
- **Vector stack passing not supported**: Only immediate registers are used.

This structure allows efficient and predictable vector programming on Tenstorrent NPUs, with careful control over compilation and hardware-level execution.
