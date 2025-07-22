# SFPI

- https://github.com/tenstorrent/sfpi

**SFPI is the programming interface for the SFPU.** It provides a lightweight C++ wrapper over a custom RISC-V GCC compiler, which has been extended with vector data types and `__builtin` intrinsics to emit SFPU instructions. The primary design goal is to ensure that all code unrelated to SFPU instructions compiles to no-ops, thereby introducing zero runtime overhead. Over time, the intention is to migrate more functionality from the wrapper into the compiler itself.
