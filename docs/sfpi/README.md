# SFPI

- https://github.com/tenstorrent/sfpi

**SFPI is the programming interface for the SFPU.** It provides a lightweight C++ wrapper over a custom RISC-V GCC compiler, which has been extended with vector data types and `__builtin` intrinsics to emit SFPU instructions. The primary design goal is to ensure that all code unrelated to SFPU instructions compiles to no-ops, thereby introducing zero runtime overhead. Over time, the intention is to migrate more functionality from the wrapper into the compiler itself.

1. Overview of SFPI
2. Architecture and Compiler Integration
3. Data Types and Register Model
4. Constants and Immediate Values
5. Control Flow and Predication
6. Operators and Conversions
7. Instruction-Level Intrinsics and Utilities
8. Compilation and Build Workflow
9. Testing and Validation
10. Optimization and Performance Considerations
11. Register Pressure and Limitations
12. Emulation Support
13. Licensing, Versioning, and Release Management
