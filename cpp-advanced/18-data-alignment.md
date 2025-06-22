# Memory Alignment in C++

## Overview

Memory alignment is a low-level but critical aspect of C++ programming that affects performance, portability, and correctness. By aligning data structures and variables appropriately, developers can optimize CPU memory access patterns and interface safely with hardware or foreign memory layouts.

C++ compilers automatically align data based on type size and platform conventions. However, when interfacing with hardware, optimizing SIMD workloads, or reducing structure padding, explicit control using `alignas`, `alignof`, and related tools becomes essential.

## What Is Alignment?

**Alignment** refers to the requirement that data must be placed in memory at addresses that are multiples of a given power of two. For example:

- `char` → 1-byte alignment
- `short` → 2-byte alignment
- `int`, `float` → 4-byte alignment
- `double`, `long long` → 8-byte alignment

An object is said to be **naturally aligned** if its address is a multiple of its size. Otherwise, it is **misaligned**, which may result in inefficient or unsafe memory access on some architectures.

## Why Alignment Matters

- **Performance**: CPUs fetch aligned memory more efficiently.
- **Portability**: Ensures predictable layout across platforms.
- **Correctness**: Some architectures (e.g., ARM, SPARC) may crash on misaligned access.
- **Interoperability**: Necessary when interacting with memory-mapped hardware, networking protocols, or packed file formats.

## Structure Padding and Layout

Compilers insert **padding bytes** to ensure proper alignment of struct/class members. Consider:

```cpp
struct x_ {
    char a;   // 1 byte
    int b;    // 4 bytes
    short c;  // 2 bytes
    char d;   // 1 byte
};
```

To satisfy alignment:

- 3 padding bytes are inserted after `a` to align `b` to a 4-byte boundary.
- 1 padding byte is added at the end to ensure the structure size is a multiple of its largest member alignment.

The actual layout in memory:

| Offset   | Member         |
| -------- | -------------- |
| 0x00     | `char a`       |
| 0x01     | `char pad0[3]` |
| 0x04     | `int b`        |
| 0x08     | `short c`      |
| 0x0A     | `char d`       |
| 0x0B     | `char pad1[1]` |
| **Size** | **12 bytes**   |

For arrays of such structs, padding ensures natural alignment for each element.

## Understanding Address Alignment

A memory address `A` is said to be aligned to `N` if:

```
A % N == 0
```

Examples:

- Address `0x10` is aligned to 8: `0x10 % 8 == 0`
- Address `0x13` is misaligned to 4: `0x13 % 4 == 3`

## Controlling Alignment in C++

### `alignas`

Specifies the desired alignment of a variable, struct/class, or member:

```cpp
alignas(16) int vec[4]; // Aligns `vec` on a 16-byte boundary
```

```cpp
struct alignas(32) AlignedStruct {
    int x;
    double y;
};
```

If multiple `alignas` are applied, the largest one is chosen.

### `alignof`

Queries the required alignment of a type or object:

```cpp
std::cout << alignof(double); // Typically prints 8
```

## Example

```cpp
#include <iostream>

struct alignas(16) Bar {
    int i;
    int n;
    alignas(4) char arr[3];
    short s;
};

int main() {
    std::cout << "alignof(Bar): " << alignof(Bar) << "\n";
    std::cout << "sizeof(Bar): " << sizeof(Bar) << "\n";
}
```

**Expected output:**

```
alignof(Bar): 16
sizeof(Bar): [Depends on compiler, typically 16 or more]
```

## Advanced Tools

### `std::aligned_storage`

Provides a type-safe way to declare uninitialized, aligned memory:

```cpp
std::aligned_storage<sizeof(T), alignof(T)>::type storage;
```

Useful for custom allocators or placement-new construction.

### `std::aligned_union` (C++11, deprecated in C++23)

Specifies a union that satisfies the alignment of all its members:

```cpp
std::aligned_union<32, int, double>::type buffer;
```

## Platform-Specific Notes

- For **packed** layouts or legacy systems, use compiler-specific directives like `#pragma pack(n)` (MSVC) or `__attribute__((packed))` (GCC/Clang).
- Be cautious: packed structs can lead to slower access or hardware faults on strict-alignment platforms.

## Conclusion

While alignment is typically managed by the compiler, system-level developers must understand and control alignment for high-performance, low-level, or hardware-adjacent software. Use C++11 standard features like `alignas` and `alignof` for portable, robust alignment handling.
