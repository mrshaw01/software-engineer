# Translation Units and Linkage in C++

## Overview

In C++, understanding how symbols are declared, defined, and linked across translation units is critical for building modular and reliable software. This document explains the core concepts related to translation units, linkage types, and the `extern` keyword.

## One Definition Rule (ODR)

- **Declaration**: Introduces a symbol (e.g., variable, function, class) and its type but not its complete definition.
- **Definition**: Provides full details to allocate memory (for variables) or executable code (for functions).

Each name can be declared multiple times but **defined only once** per program.

### Examples

**Declarations**:

```cpp
extern int i;
int f(int x);
class C;
```

**Definitions**:

```cpp
int i{42};
int f(int x) { return x * i; }
class C {
public:
    void DoSomething();
};
```

## Translation Units

A translation unit is:

- An implementation file (`.cpp`, `.cxx`) and
- All headers included by it, directly or indirectly.

Each translation unit is compiled **independently**. After compilation, the linker merges all translation units into a single executable.

> ODR violations (e.g., multiple definitions of the same symbol) often result in **linker errors**.

## Header Files and Include Guards

To share declarations across translation units:

- Declare in a `.h` or `.hpp` header file.
- Use `#include` in every `.cpp` file that needs the declaration.
- Use include guards (`#ifndef / #define / #endif`) or `#pragma once` to prevent multiple inclusion.

**Define** the symbol in **only one** `.cpp` file.

> Starting with C++20, **modules** provide a more robust alternative to header files.

## Linkage Types

### 1. **External Linkage**

- Symbol is visible across all translation units.
- Default for:

  - Non-`const` global variables
  - Free functions

### 2. **Internal Linkage**

- Symbol is visible **only within** the same translation unit.
- Enabled by:

  - `static` keyword at namespace/global scope
  - `const`, `constexpr`, `typedef` by default

**Example**:

```cpp
// fileA.cpp
static int x = 1;       // internal linkage
const int y = 2;        // also internal
```

To give a `const` global variable **external linkage**:

```cpp
extern const int value = 42;
```

## The `extern` Keyword

### 1. **For Non-const Globals**

```cpp
// fileA.cpp
int i = 42;

// fileB.cpp
extern int i;  // refers to i in fileA
```

Avoid defining the same variable in multiple `.cpp` files:

```cpp
// fileD.cpp
int i = 43;           // Linker error
extern int i = 43;    // Still an error
```

### 2. **For Const Globals**

```cpp
// fileA.cpp
extern const int i = 42;

// fileB.cpp
extern const int i;
```

### 3. **For `constexpr` Globals**

Older compilers gave `constexpr` internal linkage regardless of `extern`. Use:

```cpp
extern constexpr __declspec(selectany) int x = 10;
```

## `extern "C"` for C Linkage

Used to declare functions or variables with **C-style** linkage, preventing name mangling.

### Example

```cpp
extern "C" int printf(const char* fmt, ...);

extern "C" {
    char ShowChar(char ch);
    char GetChar(void);
}

extern "C" char ShowChar(char ch) {
    putchar(ch);
    return ch;
}
```

- Applies to functions or `#include`s
- Required to call C libraries from C++

**Note**: `extern "C"` must be used in the **first declaration** of the function.

### Invalid Usage Example

```cpp
int CFunc2();
extern "C" int CFunc2(); // Error: linkage mismatch
```

### Compatibility Notes

- MSVC allows `"C"` and `"C++"` in the linkage string.
- You cannot overload functions declared `extern "C"`.

## Summary

| Use Case                   | Default Linkage      | Override With          |
| -------------------------- | -------------------- | ---------------------- |
| Non-const globals          | External             | `static`               |
| Const globals              | Internal             | `extern`               |
| `constexpr` globals        | Internal (pre-C++17) | `extern` + `selectany` |
| Functions                  | External             | `static`, `extern "C"` |
| Class members, local names | No linkage           | N/A                    |

## Best Practices

- Use headers for declarations, and implementation files for definitions.
- Prefer `constexpr`/`const` for global constants.
- Use `extern "C"` correctly to interoperate with C libraries.
- Avoid unnecessary global variables; prefer function-local or class-member scope.
- Minimize ODR violations by careful separation of declarations and definitions.
