# `extern` in C++

The `extern` keyword in C++ is a powerful and multifaceted specifier that controls linkage, visibility, and language linkage conventions. It plays a critical role in managing declarations and definitions across translation units and interacting with C libraries or mixed-language projects. Misunderstanding or misusing `extern` can lead to linker errors or unintended symbol visibility. This document explores its meaning, contexts, and best practices.

## 1. External Linkage and the One-Definition Rule

In C++, global (non-`static`) variables and functions have external linkage by default, meaning their names are visible across translation units. The `extern` keyword reinforces or declares this explicitly. It is typically used to:

- Declare a variable or function that is defined in another translation unit.
- Prevent multiple definitions during the linking phase.
- Share constants or global data between modules.

### Example

```cpp
// fileA.cpp
int global_value = 42;  // definition

// fileB.cpp
extern int global_value;  // declaration (no storage)
```

Declaring `extern int global_value;` in multiple files is safe as long as the actual definition occurs in only one place. Defining it in more than one file (even with `extern`) results in a **LNK2005** or **ODR violation**.

## 2. `extern` with `const`

By default, `const` variables have **internal linkage**. This means they are only visible within the translation unit where they're declared. If you want to share a `const` variable across translation units, you must use `extern`.

```cpp
// fileA.cpp
extern const int MaxSize = 100;  // definition with external linkage

// fileB.cpp
extern const int MaxSize;        // declaration
```

Note: Omitting `extern` in the definition makes `MaxSize` internal to `fileA.cpp`, and the reference in `fileB.cpp` will be undefined at link time.

## 3. `extern constexpr` and Visual Studio Compatibility

C++ standard mandates that `constexpr` variables have internal linkage unless explicitly marked `extern`. This allows placing constexpr variables in headers, but care is needed.

```cpp
// In header file
extern constexpr int kLimit = 64;  // Can be included in multiple TUs
```

On MSVC, to avoid multiple definition linker errors, use `__declspec(selectany)`:

```cpp
extern constexpr __declspec(selectany) int kLimit = 64;
```

Use `/Zc:externConstexpr` to enforce standard-conforming behavior on MSVC.

## 4. `extern "C"`: Language Linkage

C++ uses _name mangling_ to support function overloading, but C does not. If you need to call C functions from C++, use `extern "C"` to disable name mangling:

```cpp
extern "C" int printf(const char*);
```

You can also wrap entire headers:

```cpp
extern "C" {
    #include <stdio.h>
}
```

### C++ Calling C-defined Functions

When using third-party C libraries, always ensure:

- Functions are declared with `extern "C"` in C++.
- You link with the correct `.c` or `.o` files compiled in C mode.

## 5. Multiple Declarations and Linkage Conflicts

**First declaration rules:**
If a function or variable is declared multiple times, the first declaration determines the linkage. For example:

```cpp
extern "C" int cfunc();  // sets C linkage

int cfunc();             // OK: redundant, retains C linkage
```

But:

```cpp
int cfunc();             // first declaration: C++ linkage

extern "C" int cfunc();  // Error: conflicting linkage
```

## 6. `extern` with Templates

Template instantiations in C++ can be controlled using `extern` to avoid duplication:

```cpp
// fileA.cpp
template class std::vector<int>;  // explicit instantiation

// fileB.cpp
extern template class std::vector<int>;  // avoid re-instantiating
```

This technique minimizes compile time and binary size when using the same instantiation across multiple TUs.

## 7. Best Practices

- **Use `extern` in headers only for declarations**, never for definitions unless you use `inline` or `selectany`.
- **Avoid global variables**, but if necessary, use `extern` carefully with proper documentation.
- When mixing C and C++, always wrap C declarations with `extern "C"` and use `#ifdef __cplusplus` guards.
- Be aware of **platform-specific behaviors**, especially with MSVC `/Zc:*` switches and linker semantics.

## Summary Table

| Use Case                      | Behavior                                         |
| ----------------------------- | ------------------------------------------------ |
| `extern int x;`               | Declares a global variable `x` defined elsewhere |
| `extern const int y = 10;`    | Const global with external linkage               |
| `extern constexpr int z = 5;` | Constexpr global, needs care on MSVC             |
| `extern "C"`                  | C linkage (no name mangling)                     |
| `extern template`             | Avoids re-instantiation of templates across TUs  |
