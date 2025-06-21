# C++ Type System

C++ is a statically and strongly typed language. Every variable, function parameter, and expression has an associated type known at compile time. Understanding C++’s type system is essential for writing safe, performant, and maintainable code.

## Overview

In C++, a type defines:

- How much memory is allocated
- What values can be stored
- What operations are valid
- How the compiler interprets the underlying bits

### Type Examples

```cpp
int x = 42;               // Integral type
double y = 3.14;          // Floating-point type
std::string name = "Ada"; // User-defined class type
```

C++ also supports user-defined types (classes, structs, enums), enabling abstraction and encapsulation of behavior.

## Type Categories

| Category                 | Description                                                                                     |
| ------------------------ | ----------------------------------------------------------------------------------------------- |
| **Scalar Types**         | Hold single values (e.g., `int`, `double`, `bool`, pointers).                                   |
| **Compound Types**       | Include arrays, functions, classes, references, and unions.                                     |
| **POD (Plain Old Data)** | Types compatible with C: no user-defined ctor/dtor, no inheritance, no virtual functions.       |
| **Object**               | Any instance of a type occupying memory. In this guide, we refer to all instances as "objects". |

## Type Qualifiers

### `const`

Makes a value immutable after initialization.

```cpp
const int max_users = 100;
```

### `auto`

Allows type inference from the initializer.

```cpp
auto score = 88.5;  // Deduced as double
```

> ❌ `auto x;` → Error: No initializer, cannot deduce type.

## Built-in Types

| Type           | Size    | Notes                              |
| -------------- | ------- | ---------------------------------- |
| `int`          | 4 bytes | Default for integers               |
| `double`       | 8 bytes | Default for floating-point         |
| `bool`         | 1 byte  | `true` or `false`                  |
| `char`         | 1 byte  | ASCII character                    |
| `wchar_t`      | 2 bytes | Wide character (UTF-16 on Windows) |
| `unsigned int` | 4 bytes | Non-negative integers              |
| `long long`    | 8 bytes | Larger integer range               |

The C++ standard does not fix exact sizes. Use `<cstdint>` types like `int32_t` when precise width matters.

## User-defined Types

Custom types are created via:

- `class` / `struct`
- `enum`
- `union`

Example:

```cpp
struct Point {
    int x, y;
};
```

You can overload operators and define conversions for these types.

## String Types

C++ offers no built-in string type, but the Standard Library provides:

- `std::string`: for `char`-based text
- `std::wstring`: for `wchar_t`-based text

```cpp
#include <string>
std::string message = "Hello, world!";
```

Avoid C-style null-terminated strings (`char[]`) in modern C++.

## Pointer Types

### Raw Pointers

```cpp
int value = 42;
int* ptr = &value;
*ptr = 10;  // Dereference to modify
```

> ⚠️ Dereferencing uninitialized or dangling pointers is **undefined behavior**.

### Smart Pointers

Prefer `std::unique_ptr`, `std::shared_ptr`, or `std::weak_ptr` for ownership management.

```cpp
#include <memory>
std::unique_ptr<MyClass> obj = std::make_unique<MyClass>();
```

Avoid manual `new` / `delete` in modern C++.

## Void Type

- `void`: signifies absence of a value
- `void*`: raw untyped memory pointer (discouraged in modern C++)

```cpp
void doNothing();
void* buffer = malloc(100);  // C-style, discouraged
```

## Windows-specific Types

Windows SDK defines aliases like:

```cpp
typedef int INT;
typedef unsigned long DWORD;
typedef void* LPVOID;
```

- Use fundamental types unless a Windows type adds semantic clarity (e.g., `HRESULT`).

## Best Practices

- Use `auto` for type inference when it improves readability
- Prefer `const` to enforce immutability
- Initialize all variables before use
- Avoid raw pointers for ownership; use smart pointers
- Favor `std::string` over C-style strings
- Avoid casting unless absolutely necessary; use `static_cast` or `reinterpret_cast` judiciously
- Understand the lifetime and size of your types to avoid undefined behavior
