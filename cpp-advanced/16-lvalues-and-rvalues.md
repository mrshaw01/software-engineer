# Lvalues and Rvalues in C++

## Overview

In C++, every expression has both a type and a **value category**. These categories influence how values are accessed, copied, moved, and stored during execution. Starting with C++11, the language introduced a refined taxonomy of expression categories, enabling more efficient memory management and precise expression semantics.

```
expression
├── glvalue
│   ├── lvalue
│   └── xvalue
└── rvalue
    ├── xvalue
    └── prvalue
```

## Value Category Definitions

### glvalue (Generalized Lvalue)

- Refers to an identifiable object, bit-field, or function.
- Examples: `x`, `*ptr`, `arr[2]`, `obj.member`

### prvalue (Pure Rvalue)

- Computes a value or initializes an object but does not identify a resource with a stable address.
- Examples: `42`, `true`, `std::string("abc")`, function calls returning by value

### xvalue (Expiring Value)

- A glvalue that denotes an object whose resources can be reused.
- Examples: `std::move(x)`, a function returning an rvalue reference, `std::string("a") + "b"`

### lvalue

- A glvalue that is not an xvalue; it refers to a persistent object with identity.
- Examples: variable names, dereferenced pointers, function calls returning lvalue references

### rvalue

- Either a prvalue or an xvalue. Typically represents temporary objects.
- Examples: literals, temporaries, move results

## Addressability and Lifetime

| Category | Addressable | Temporary | Can Bind to `T&` | Can Bind to `T&&` |
| -------- | ----------- | --------- | ---------------- | ----------------- |
| lvalue   | Yes         | No        | Yes              | No                |
| xvalue   | Yes         | Yes       | No               | Yes               |
| prvalue  | No          | Yes       | No               | Yes               |

## Practical Examples

```cpp
int i, j, *p;

// Correct: lvalue = prvalue
i = 7;

// Error: prvalue cannot be on the left-hand side
7 = i;          // Invalid
j * 4 = 7;      // Invalid

// Correct: dereferencing a pointer yields an lvalue
*p = i;

// Correct: conditional operator yields lvalue if both operands are lvalues
((i < 3) ? i : j) = 7;

// Error: const lvalue cannot be assigned
const int ci = 7;
ci = 9;         // Invalid
```

> Note: Operator overloading can change the categorization of an expression. For example, `j * 4` could become an lvalue if the `*` operator is overloaded to return a reference.

## Why It Matters

Understanding expression categories is critical for:

- Efficient API design (e.g., move constructors, `T&&` parameters)
- Avoiding unnecessary copies
- Enabling move semantics
- Writing generic code with `std::move` and `std::forward`

These categories form the foundation of resource management, particularly in the context of modern C++ features like RAII, smart pointers, and perfect forwarding.

## Summary

- **lvalue**: Named, persistent object
- **prvalue**: Temporary value with no identity
- **xvalue**: Temporary object that can be moved from
- **glvalue** = lvalue + xvalue (has identity)
- **rvalue** = prvalue + xvalue (can be moved)
