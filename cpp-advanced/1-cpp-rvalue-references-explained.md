# C++ Rvalue References Explained

**Author**: Shaw Nguyen
**Last Updated**: June 2025

## Table of Contents

1. [Introduction](#introduction)
2. [Move Semantics](#move-semantics)
3. [Rvalue References](#rvalue-references)
4. [Forcing Move Semantics](#forcing-move-semantics)
5. [Are Rvalue References Rvalues?](#are-rvalue-references-rvalues)
6. [Compiler Optimizations and Move Semantics](#compiler-optimizations-and-move-semantics)
7. [Perfect Forwarding: The Problem](#perfect-forwarding-the-problem)
8. [Perfect Forwarding: The Solution](#perfect-forwarding-the-solution)
9. [Rvalue References and Exceptions](#rvalue-references-and-exceptions)
10. [Implicit Move Semantics](#implicit-move-semantics)
11. [Summary and Best Practices](#summary-and-best-practices)
12. [Further Reading](#further-reading)

## Introduction

Rvalue references (`T&&`) were introduced in C++11 to address two core needs:

- Efficient **move semantics**
- Flexible **perfect forwarding**

To understand their motivation, it’s important to revisit the concepts of lvalues and rvalues.

- **Lvalue**: Refers to a memory location; can have its address taken using `&`.
- **Rvalue**: A temporary value that does not have a persistent memory address.

```cpp
int a = 42;
a = a * 2;        // Valid
int* p = &a;       // Valid

int b = a * 2;     // Valid
int* q = &(a * 2); // Invalid: rvalue has no address
```

## Move Semantics

Classes with dynamically allocated resources traditionally relied on deep copies:

```cpp
X& X::operator=(const X& rhs);
```

C++11 allows a more efficient form:

```cpp
X& X::operator=(X&& rhs);
```

This move assignment steals resources from `rhs`:

```cpp
X foo();
X x;
x = foo(); // Uses move assignment, not copy
```

## Rvalue References

An rvalue reference is written as `T&&` and behaves like a normal reference, except:

- Lvalues bind to `T&`
- Rvalues bind to `T&&`

```cpp
void foo(X&);  // lvalue
void foo(X&&); // rvalue

X x;
foo(x);       // Calls foo(X&)
foo(X());     // Calls foo(X&&)
```

This mechanism is primarily useful for implementing move constructors and move assignment operators.

```cpp
X(X&& rhs) noexcept;
X& operator=(X&& rhs) noexcept;
```

## Forcing Move Semantics

Named variables are lvalues—even if declared as rvalue references. Use `std::move` to explicitly convert them:

```cpp
template<class T>
void swap(T& a, T& b) {
    T tmp(std::move(a));
    a = std::move(b);
    b = std::move(tmp);
}
```

`std::move` is a cast, not a move operation.

## Are Rvalue References Rvalues?

Surprisingly, no. If an rvalue reference has a **name**, it behaves as an lvalue.

```cpp
void foo(X&& x) {
    X y = x;           // x is an lvalue, calls copy constructor
    X z = std::move(x); // Correctly uses move constructor
}
```

Unnamed temporaries remain rvalues:

```cpp
X&& create();
X y = create(); // Calls move constructor
```

## Compiler Optimizations and Move Semantics

Avoid forcing move semantics where the compiler already applies **Return Value Optimization (RVO)**:

```cpp
X foo() {
    X x;
    return x; // RVO applies
    // return std::move(x); // Often disables RVO — avoid
}
```

## Perfect Forwarding: The Problem

Templated wrapper functions can interfere with value categories:

```cpp
template<typename T>
void create(T arg) { new MyType(arg); } // Always copies
```

Overloading for all combinations of reference types doesn't scale.

## Perfect Forwarding: The Solution

Use a **universal reference** (`T&&`) and `std::forward`:

```cpp
template<typename T>
void create(T&& arg) {
    new MyType(std::forward<T>(arg));
}
```

### How it Works

- Called with lvalue: `T = U&` → `T&&` becomes `U&`
- Called with rvalue: `T = U` → `T&&` becomes `U&&`

`std::forward` preserves the original value category.

```cpp
template<typename U>
U&& forward(remove_reference_t<U>& a) {
    return static_cast<U&&>(a);
}
```

## Rvalue References and Exceptions

To maximize usability in the STL:

- Ensure move constructors/assignments are marked `noexcept`
- This allows safe use in containers like `std::vector`

```cpp
X(X&& rhs) noexcept;
X& operator=(X&& rhs) noexcept;
```

## Implicit Move Semantics

The compiler can generate move constructors/assignments:

```cpp
class MyClass {
    MyClass(MyClass&&) = default;
    MyClass& operator=(MyClass&&) = default;
};
```

However, this behavior is restricted to prevent breaking legacy code. For resource-managing types, it's best to explicitly define move behavior.

## Summary and Best Practices

- Use `T&&` to enable move semantics and perfect forwarding.
- Use `std::move` to cast lvalues to rvalues.
- Use `std::forward<T>` inside templates to preserve value category.
- Always mark move operations as `noexcept` where possible.
- Don't overuse `std::move` — let RVO do its job when applicable.

## Further Reading

- Scott Meyers — _Effective Modern C++_: Items 14, 17, 25, 41
- cppreference: [`std::move`](https://en.cppreference.com/w/cpp/utility/move), [`std::forward`](https://en.cppreference.com/w/cpp/utility/forward)
- Howard Hinnant, Bjarne Stroustrup — _Rvalue References: C++ Source_

> Mastering rvalue references is essential for writing high-performance modern C++. Know the rules, respect the subtleties, and let the compiler work for you.
