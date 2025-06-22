# `noexcept` in C++

### Overview

The `noexcept` specifier in C++ indicates whether a function **is guaranteed not to throw exceptions**. Introduced in C++11 and refined in later standards, it is a compile-time contract that enables **optimizations**, improves **code correctness**, and helps enforce **exception safety guarantees**.

### Syntax

```cpp
void func() noexcept;                     // Does not throw
void func() noexcept(true);              // Same as above
void func() noexcept(false);             // May throw

template<typename T>
void f(T t) noexcept(noexcept(g(t)));    // Conditional based on another expression
```

### Benefits

1. **Optimization**
   Functions marked `noexcept` enable the compiler to apply certain optimizations such as:

   - Skipping exception unwinding code
   - Using `move` instead of `copy` in STL containers (`std::vector` growth prefers `noexcept` move constructors)

2. **Better Diagnostics**
   Marking functions `noexcept` lets the compiler detect and warn about unexpected throws during static analysis.

3. **Enforcing Contracts**
   Signals developer intent. Violations result in `std::terminate()` being called, avoiding undefined behavior.

### Practical Use Cases

#### 1. Move Constructors and Move Assignment Operators

Always mark them `noexcept` if they truly don’t throw. This is crucial for STL compatibility.

```cpp
class MyType {
public:
    MyType(MyType&&) noexcept; // Safe to move
    MyType& operator=(MyType&&) noexcept;
};
```

#### 2. Utility Functions or Destructors

Functions that obviously won’t throw should be marked `noexcept` to reflect intent and support optimizations.

```cpp
void reset() noexcept;
~MyType() noexcept;
```

#### 3. Conditional `noexcept`

Sometimes whether a function throws depends on the types it's templated on:

```cpp
template <typename T>
void wrapper(T&& t) noexcept(noexcept(process(std::forward<T>(t)))) {
    process(std::forward<T>(t));
}
```

### Common Pitfalls

- **Forgetting `noexcept` on move constructors** leads to unnecessary copies.
- **Incorrectly using `noexcept`** (i.e., marking something `noexcept` when it may throw) results in runtime termination.
- Don’t add `noexcept` **just for style**—only when it's semantically accurate.

### Guidelines

| Context                      | Should it be `noexcept`?        |
| ---------------------------- | ------------------------------- |
| Destructor                   | Yes (always `noexcept` if safe) |
| Move Constructor/Assignment  | Yes (critical for STL use)      |
| Accessor functions           | Often yes                       |
| Complex operations           | Only if guaranteed not to throw |
| Template forwarding wrappers | Use `noexcept(...)` expression  |

### Final Notes

Use `noexcept` as both an optimization tool and a self-documenting code contract. It's an essential part of writing robust, performant, and modern C++ code—especially in large systems or library development.
