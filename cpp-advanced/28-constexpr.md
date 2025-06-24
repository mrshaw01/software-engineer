# `constexpr` in C++

Introduced in C++11 and significantly enhanced in C++14, C++17, and C++20, the `constexpr` keyword enables expressions to be evaluated at compile time. It empowers developers to write functions, variables, and class constructors whose results are known and embedded during compilation, offering opportunities for performance improvements, stronger type safety, and more robust metaprogramming.

## Purpose and Motivation

`constexpr` enables _compile-time computation_. Unlike `const`, which guarantees immutability but allows run-time initialization, `constexpr` requires immediate evaluation at compile time. This distinction has several implications:

- Enables compile-time constant folding and code generation optimizations.
- Ensures that APIs depending on constant expressions—like array bounds, template arguments, and alignment declarations—can leverage user-defined logic.
- Promotes safer and more predictable code by shifting failure detection to compilation.

## Syntax and Semantics

```cpp
constexpr type identifier = constant_expression;
constexpr type identifier { constant_expression };
constexpr return_type function(params);
```

A `constexpr` variable or function must:

- Be initialized with a constant expression.
- Operate on _literal types_ (integral types, floating-point types, pointers, or custom types satisfying literal type constraints).

C++14 and newer relax many restrictions. For instance:

- `constexpr` functions may include local variables, branching, and loops.
- They may mutate local state as long as the results remain deterministically computable at compile time.

## Example: Basic Usage

```cpp
constexpr int square(int x) {
    return x * x;
}

constexpr int val = square(10);  // Computed at compile time
```

If the compiler _requires_ a constant (e.g., template arguments, `static_assert`, array bounds), it will enforce compile-time evaluation.

## `constexpr` vs `const`

| Feature                 | `const`               | `constexpr`            |
| ----------------------- | --------------------- | ---------------------- |
| Immutability            | Yes                   | Yes                    |
| Compile-time required?  | No (run-time allowed) | Yes (must be constant) |
| Usable in templates?    | Sometimes             | Always                 |
| Can apply to functions? | No                    | Yes                    |

## Compile-Time Functions

A `constexpr` function behaves like a regular function when used in run-time context, and like a constant evaluator when used where compile-time is required:

```cpp
constexpr int factorial(int n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
}

constexpr int f5 = factorial(5); // 120, at compile time
```

As of C++14:

- Loops and branching (if, switch) are allowed.
- Local variables can be declared (must be literal types).
- Functions may be recursive.

## `constexpr` Constructors and User Types

```cpp
class Vec2 {
public:
    constexpr Vec2(float x, float y) : x_(x), y_(y) {}
    constexpr float length_squared() const { return x_ * x_ + y_ * y_; }

private:
    float x_, y_;
};

constexpr Vec2 origin(0.0f, 0.0f);
constexpr float len_sq = origin.length_squared();
```

To be `constexpr`, constructors and all methods used in constexpr contexts must themselves be constexpr.

## C++20 Enhancements

C++20 further strengthens `constexpr` by allowing:

- Virtual functions to be `constexpr`
- `dynamic_cast` and `typeid` in limited constexpr contexts
- `try/catch` inside `constexpr` (evaluated as immediate context failures)

This allows for more expressive compile-time computations and even constexpr-polymorphism.

## Advanced Use Cases

### Template Metaprogramming

```cpp
template <int N>
struct Factorial {
    static constexpr int value = N * Factorial<N - 1>::value;
};

template <>
struct Factorial<0> {
    static constexpr int value = 1;
};
```

Can now be replaced by:

```cpp
constexpr int factorial(int n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
}
```

Much more readable and flexible.

### Static Assertions

```cpp
static_assert(factorial(5) == 120, "factorial(5) should be 120");
```

## `extern constexpr`

By default, `constexpr` variables have internal linkage. In modern toolchains (e.g., MSVC with `/Zc:externConstexpr`), `extern constexpr` gives external linkage when specified:

```cpp
// header.hpp
extern constexpr int default_size;

// source.cpp
constexpr int default_size = 42;
```

This is critical for constexpr values that must be shared across translation units.

## Debugging Tip

In Visual Studio or similar debuggers, if a breakpoint inside a `constexpr` function is not hit, it was evaluated at compile time. Otherwise, it was executed at run time.

## Best Practices

- Use `constexpr` to avoid magic numbers and hard-coded values in template metaprogramming.
- Prefer `constexpr` over `const` where compile-time evaluation is beneficial.
- Avoid overusing `constexpr` in complex runtime logic; balance readability and compile-time efficiency.
- Be cautious of compile-time bloat—constexpr recursion and evaluation can lead to long compilation times.

## Summary

`constexpr` enables C++ programs to execute logic at compile time, promoting efficiency and correctness. With each C++ standard, its capabilities have expanded, making it an indispensable tool for performance-critical and template-heavy domains, such as embedded systems, HPC, and compile-time reflection frameworks.

Understanding and applying `constexpr` correctly is essential for modern C++ software engineers aiming to write efficient, maintainable, and expressive code.
