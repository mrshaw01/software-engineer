# Lambda Expressions in Modern C++

It’s essential to understand lambda expressions not only as syntactic sugar, but as a foundational tool in writing concise, expressive, and high-performance C++ code. Introduced in C++11 and progressively enhanced through C++14, C++17, and C++20, lambda expressions enable the definition of anonymous function objects (closures) in place—improving code locality and reducing boilerplate.

## **1. What is a Lambda Expression?**

A lambda expression is an anonymous function object that can capture variables from its enclosing scope. Its primary utility lies in defining short, inline operations—typically passed to STL algorithms, asynchronous tasks, or event handlers.

### **Basic Syntax**

```cpp
[capture](parameters) -> return_type {
    // function body
}
```

Each component is optional except for the capture clause.

## **2. Example: Using Lambda with `std::sort`**

```cpp
#include <algorithm>
#include <cmath>
#include <vector>

void abssort(float* x, unsigned n) {
    std::sort(x, x + n, [](float a, float b) {
        return std::abs(a) < std::abs(b);
    });
}
```

**Explanation:**
A lambda is passed as a custom comparator to `std::sort`, ordering elements by absolute value. This eliminates the need for a separate function or function object.

## **3. Breakdown of Lambda Syntax**

Consider this lambda:

```cpp
[=]() mutable noexcept -> int { return x + y; }
```

| Component           | Meaning                                            |
| ------------------- | -------------------------------------------------- |
| `[=]`               | Capture all external variables by value            |
| `()`                | No parameters                                      |
| `mutable`           | Allows modification of captured-by-value variables |
| `noexcept`          | Promises not to throw exceptions                   |
| `-> int`            | Trailing return type                               |
| `{ return x + y; }` | Lambda body                                        |

## **4. Capture Clause (`[]`)**

The capture clause governs access to variables in the enclosing scope:

### **Examples**

```cpp
int a = 5, b = 10;

// Capture by value
auto f1 = [a]() { return a + 1; };

// Capture by reference
auto f2 = [&b]() { b += 1; };

// Mixed capture
auto f3 = [=, &b]() { return a + (++b); };
```

### **Generalized Capture (C++14)**

Allows initializing new variables in the capture clause:

```cpp
auto ptr = std::make_unique<int>(42);
auto f = [val = std::move(ptr)]() {
    return *val;
};
```

**Use case:** Capture move-only types like `std::unique_ptr`.

## **5. Parameter List and Type Deduction**

Lambda parameters mirror function parameters:

```cpp
auto add = [](int x, int y) { return x + y; };
```

In C++14 and later, lambdas can use `auto` for generic lambdas:

```cpp
auto multiply = [](auto x, auto y) { return x * y; };
```

## **6. `mutable` Keyword**

By default, lambdas capturing by value produce a `const` call operator. `mutable` lifts this restriction:

```cpp
int x = 0;
auto f = [x]() mutable { x += 1; return x; };

f();  // returns 1
f();  // returns 2
x;    // still 0
```

Captured `x` is a copy, and mutable lets us modify that copy.

## **7. Exception Specification**

Lambda expressions can use `noexcept` or `throw()`:

```cpp
auto safe = []() noexcept {
    // does not throw
};
```

If a lambda marked `noexcept` throws, it results in `std::terminate()` at runtime.

## **8. Return Type**

C++ automatically deduces the return type from the body unless a trailing return type is specified:

```cpp
auto lambda = [](int x) -> double {
    return x + 0.5;
};
```

If multiple `return` paths have different types, or if returning braced-init lists, specify the return type explicitly.

## **9. Capturing `this`**

Lambdas used inside member functions often capture `this` for access to member variables or methods:

```cpp
class MyClass {
    int factor = 2;
public:
    void compute() {
        auto lambda = [this](int x) { return factor * x; };
        lambda(3); // returns 6
    }
};
```

From C++17, you can capture `*this` by value to copy the entire object into the closure:

```cpp
auto lambda = [*this]() { return factor; };
```

## **10. Lambdas in STL Algorithms**

```cpp
std::vector<int> vec = {1, 2, 3, 4, 5};
std::for_each(vec.begin(), vec.end(), [](int& n) {
    n *= 2;
});
```

**Output:**
`vec` becomes `{2, 4, 6, 8, 10}`

## **11. Stateful Closures and Lifetime Considerations**

When using lambdas asynchronously (e.g., in threads), avoid capturing local variables by reference unless you ensure their lifetime:

```cpp
int val = 42;
std::thread t([&]() { std::cout << val; }); // DANGER: val may be gone
```

Use by-value capture or synchronization mechanisms instead.

## **12. `constexpr` Lambdas (C++17)**

If the body of a lambda can be evaluated at compile-time, it may be marked or inferred as `constexpr`:

```cpp
constexpr auto add = [](int a, int b) { return a + b; };
static_assert(add(3, 4) == 7);
```

## **13. Higher-Order and Recursive Lambdas**

### **Returning a Lambda**

```cpp
auto make_adder = [](int x) {
    return [=](int y) { return x + y; };
};

auto add5 = make_adder(5);
add5(3); // returns 8
```

### **Recursive Lambda (C++14)**

```cpp
std::function<int(int)> fib = [&](int n) {
    return (n <= 1) ? n : fib(n - 1) + fib(n - 2);
};
```

## **14. Full Example: Fibonacci with `generate_n` and Captures**

```cpp
#include <vector>
#include <algorithm>
#include <iostream>

int main() {
    std::vector<int> v(9, 1);
    int x = 1, y = 1;

    std::generate_n(v.begin() + 2, 7, [=]() mutable {
        int n = x + y;
        x = y;
        y = n;
        return n;
    });

    for (int n : v) std::cout << n << " ";
}
```

**Output:**

```
1 1 2 3 5 8 13 21 34
```

## **Best Practices**

- Prefer **value captures** for thread safety; avoid dangling references.
- Use `mutable` sparingly; favor immutability for safer code.
- For complex lambdas, consider naming them or converting to a functor for readability.
- Leverage **generic lambdas** in template-heavy or functional-style code.
- Avoid `static` or shared state in lambdas unless thread-safe.

## **Conclusion**

Lambda expressions in C++ provide powerful functional programming capabilities while preserving type safety, locality, and expressiveness. Mastering lambdas allows you to write more declarative, concise, and maintainable C++—especially when combined with STL algorithms, concurrency primitives, and template metaprogramming techniques.
