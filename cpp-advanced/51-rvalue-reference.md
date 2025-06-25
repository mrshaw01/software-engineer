# Rvalue Reference Declarator (`&&`) in C++

In modern C++ (C++11 and later), the **rvalue reference declarator** `&&` plays a critical role in enabling **move semantics** and **perfect forwarding**. Unlike lvalue references (`&`), which bind to named, addressable objects, **rvalue references** bind to **temporary objects** (rvalues) that are about to be destroyed and can thus safely transfer ownership of their resources.

## 1. **Syntax and Declaration**

```cpp
type-specifier-seq && identifier;
```

This declares an rvalue reference to the specified type:

```cpp
int&& x = 10; // x is an rvalue reference bound to the temporary rvalue 10
```

## 2. **Purpose of Rvalue References**

### a. **Move Semantics**

Move semantics allow for the transfer of resources from one object to another without the overhead of copying. This is particularly beneficial for classes that manage heap-allocated memory, file handles, or other non-trivial resources.

### b. **Perfect Forwarding**

Rvalue references enable a function template to forward arguments while preserving their value category (lvalue or rvalue). This avoids unnecessary copies and supports highly generic, efficient code.

## 3. **Move Semantics in Action**

### Example: Move Constructor and Move Assignment

```cpp
#include <iostream>
#include <vector>

class Buffer {
    std::vector<int> data;
public:
    Buffer(std::vector<int>&& vec) : data(std::move(vec)) {
        std::cout << "Move constructor\n";
    }

    Buffer& operator=(Buffer&& other) {
        if (this != &other) {
            data = std::move(other.data);
            std::cout << "Move assignment\n";
        }
        return *this;
    }
};
```

**Explanation**:
The class `Buffer` moves ownership of the `std::vector<int>` using `std::move`, avoiding costly deep copies.

## 4. **Function Overloading with Rvalue References**

```cpp
void f(const std::string& s) {
    std::cout << "lvalue overload\n";
}

void f(std::string&& s) {
    std::cout << "rvalue overload\n";
}
```

### Usage:

```cpp
std::string name = "example";
f(name);               // lvalue overload
f(std::string("tmp")); // rvalue overload
```

**Expected Output:**

```
lvalue overload
rvalue overload
```

**Explanation**:
The second call uses a temporary `std::string`, which binds to the rvalue reference overload.

## 5. **Named Rvalue References Are Lvalues**

Even though a function receives an rvalue reference parameter, once it's named, it becomes an lvalue.

```cpp
void g(const std::string&) { std::cout << "const lvalue\n"; }
void g(std::string&&) { std::cout << "rvalue\n"; }

std::string&& getTemp() { return std::string("temp"); }

void f(std::string&& s) {
    g(s);              // s is an lvalue, resolves to g(const std::string&)
    g(std::move(s));   // now g(std::string&&) is called
}
```

**Expected Output:**

```
const lvalue
rvalue
```

## 6. **Perfect Forwarding with `std::forward`**

```cpp
template <typename T>
void wrapper(T&& arg) {
    process(std::forward<T>(arg));
}
```

**Explanation**:

- If `arg` is an lvalue, it forwards as `T&`.
- If `arg` is an rvalue, it forwards as `T&&`.
  This allows a single overload to efficiently support both kinds of arguments.

## 7. **Reference Collapsing Rules**

When combining reference types during template deduction, the compiler applies **reference collapsing**:

| Combined Type | Collapsed Type |
| ------------- | -------------- |
| `T& &`        | `T&`           |
| `T& &&`       | `T&`           |
| `T&& &`       | `T&`           |
| `T&& &&`      | `T&&`          |

These rules ensure references remain valid and consistent in generic code.

## 8. **Factory Function with Perfect Forwarding**

```cpp
template <typename T, typename A1, typename A2>
T* factory(A1&& a1, A2&& a2) {
    return new T(std::forward<A1>(a1), std::forward<A2>(a2));
}
```

This single version of the `factory` function supports all permutations of lvalue/rvalue and const/non-const combinations, without needing a combinatorial explosion of overloads.

## 9. **std::move and Casting**

To **cast** an lvalue to an rvalue reference, use:

```cpp
std::string s = "hello";
std::string&& r = std::move(s); // safely transfers resources
```

You can also explicitly cast:

```cpp
g(static_cast<std::string&&>(s));
```

Use `std::move` to indicate that a named variable can be **moved from**, not that it **is** an rvalue.

## 10. **Template Deduction with Rvalue References**

Consider:

```cpp
template <typename T>
void inspect(T&& value) {
    S<T&&>::print(std::forward<T>(value));
}
```

Depending on how `inspect` is called, `T` could be:

- `T&` if passed an lvalue
- `T` if passed an rvalue

This allows deduction of value categories and correct overload resolution via reference collapsing.

## Summary

The `&&` declarator introduces **rvalue references**, which:

- Enable **move semantics**, reducing copies and heap allocations.
- Facilitate **perfect forwarding**, supporting generic, high-performance APIs.
- Are subject to **reference collapsing** during template deduction.
- Can distinguish between modifiable temporaries and immutable lvalues.
