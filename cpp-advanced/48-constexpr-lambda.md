# `constexpr` Lambda Expressions in C++

`constexpr` lambda expressions are a modern C++ feature that enable lambdas to be evaluated at compile time. This is particularly useful in performance-critical applications, template metaprogramming, and contexts where constant expressions are required (such as array bounds, enum values, or `constexpr` functions).

This feature was introduced in **C++17** and refined in **C++20**, enabling both **explicit** and **implicit** `constexpr` lambdas. The use of `constexpr` makes lambdas not just anonymous and inline, but also **compile-time evaluable**, when all conditions are met.

## **1. Syntax and Semantics of `constexpr` Lambdas**

### **Explicit Declaration**

```cpp
auto lambda = [](int x) constexpr {
    return x * 2;
};
```

This lambda is explicitly marked as `constexpr`, and will be usable in compile-time expressions **if**:

- Its body contains only constant expressions.
- Captures (if any) are allowed in constant expressions (i.e., capture by value with constant-valued variables).

### **Implicit `constexpr` (C++17 and later)**

If a lambda meets the requirements of a `constexpr` function, it is implicitly treated as one.

```cpp
auto lambda = [](int x) {
    return x + 1;
};

constexpr int result = lambda(3);  // Valid
```

This behavior makes code more concise and expressive without sacrificing compile-time guarantees.

## **2. Example: Explicit `constexpr` Lambda**

```cpp
int y = 32;
auto answer = [y]() constexpr {
    int x = 10;
    return y + x;
};
```

**Note:** This code is **not valid** as a `constexpr` because `y` is captured from the enclosing (non-`constexpr`) context. For a lambda to be truly `constexpr`, captured variables must themselves be `constexpr`.

### **Correct Version**

```cpp
constexpr int y = 32;
auto answer = [y]() constexpr {
    int x = 10;
    return y + x;
};

constexpr int result = answer();  // OK
```

**Output:**
`result == 42`

**Explanation:**
All variables used are `constexpr`, so the lambda is eligible for compile-time execution.

## **3. `constexpr` Lambdas in Functions**

```cpp
constexpr int Increment(int n) {
    return [n] { return n + 1; }();
}
```

### **Usage:**

```cpp
constexpr int val = Increment(9);
static_assert(val == 10);
```

**Explanation:**
The lambda within `Increment` captures `n` by value and adds one. Since `n` is a parameter of a `constexpr` function, the capture is valid at compile time.

## **4. Implicit `constexpr` and Function Pointers**

A `constexpr` lambda can be converted to a function pointer that is also `constexpr`.

```cpp
auto Increment = [](int n) {
    return n + 1;
};

constexpr int (*inc)(int) = Increment;
constexpr int result = inc(5);
```

**Output:**
`result == 6`

**Explanation:**
The lambda is implicitly `constexpr`, and conversion to a function pointer retains this property.

## **5. Practical Applications of `constexpr` Lambdas**

### **Use Case 1: Compile-Time Initialization**

```cpp
constexpr auto square = [](int x) { return x * x; };
constexpr int squares[] = { square(1), square(2), square(3) };
```

### **Use Case 2: Template Metaprogramming**

```cpp
template <typename T, int N>
constexpr auto generate_array(T (*f)(int)) {
    std::array<T, N> arr = {};
    for (int i = 0; i < N; ++i) {
        arr[i] = f(i);
    }
    return arr;
}

constexpr auto powers_of_two = generate_array<int, 5>([](int x) constexpr {
    return 1 << x;
});
```

### **Use Case 3: Enums and Constant Tables**

```cpp
enum class MyEnum {
    Value0 = [](int x) constexpr { return x * 2; }(0),
    Value1 = [](int x) constexpr { return x * 2; }(1),
};
```

## **6. Restrictions and Considerations**

- Lambdas that capture by reference **cannot** be `constexpr`.
- All statements inside the lambda must be valid in a constant expression context.
- Capturing variables must themselves be `constexpr` to participate in a `constexpr` lambda.

### **Invalid Example (Non-`constexpr` capture):**

```cpp
int x = 42;
auto f = [x]() constexpr { return x; }; // Error: x is not constexpr
```

### **Correct Version:**

```cpp
constexpr int x = 42;
auto f = [x]() constexpr { return x; }; // OK
constexpr int result = f();             // OK
```

## **7. C++20 Enhancements**

C++20 further strengthens `constexpr` support:

- Lambdas with `template` parameters (generic lambdas) can be `constexpr`.
- Use of `constexpr` lambdas in `consteval` and `constinit` contexts is now standard.

```cpp
constexpr auto add = [](auto a, auto b) {
    return a + b;
};

static_assert(add(3, 4) == 7);
```

## **Conclusion**

`constexpr` lambda expressions elevate lambdas from runtime tools to compile-time assets. When used appropriately, they enable safer, faster, and more expressive code that takes advantage of compile-time evaluation. This capability is especially powerful in systems programming, embedded software, and template-heavy libraries that benefit from early evaluation and zero-runtime overhead.

To ensure correct usage:

- Capture only `constexpr` variables.
- Avoid reference captures.
- Use `constexpr` lambdas in contexts where performance or determinism matter.
