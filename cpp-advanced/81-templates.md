# Templates

## **1. Overview of Templates**

Templates enable **generic programming** in C++. Unlike in C or Java where developers might rely on void pointers or base classes with casts, templates allow **compile-time generation of type-safe code** that operates on arbitrary types. This avoids duplication and promotes reusability while preserving type safety and performance.

### **Key Ideas**

- Templates defer type specification to instantiation time.
- Templates are expanded by the compiler, creating **type-specific code** for each instantiation, resulting in **zero runtime overhead** compared to hand-written type-specific functions.

## **2. Function Templates**

### **Basic Example**

```cpp
#include <iostream>

template <typename T>
T minimum(const T& lhs, const T& rhs) {
    return lhs < rhs ? lhs : rhs;
}

int main() {
    int a = 5, b = 10;
    double x = 4.5, y = 2.3;

    std::cout << minimum(a, b) << std::endl; // Output: 5
    std::cout << minimum(x, y) << std::endl; // Output: 2.3
}
```

#### **Explanation**

- `T` is a **template type parameter**.
- The compiler generates a specific `minimum<int>` and `minimum<double>` function when called.
- Type deduction avoids explicit specification in most cases, improving readability.

## **3. Class Templates**

### **Basic Example**

```cpp
#include <iostream>

template <typename T>
class Box {
private:
    T value;

public:
    Box(T v) : value(v) {}

    T get() const { return value; }
};

int main() {
    Box<int> intBox(42);
    Box<std::string> strBox("Hello");

    std::cout << intBox.get() << std::endl;  // Output: 42
    std::cout << strBox.get() << std::endl;  // Output: Hello
}
```

#### **Explanation**

- Defines a generic `Box` that can hold any type `T`.
- The compiler generates different versions (e.g. `Box<int>`, `Box<std::string>`).

## **4. Multiple Type Parameters**

```cpp
template <typename T, typename U>
class Pair {
public:
    T first;
    U second;

    Pair(T f, U s) : first(f), second(s) {}
};

int main() {
    Pair<int, std::string> p(1, "one");
    std::cout << p.first << ", " << p.second << std::endl; // Output: 1, one
}
```

## **5. Variadic Templates**

Introduced in C++11, variadic templates allow **arbitrary numbers of type parameters**, enabling flexible and reusable designs like tuple implementations.

### **Example**

```cpp
#include <iostream>

template <typename... Args>
void print_all(Args... args) {
    (std::cout << ... << args) << std::endl; // Fold expression (C++17)
}

int main() {
    print_all(1, " + ", 2, " = ", 3); // Output: 1 + 2 = 3
}
```

## **6. Non-Type Template Parameters**

Templates can also take **constant values** as parameters.

### **Example: Fixed-size Array**

```cpp
template <typename T, size_t N>
class FixedArray {
private:
    T data[N];

public:
    size_t size() const { return N; }

    T& operator[](size_t i) { return data[i]; }
};

int main() {
    FixedArray<int, 5> arr;
    for (size_t i = 0; i < arr.size(); ++i)
        arr[i] = i * i;

    for (size_t i = 0; i < arr.size(); ++i)
        std::cout << arr[i] << " "; // Output: 0 1 4 9 16
}
```

## **7. Template Specialization**

Allows defining **type-specific behavior** for a particular type.

### **Example: Full Specialization**

```cpp
template <typename T>
class Printer {
public:
    void print(T value) {
        std::cout << "General: " << value << std::endl;
    }
};

// Full specialization for const char*
template <>
class Printer<const char*> {
public:
    void print(const char* value) {
        std::cout << "C-string: " << value << std::endl;
    }
};

int main() {
    Printer<int> pi;
    pi.print(42); // Output: General: 42

    Printer<const char*> ps;
    ps.print("Hello"); // Output: C-string: Hello
}
```

## **8. Partial Specialization (Class Templates Only)**

```cpp
template <typename T, typename U>
class Pair {
    // general implementation
};

template <typename U>
class Pair<int, U> {
    // partial specialization when first type is int
};
```

> **Note**: Function templates **cannot** be partially specialized. Instead, use function overloading or tag dispatching.

## **9. Template Template Parameters**

Templates can accept other templates as parameters.

```cpp
template <typename T, template <typename> class Container>
class Holder {
    Container<T> data;
};

#include <vector>

int main() {
    Holder<int, std::vector> h; // uses std::vector<int> internally
}
```

## **10. Default Template Arguments**

Improve usability by providing defaults.

```cpp
template <typename T = int>
class DefaultBox {
public:
    T value;
};

DefaultBox<> box; // uses int as default type
```

## **11. Best Practices**

1. **Prefer `typename` over `class`**: Both are equivalent, but `typename` emphasizes generic types.
2. **Minimize compilation overhead**: Templates can cause code bloat; balance reuse and compile time.
3. **Use `constexpr` and `inline` appropriately** in template definitions to avoid multiple definitions across translation units.
4. **Combine with SFINAE or concepts** (C++20) for cleaner constraints and improved error messages.
5. **Avoid template specialization misuse**: Specialize only when absolutely necessary to avoid maintenance complexity.

## **12. Modern C++: Concepts and Constraints (C++20)**

Enhance templates with **concepts** to constrain type parameters explicitly.

```cpp
#include <concepts>

template <std::integral T>
T add_one(T x) {
    return x + 1;
}

int main() {
    std::cout << add_one(5) << std::endl; // OK
    // std::cout << add_one(5.5) << std::endl; // Compilation error: not integral
}
```

## **Conclusion**

Templates are fundamental to **modern C++ generic programming**, enabling libraries such as STL, Boost, and Eigen to deliver **type-safe, efficient, and flexible solutions**. Mastering templates, including specialization and variadic forms, is crucial for designing reusable and high-performance software systems.
