# Function Templates: Professional Overview

### Purpose

Function templates enable **generic programming**, allowing a single function definition to operate on different data types without code duplication. They are resolved at compile-time, yielding type safety and performance comparable to manually specialized functions.

### Syntax and Basic Example

**Generic Swap Function**

```cpp
template<class T>
void MySwap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

int main() {
    int x = 1, y = 2;
    MySwap(x, y); // Swaps two ints

    double a = 3.14, b = 2.71;
    MySwap(a, b); // Swaps two doubles
}
```

#### Output

```
x = 2, y = 1
a = 2.71, b = 3.14
```

**Explanation:**

- `template<class T>` defines a template parameter `T` used within the function signature.
- The compiler generates specific instantiations for each type used (`int`, `double` here) at compile-time.
- Type safety is preserved: attempting to swap different types (e.g. `int` and `std::string`) triggers a compile-time error.

### **Template Argument Deduction and Explicit Specification**

By default, the compiler deduces the template type from arguments:

```cpp
int i = 10;
double d = 20.5;
// MySwap(i, d); // Compile error: deduced types are not identical
```

However, **explicit specification** forces the type:

```cpp
template<class T>
void f(T) {}

int main() {
    int x = 42;
    f<char>(x); // x is converted to char before calling f<char>
}
```

#### Best Practice:

- Rely on type deduction for clarity.
- Use explicit specification only when necessary (e.g., forcing a wider or narrower type).

### **Function Template Instantiation**

When a function template is used with a type for the first time, the compiler **instantiates** it by generating a concrete version.

```cpp
template<class T>
void print_type(T) {}

template void print_type<int>(int); // Explicit instantiation
template void print_type(double);   // Implicit instantiation by usage

int main() {}
```

- Explicit instantiation (`template void print_type<int>(int);`) forces code generation in the translation unit, useful in library development to reduce duplication across modules.

### **Explicit Specialization**

You can define custom behavior for specific types.

```cpp
template<class T>
void print(T) {
    std::cout << "Generic print\n";
}

// Specialization for double
template<>
void print<double>(double) {
    std::cout << "Double specialization\n";
}

int main() {
    print(1);     // Generic print
    print(3.14);  // Double specialization
}
```

#### Output

```
Generic print
Double specialization
```

### **Partial Ordering and Overload Resolution**

When multiple function templates match, the **more specialized** is chosen.

```cpp
template<class T>
void f(T) {
    std::cout << "Generic\n";
}

template<class T>
void f(T*) {
    std::cout << "Pointer specialization\n";
}

template<class T>
void f(const T*) {
    std::cout << "Const pointer specialization\n";
}

int main() {
    int i = 0;
    int* pi = &i;
    const int* cpi = &i;

    f(i);   // Generic
    f(pi);  // Pointer specialization
    f(cpi); // Const pointer specialization
}
```

#### Output

```
Generic
Pointer specialization
Const pointer specialization
```

### **Member Function Templates**

#### Inside Non-template Class

```cpp
struct Printer {
    template<class T>
    void print(const T& val) {
        std::cout << val << std::endl;
    }
};

int main() {
    Printer p;
    p.print(42);        // Prints integer
    p.print("hello");   // Prints string
}
```

#### Inside Class Template

```cpp
template<class T>
class Wrapper {
public:
    template<class U>
    void show(const U& val) {
        std::cout << "Value: " << val << std::endl;
    }
};

int main() {
    Wrapper<int> w;
    w.show(3.14);
}
```

#### Definition Outside Class

```cpp
template<class T>
class Wrapper {
public:
    template<class U>
    void show(const U& val);
};

template<class T>
template<class U>
void Wrapper<T>::show(const U& val) {
    std::cout << val << std::endl;
}
```

### **Templated User-defined Conversions**

Templates can define generic conversion operators.

```cpp
template<class T>
struct S {
    template<class U>
    operator S<U>() const {
        return S<U>();
    }
};

int main() {
    S<int> s1;
    S<double> s2 = s1; // Converts via template conversion operator
}
```

### **Best Practices**

1. **Prefer function templates** over `void*` for type safety.
2. **Avoid unnecessary explicit specializations** unless behavior changes meaningfully.
3. **Use explicit instantiation** in library development to reduce code bloat and compilation times.
4. **Avoid function templates with excessive overloads** when type traits (`std::enable_if`, concepts) provide clearer constraints in C++11/14/17/20.
5. In modern C++20, **concepts** improve readability and constraint enforcement for function templates.

### **Summary**

Function templates are foundational for **generic and reusable code design**. They enable type-safe operations across diverse types without runtime overhead. Modern usage often combines function templates with **concepts** to express constraints declaratively, improving code clarity and maintainability in large-scale systems.
