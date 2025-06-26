# Delegating Constructors

**Delegating constructors**, introduced in C++11, enable one constructor to invoke another constructor of the same class. This allows for a **centralization of common initialization logic**, reducing code duplication and enhancing maintainability, especially in classes with multiple constructors.

## 1. **Motivation for Delegating Constructors**

Without delegation, constructors often repeat similar logic:

```cpp
class Config {
public:
    int timeout;
    int retry;

    Config() {
        timeout = 30;
        retry = 3;
    }

    Config(int t) {
        timeout = (t > 0) ? t : 30;
        retry = 3;
    }

    Config(int t, int r) {
        timeout = (t > 0) ? t : 30;
        retry = (r > 0) ? r : 3;
    }
};
```

This is repetitive and error-prone. Delegation solves this:

```cpp
class Config {
public:
    int timeout;
    int retry;

    Config() : Config(30, 3) {}

    Config(int t) : Config(t, 3) {}

    Config(int t, int r) {
        timeout = (t > 0) ? t : 30;
        retry = (r > 0) ? r : 3;
    }
};
```

## 2. **Syntax and Semantics**

Delegation uses the **constructor initializer list** to invoke another constructor:

```cpp
class_name(parameter_list) : class_name(arguments) {
    // additional setup (optional)
}
```

**Important rule**: If a constructor delegates to another, it **cannot initialize members itself** in the initializer list. Initialization must be done **inside the constructor body**.

## 3. **Example: Parameter Validation with Constructor Chaining**

```cpp
class class_c {
public:
    int max;
    int min;
    int middle;

    class_c(int my_max) {
        max = my_max > 0 ? my_max : 10;
    }

    class_c(int my_max, int my_min) : class_c(my_max) {
        min = my_min > 0 && my_min < max ? my_min : 1;
    }

    class_c(int my_max, int my_min, int my_middle) : class_c(my_max, my_min) {
        middle = my_middle < max && my_middle > min ? my_middle : 5;
    }
};

int main() {
    class_c c{ 100, 20, 50 };  // All validations performed hierarchically
}
```

### Execution Flow:

- `class_c(int, int, int)` calls `class_c(int, int)`
- `class_c(int, int)` calls `class_c(int)`
- Each constructor performs **only the new work** relevant to its scope

## 4. **Limitations of Delegating Constructors**

### a. **Cannot Mix Delegation and Member Initialization**

```cpp
class class_a {
public:
    class_a(std::string str, double d)
        : class_a(str), m_double(d) {} // ❌ Error: can't initialize m_double here
};
```

### Correct Form:

```cpp
class class_a {
public:
    class_a(std::string str, double d)
        : class_a(str) {
        m_double = d;  // Assign in body
    }

private:
    double m_double{1.0};
    std::string m_string;
};
```

### b. **No Compile-Time Recursion Detection**

```cpp
class class_f {
public:
    class_f() : class_f(5, 2) {}        // calls constructor with 2 args
    class_f(int a, int b) : class_f() {} // calls default constructor
};
```

This results in **infinite recursion** at runtime → stack overflow. The compiler **does not** diagnose this cycle.

## 5. **Interaction with Non-Static Data Member Initializers**

C++11 allows initializing members directly at their point of declaration:

```cpp
class class_a {
public:
    class_a() {}
    class_a(std::string str) : m_string{str} {}
    class_a(std::string str, double d) : class_a(str) {
        m_double = d;
    }

    double m_double{1.0};
    std::string m_string{m_double < 10.0 ? "alpha" : "beta"};
};
```

### Behavior:

- If a constructor **initializes a member explicitly**, it overrides the default member initializer.
- If not, the member initializer applies.

### Example:

```cpp
class_a a("hello", 2.0);
// a.m_string == "hello"
// a.m_double == 2.0
```

## 6. **Best Practices for Delegating Constructors**

| Best Practice                                                             | Rationale                                           |
| ------------------------------------------------------------------------- | --------------------------------------------------- |
| Delegate common initialization logic to a core constructor                | Prevents code duplication                           |
| Avoid mixing delegation and member initializer list                       | Violates C++ rules                                  |
| Use non-static member initializers for defaults                           | Simplifies constructor logic                        |
| Guard against infinite delegation cycles                                  | No compile-time check exists                        |
| Favor delegation over private helper functions if it improves readability | Delegation is semantically clearer for constructors |

## 7. **Comparison with Factory Functions**

Delegating constructors improve clarity for constructor logic, but **do not replace factory methods** when:

- Polymorphic construction is required
- Initialization might fail (constructors cannot return error codes)
- You want to cache or pool objects

## 8. **Summary**

Delegating constructors streamline initialization by allowing constructors to call each other directly. When used correctly, they promote clean, DRY code and eliminate repetitive validation and assignment logic. However, developers must handle their constraints carefully, especially around initializer list restrictions and avoiding recursive calls.

This feature is a foundational building block for writing robust, modern C++ classes, particularly in value-oriented and resource-managing types.
