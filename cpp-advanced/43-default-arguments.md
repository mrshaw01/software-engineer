# Default Arguments in C++

Understanding and leveraging default arguments is fundamental to writing clean, flexible, and maintainable interfaces. Default arguments in C++ allow a function to be called with fewer arguments than it is defined to accept, reducing overloads and enhancing clarity for common usage patterns.

### 1. **Definition and Use Case**

A _default argument_ is a value automatically assigned by the compiler when the function call omits that parameter. This is particularly useful for functions with optional parameters, where a reasonable default behavior can be defined.

**Example:**

```cpp
int print(double dvalue, int prec = 2);
```

Here, the second parameter `prec` is optional. If omitted, the compiler automatically uses `prec = 2`.

### 2. **Eliminating Redundant Overloads**

Without default arguments, you might resort to overloading functions to support variations in argument lists:

```cpp
int print(double dvalue);           // Without precision
int print(double dvalue, int prec); // With precision
```

This can be consolidated:

```cpp
int print(double dvalue, int prec = 2);
```

This approach reduces API surface complexity and centralizes the logic into a single implementation.

### 3. **Trailing Position Only**

**Rule**: Default arguments must be provided from right to left (i.e., only trailing parameters can have default values).

**Invalid:**

```cpp
int print(double dvalue = 0.0, int prec); // Error
```

**Valid:**

```cpp
int print(double dvalue, int prec = 2);
```

If you need multiple defaults, they must be in sequence at the end:

```cpp
void logMessage(std::string msg, int level = 1, bool verbose = false);
```

### 4. **Default Argument Declaration Rules**

#### A. **Declared Only Once**

A default argument can only be declared **once** across all declarations/definitions in a given scope.

**Invalid:**

```cpp
int print(double dvalue, int prec = 2); // Declaration
int print(double dvalue, int prec = 2) { ... } // Redefinition — ERROR
```

**Valid:**

```cpp
int print(double dvalue, int prec = 2); // Declaration
int print(double dvalue, int prec) { ... }     // Definition without default
```

#### B. **Default Arguments in Multiple Declarations**

You can **extend** the number of default parameters in later declarations, but not redefine the same one:

```cpp
void f(int a, int b = 5);         // Initial declaration
void f(int a = 1, int b = 5);     // Error: redefinition of `b`
```

### 5. **Default Arguments and Function Pointers**

Default arguments work with pointers to functions, but they are associated with the function declaration, not with the pointer itself:

```cpp
int printVal(int x = 0);
int (*funcPtr)(int) = printVal; // Default of 0 is valid when calling through pointer
```

Calling through `funcPtr(42)` works as expected, but `funcPtr()` will not apply the default — it must be called directly via the original function name to benefit from default arguments.

### 6. **Interaction with Overloading**

When overloading functions, the presence of default arguments can introduce ambiguity:

```cpp
void draw(int size);
void draw(int size, int color = 0); // Ambiguous call: draw(10);
```

Avoid such constructs. Instead, favor either default arguments **or** overloads—but not both unless overloads are explicitly disambiguated.

### 7. **Practical Example: Rounding Double Values**

```cpp
int print(double dvalue, int prec = 2) {
    static const double rgPow10[] = {
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0,
        1e1, 1e2, 1e3, 1e4, 1e5, 1e6
    };
    const int iPowZero = 6;
    if (prec >= -6 && prec <= 6) {
        dvalue = floor(dvalue / rgPow10[iPowZero - prec]) * rgPow10[iPowZero - prec];
    }
    std::cout << dvalue << std::endl;
    return std::cout.good();
}
```

**Usage:**

```cpp
print(123.456);      // Uses default precision (2)
print(123.456, 0);   // Overrides default
```

### 8. **Best Practices**

- Use default arguments sparingly and only when a logical default exists.
- Prefer default arguments over overloads when the function semantics remain identical.
- Avoid mixing default arguments and overloads that can cause ambiguity.
- Do not repeat default arguments in both declarations and definitions.
- Be cautious when using them in interfaces exposed across shared libraries or dynamically loaded modules—some compilers may evaluate defaults at the call site, not the definition site.

### 9. **Conclusion**

Default arguments provide a powerful mechanism for reducing code duplication, simplifying interfaces, and improving API usability. When applied with discipline and in alignment with modern C++ best practices, they can greatly improve the maintainability and clarity of your codebase. However, misusing them—particularly in conjunction with overloading—can lead to ambiguity and maintenance headaches. Always document the intended behavior of default parameters clearly in your API references.
