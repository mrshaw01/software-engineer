# Standard Conversions in C++

## Overview

In C++, **standard conversions** refer to the implicit type transformations that occur between fundamental types, pointers, references, and pointer-to-member types. These conversions are intrinsic to the language and governed by well-defined rules. Understanding these conversions is critical to writing safe and correct code, particularly when working with mixed-type arithmetic, low-level systems programming, or overloading functions.

## Integral Promotions

Integral promotion is the process of converting an object of a smaller integral type to a larger integral type that can represent all its possible values. This occurs automatically in many arithmetic expressions.

Types subject to promotion:

- `char`, `signed char`, `unsigned char`
- `short`, `unsigned short`
- Enumeration types
- Bit-fields of type `int` or narrower

### Promotion Rules

1. If `int` can represent all values of the original type, the value is promoted to `int`.
2. Otherwise, it is promoted to `unsigned int`.

These promotions are **value-preserving**: the promoted result has the same numeric value. However, **signedness may be lost**, which can change behavior in signed-sensitive operations.

### Subtle Cases

Differences between value-preserving and sign-preserving promotions can appear when the promoted values are:

- Operands of comparison (`<`, `>=`, etc.)
- Operands of division or modulus (`/`, `%`)
- Operands of right shift (`>>`)
- Passed to overloaded functions or operators

```cpp
long long_num1, long_num2;
int int_num;

// int_num is promoted to long before assignment
long_num1 = int_num;

// int_num is promoted to long before multiplication
long_num2 = int_num * long_num2;
```

## Integral Conversions

These conversions occur when values are assigned or passed between different-sized integral types (with or without signedness).

### Signed to Unsigned

The bit pattern remains unchanged, but its interpretation changes:

```cpp
short i = -3;
unsigned short u = i;  // u now holds 65533 (2^16 - 3)
std::cout << u << "\n";  // Output: 65533
```

### Unsigned to Signed

If the unsigned value exceeds the range of the target signed type, the result is implementation-defined:

```cpp
unsigned short u = 65533;
short i = u;
std::cout << i << "\n";  // Output: -3 (interpretation of raw bits)
```

## Floating Point Conversions

Floating-point types (`float`, `double`, `long double`) can be converted between each other. These conversions may or may not preserve precision depending on direction.

### Widening Conversions (Safe)

From less precise to more precise type:

```cpp
float f = 3.14f;
double d = f;  // Safe: no loss of information
```

### Narrowing Conversions (Potentially Unsafe)

From more precise to less precise type:

```cpp
std::cout << (float)1E300 << "\n";  // Output: inf
```

In this case, the original value (1E300) is beyond the `float` range and results in **infinity**.

## Conversions Between Integral and Floating Point Types

### Integral to Floating Point

May involve rounding if the value is not exactly representable:

```cpp
int i = 16777217;  // float can represent only up to 16777216 exactly
float f = i;       // f holds a rounded value
```

### Floating Point to Integral

Fractional parts are **truncated** (rounded toward zero). If the value is out of the target integral type’s range, the result is undefined.

```cpp
float f = -1.9;
int i = f;  // i == -1
```

Here's **Part 2 of Standard Conversions in C++** written in professional Markdown style for technical documentation, continuing from Part 1:

## Arithmetic Conversions

Arithmetic conversions—often referred to as the **usual arithmetic conversions**—occur in binary expressions involving operands of differing types. These conversions are applied implicitly to bring both operands to a common type suitable for the operation. This standardization ensures consistent and predictable behavior in arithmetic operations.

### Conversion Rules

Operands are converted based on the following hierarchy:

| Condition                                          | Conversion                                       |
| -------------------------------------------------- | ------------------------------------------------ |
| Either operand is `long double`                    | Convert the other to `long double`               |
| Else, either operand is `double`                   | Convert the other to `double`                    |
| Else, either operand is `float`                    | Convert the other to `float`                     |
| Else (both operands are integral types)            | Apply integral promotions and follow this order: |
| • Either operand is `unsigned long`                | Convert the other to `unsigned long`             |
| • Else, one is `long` and the other `unsigned int` | Convert both to `unsigned long`                  |
| • Else, either is `long`                           | Convert the other to `long`                      |
| • Else, either is `unsigned int`                   | Convert the other to `unsigned int`              |
| • Else                                             | Convert both to `int`                            |

These rules preserve precision and value consistency as much as possible.

### Example

```cpp
double dVal;
float fVal;
int iVal;
unsigned long ulVal;

int main() {
    dVal = iVal * ulVal;   // iVal → unsigned long → result → double
    dVal = ulVal + fVal;   // ulVal → float → result → double
}
```

## Pointer Conversions

### Pointer to Class

A pointer to a derived class can be converted to a pointer to a base class if:

- The base class is accessible from the derived class
- The conversion is unambiguous

Access depends on inheritance type (`public`, `protected`, `private`). For example:

```cpp
class A { public: int AComponent; };
class B : public A { public: int BComponent; };

B bObj;
A* pA = &bObj; // OK: B → A
pA->AComponent; // Valid
// pA->BComponent; // Error: not accessible via A*
```

### Explicit Base Pointer Conversion

An explicit cast can be used when implicit conversion fails due to access restrictions or ambiguity.

### Pointer to Function

A pointer to a function can be cast to `void*` if `void*` is large enough. However, this is implementation-defined and generally discouraged.

### Pointer to Void

C++ allows implicit conversion:

- From any object pointer to `void*`
- From `void*` to any object pointer via **explicit cast only**

```cpp
int* pi = nullptr;
void* pv = pi;            // Implicit
int* pi2 = (int*)pv;      // Explicit
```

Note: No implicit conversion between `void*` and function pointers.

### Const and Volatile Pointers

C++ disallows implicit conversion **from** `const` or `volatile` **to** non-`const` or non-`volatile`. You must use an explicit cast.

```cpp
const int* pci;
int* pi = const_cast<int*>(pci); // Unsafe
```

## Null Pointer Conversions

An integer constant `0`, or an expression cast to a pointer type from zero, is treated as a **null pointer**. In C++11 and later, prefer `nullptr`:

```cpp
int* p = 0;          // Legacy null pointer
int* q = nullptr;    // Preferred
```

## Pointer Expression Conversions

### Array-to-Pointer Decay

Arrays decay to pointers to their first element:

```cpp
char path[260];
char* p = path; // Equivalent to &path[0]
```

### Function-to-Pointer Decay

A function name in most contexts decays into a pointer to that function, except:

- When the function is used with `&`
- When the function is directly called

```cpp
void f();
void (*pf)() = f;  // OK
```

## Reference Conversions

Similar to pointer-to-base conversions, a reference to a derived class can be converted to a reference to a base class if:

- The base is accessible
- The conversion is unambiguous

```cpp
B b;
A& aRef = b; // A is base of B
```

## Pointer-to-Member Conversions

Pointers to members (non-static) follow different rules from regular pointers.

### Base-to-Derived Member Conversion

A pointer to a base class member can be converted to a pointer to the derived class member if:

- The inverse conversion is accessible
- No virtual inheritance occurs

```cpp
class A { public: int i; };
class B : public A { };

int A::*pai = &A::i;
int B::*pbi = pai; // OK
```

### Null Member Pointers

Use `0` or `nullptr` to define a null pointer to member:

```cpp
class A { public: int i; };
int A::* pai = nullptr;
```
