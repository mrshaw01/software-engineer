# Trivial, Standard-Layout, POD, and Literal Types in C++

In C++, understanding the memory layout and type characteristics of objects is crucial for performance, interoperability, and low-level programming. The terms **trivial**, **standard-layout**, **POD**, and **literal types** define specific constraints on class and struct types that impact how they are constructed, laid out, and used in contexts like serialization, ABI compatibility, or metaprogramming.

## Memory Layout and Type Categories

The term _layout_ refers to how members of a `class`, `struct`, or `union` are arranged in memory. For most modern C++ applications, the compiler determines this layout based on a set of rules and optimizations. However, when dealing with low-level APIs, memory-mapped hardware, or C interoperability, explicit control and understanding of layout becomes necessary.

To reason about the suitability of types for these operations, C++14 introduced traits and terminology for classifying types:

- `std::is_trivial<T>`
- `std::is_standard_layout<T>`
- `std::is_pod<T>` _(deprecated in C++20)_

## Trivial Types

A **trivial** type has special member functions that are compiler-provided or explicitly `= default`. Trivial types are:

- Contiguously laid out in memory
- Safe to copy using `memcpy`
- Not necessarily compatible with C

### Requirements

A type is trivial if:

- It has trivial default/copy/move constructors, assignment operators, and destructors
- It has no virtual functions or virtual base classes
- All its base classes and non-static data members are also trivial

```cpp
struct Trivial {
    int i;
private:
    int j;
};

struct Trivial2 {
    int i;
    Trivial2(int a, int b) : i(a), j(b) {}
    Trivial2() = default;
private:
    int j;
};
```

## Standard-Layout Types

A **standard-layout** type has a layout compatible with C structs and is safely consumable by C programs. It may have user-defined special member functions, but must avoid certain features.

### Requirements

A standard-layout type must:

- Not have virtual functions or virtual base classes
- Have all non-static data members with the same access control (e.g., all `public`)
- Have all base classes and data members of standard-layout type
- Not have a base class of the same type as the first non-static data member
- Satisfy either:

  - No non-static members in the derived class and only one base with members, or
  - No base classes with non-static data members

```cpp
struct SL {
    int i;
    int j;
    SL(int a, int b) : i(a), j(b) {}
};
```

### Example: Violating Standard Layout

```cpp
struct Base {
    int i;
    int j;
};

struct Derived : public Base {
    int x;
    int y;
};
// Not standard-layout because both Derived and Base have non-static data members
```

## POD Types (Plain Old Data)

A **POD type** is both **trivial** and **standard-layout**. These types behave like C structs:

- Their layout is fully predictable
- Can be safely copied, serialized, and used in `extern "C"` functions
- All members are PODs

### Example

```cpp
struct POD {
    int a;
    int b;
};
```

## Type Trait Demonstration

```cpp
#include <type_traits>
#include <iostream>
using namespace std;

struct B {
protected:
    virtual void Foo() {}
};

struct A : B {
    int a;
    int b;
    void Foo() override {}
};

struct C {
    int a;
private:
    int b;
};

struct D {
    int a;
    int b;
    D() {} // user-defined constructor
};

struct POD {
    int a;
    int b;
};

int main() {
    cout << boolalpha;
    cout << "A is trivial: " << is_trivial<A>() << endl;
    cout << "A is standard-layout: " << is_standard_layout<A>() << endl;

    cout << "C is trivial: " << is_trivial<C>() << endl;
    cout << "C is standard-layout: " << is_standard_layout<C>() << endl;

    cout << "D is trivial: " << is_trivial<D>() << endl;
    cout << "D is standard-layout: " << is_standard_layout<D>() << endl;

    cout << "POD is trivial: " << is_trivial<POD>() << endl;
    cout << "POD is standard-layout: " << is_standard_layout<POD>() << endl;
}
```

## Literal Types

A **literal type** is one whose value and structure can be computed at compile-time. Literal types enable the use of `constexpr` variables and functions.

### Requirements (C++14 and C++20)

A class is a literal type if:

- It has a `constexpr` constructor (not copy or move)
- All its base classes and non-static data members are literal types
- It has a trivial destructor
- It does not contain `volatile` members

### Examples of Literal Types:

- `int`, `char`, `double`
- `const int&`
- Arrays of literal types
- `constexpr` classes without virtual members or user-defined destructors

## Summary Table

| Type Category                         | Trivial | Standard-Layout | POD |       Literal        |
| ------------------------------------- | :-----: | :-------------: | :-: | :------------------: |
| `int`, `float`                        |   ✅    |       ✅        | ✅  |          ✅          |
| Class with only `int` members         |   ✅    |       ✅        | ✅  | ✅ if constexpr ctor |
| Class with virtual func               |   ❌    |       ❌        | ❌  |          ❌          |
| Class with private and public members |   ✅    |       ❌        | ❌  |          ❌          |
| Class with user-defined constructor   |   ❌    |       ✅        | ❌  |        maybe         |

## Conclusion

Understanding **trivial**, **standard-layout**, **POD**, and **literal** types helps C++ developers write safer, faster, and more interoperable code. These type traits enable optimizations like `memcpy`, allow compatibility with C, and make types usable in `constexpr` contexts and metaprogramming logic.

Use the `<type_traits>` library to verify these properties and design your data structures accordingly for your specific constraints, whether that’s ABI compatibility, embedded systems, or compile-time computation.
