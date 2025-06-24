# `const` in C++

In C++, the `const` keyword is a powerful feature used to express **intent**, enforce **read-only semantics**, and enable **compiler-assisted correctness**. It can be applied to variables, pointers, function parameters, return types, and member functions.

## 1. Constant Variables

Declaring a variable as `const` means its value cannot be modified after initialization.

```cpp
const int max_users = 100;
// max_users = 200; // Error: cannot modify a const variable
```

Unlike `#define`, `const` values are type-safe, scoped, and can participate in constant expressions:

```cpp
const int max_size = 64;
char buffer[max_size]; // Valid in C++
```

In C++, const variables have **internal linkage** by default. This makes them suitable for header files without violating the One Definition Rule (ODR).

## 2. Const Pointers and Pointer to Const

Pointer declarations with `const` can be interpreted differently depending on the placement:

```cpp
int x = 10, y = 20;
int* const p1 = &x;     // constant pointer to int (can't point elsewhere)
const int* p2 = &x;     // pointer to constant int (can't modify pointed value)
int const* p3 = &x;     // same as above
const int* const p4 = &x; // constant pointer to constant int
```

## 3. Const Member Functions

A member function declared with `const` guarantees it won’t modify the object’s state:

```cpp
class User {
public:
    int get_id() const;  // OK on const object
    void set_id(int);    // Not callable on const object

private:
    int id;
};

const User u;
u.get_id();   // OK
u.set_id(10); // Error
```

Const member functions:

- Cannot modify member variables.
- Can only call other `const` member functions.

You **must** mark the method as `const` in both declaration and definition.

## 4. Const Function Parameters

Declaring function parameters as `const` improves readability and enforces immutability:

```cpp
void print_name(const std::string& name);
```

Passing by `const&` avoids unnecessary copies and protects inputs from modification.

## 5. Const and Overloading

C++ allows method overloading based on constness:

```cpp
class Buffer {
public:
    char& operator[](size_t index)       { return data[index]; }
    const char& operator[](size_t index) const { return data[index]; }

private:
    char data[128];
};
```

This enables read-only access for `const` instances and modifiable access otherwise.

## 6. Const Global Variables and `extern`

C++ treats `const` globals as `internal` by default. To expose them across translation units:

```cpp
// header.h
extern const int threshold;

// source.cpp
const int threshold = 10;
```

For interoperability with C:

```cpp
extern "C" const int version = 3;
```

## Summary

- Use `const` to express intent and enforce invariants.
- Prefer `const&` for function parameters to avoid copies.
- Declare member functions `const` when they don’t modify state.
- Take advantage of `const`-based overloading to support read-only semantics.

Correct use of `const` improves code **safety**, **maintainability**, and enables compiler **optimizations** by clarifying mutability.
