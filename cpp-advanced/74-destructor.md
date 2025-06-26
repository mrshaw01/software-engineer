### Destructors in C++

A destructor is a special member function that performs cleanup activities when an object’s lifetime ends. Destructors play a critical role in _RAII_ (Resource Acquisition Is Initialization), ensuring exception safety and preventing resource leaks in modern C++ systems.

## 1. **Definition and Syntax**

A **destructor** is declared with the same name as the class, prefixed by a tilde `~`. It takes no parameters and returns nothing.

```cpp
class MyClass {
public:
    ~MyClass(); // Destructor
};
```

If you don’t define one, the compiler automatically generates a default destructor, which performs _memberwise destruction_.

## 2. **When to Define a Destructor**

You should define a custom destructor when your class:

- Manages a resource (e.g., dynamic memory, file handles, sockets).
- Needs to perform custom cleanup (e.g., flushing logs, unregistering events).
- Interacts with legacy C APIs or manual allocation.

Example:

```cpp
class String {
public:
    String(const char* str);
    ~String(); // releases dynamic memory
private:
    char* _text;
};

String::String(const char* str) {
    size_t len = strlen(str) + 1;
    _text = new char[len];
    strcpy_s(_text, len, str);
}

String::~String() {
    delete[] _text; // ensures no memory leak
}
```

## 3. **Destructor Properties**

- **Cannot be overloaded** — only one destructor per class.
- **Cannot take parameters or return values**.
- **Cannot be `const`, `volatile`, or `static`**.
- **Can be virtual**, which is critical for polymorphic deletion.
- **Can be pure virtual** for abstract base classes, but it must still have a body.

## 4. **Virtual Destructors**

Use a `virtual` destructor when your class is intended to be inherited polymorphically.

```cpp
class Base {
public:
    virtual ~Base() { std::cout << "Base destroyed\n"; }
};

class Derived : public Base {
public:
    ~Derived() { std::cout << "Derived destroyed\n"; }
};

void example() {
    Base* ptr = new Derived;
    delete ptr; // invokes Derived::~Derived() and then Base::~Base()
}
```

**Without a virtual destructor**, deleting through a base pointer causes undefined behavior.

## 5. **Destruction Order**

### Non-virtual Inheritance

- Members are destroyed in _reverse order_ of their declaration.
- Base class destructors are called after the derived class destructor finishes.

```cpp
class A {
public: ~A() { std::cout << "A\n"; } };
class B : public A {
public: ~B() { std::cout << "B\n"; } };

// Output of `delete new B();`: B, then A
```

### Virtual Inheritance

For virtual base classes, destruction order is determined by _depth-first, left-to-right_ post-order traversal, ensuring each virtual base is destroyed exactly once.

## 6. **Explicit Destructor Calls**

In rare cases (e.g., placement new), you may need to explicitly call a destructor:

```cpp
String* p = new (buffer) String("temp");
p->~String(); // explicit cleanup
```

**Note:** You should only do this when managing the object lifecycle manually. Otherwise, it's error-prone.

## 7. **Compiler-Generated Destructors**

If you do not define a destructor, the compiler provides one that:

- Calls destructors for all member subobjects.
- Performs _default destruction_ for base classes and fields.

### Rule of Three/Five

If your class manages resources (e.g., raw pointers), define the following:

| Rule  | Functions to Define                           |
| ----- | --------------------------------------------- |
| Three | Destructor, Copy Constructor, Copy Assignment |
| Five  | Plus Move Constructor, Move Assignment        |

Example of why a default copy can be problematic:

```cpp
void copy_strings() {
    String a("oops");
    String b = a; // shallow copy: both a._text and b._text point to the same memory
} // destructor called twice on same pointer → undefined behavior
```

To fix this, provide proper **deep copy** logic and destructor.

## 8. **Common Pitfalls**

- **Calling `delete` instead of `delete[]`**: Causes undefined behavior when deallocating arrays.
- **Omitting `virtual` in base class destructors**: Leads to resource leaks and partial destruction.
- **Failing to follow the Rule of Three/Five**: Introduces shallow copy bugs and memory issues.
- **Throwing exceptions from destructors**: Should be avoided, as it can terminate the program if another exception is active.

## 9. **Example: Demonstrating Destruction Order**

```cpp
#include <iostream>

struct A1      { virtual ~A1() { std::cout << "A1 dtor\n"; } };
struct A2 : A1 { virtual ~A2() { std::cout << "A2 dtor\n"; } };
struct A3 : A2 { virtual ~A3() { std::cout << "A3 dtor\n"; } };

int main() {
    A1* a = new A3;
    delete a;
}
```

### Output:

```
A3 dtor
A2 dtor
A1 dtor
```

Explanation: The virtual table ensures `A3`’s destructor is invoked first, then its base classes.

## 10. **Best Practices**

- Always declare destructors as `virtual` in polymorphic base classes.
- Prefer using RAII-friendly resource wrappers (e.g., `std::unique_ptr`, `std::vector`) instead of raw pointers.
- Avoid manual memory management where possible. Use smart pointers.
- Ensure copy and move operations are defined correctly when resources are involved.
- Avoid side effects or exceptions in destructors.

## Summary

Destructors are essential for writing robust, maintainable C++ code. They ensure that resources are released correctly and predictably, and are tightly coupled with object lifetime management. Proper destructor design is foundational for RAII, exception safety, and polymorphic behavior in complex systems.

By understanding destruction order, virtual behavior, and resource management implications, developers can prevent leaks, corruption, and undefined behavior in high-performance or large-scale C++ applications.
