# Copy Constructors and Copy Assignment Operators

In C++, object copying is a fundamental part of the language's semantics. Two mechanisms are responsible for defining how objects of a user-defined type are copied:

1. **Copy Constructor** – Invoked during initialization.
2. **Copy Assignment Operator** – Invoked during assignment after initialization.

These operations are critical in ensuring value semantics, safe resource management, and predictable object behavior, particularly when managing dynamic resources like memory or file handles.

### 1. **Copy Constructor**

A **copy constructor** is a special constructor used to create a new object as a copy of an existing one. It has the following canonical form:

```cpp
ClassName(const ClassName& other);
```

#### Best Practice:

Use `const ClassName&` as the parameter to:

- Prevent modifications to the source object.
- Enable copying of `const` objects.

#### Example:

```cpp
class Buffer {
public:
    Buffer(const Buffer& other)
        : size(other.size), data(new int[other.size])
    {
        std::copy(other.data, other.data + size, data);
    }

    ~Buffer() { delete[] data; }

private:
    int* data;
    size_t size;
};
```

Here, the copy constructor performs a **deep copy** to avoid aliasing and double deletion.

### 2. **Copy Assignment Operator**

A **copy assignment operator** assigns the contents of one existing object to another already-initialized object:

```cpp
ClassName& operator=(const ClassName& other);
```

It should return a reference to `*this` to allow chained assignments.

#### Best Practice:

Always check for **self-assignment** and **release previous resources** safely.

#### Example:

```cpp
class Buffer {
public:
    Buffer& operator=(const Buffer& other) {
        if (this == &other)
            return *this; // handle self-assignment

        delete[] data;
        size = other.size;
        data = new int[size];
        std::copy(other.data, other.data + size, data);
        return *this;
    }

private:
    int* data;
    size_t size;
};
```

### 3. **Compiler-Generated Copy Semantics**

If the programmer does **not** define a copy constructor or copy assignment operator:

- The compiler generates a **member-wise copy constructor** and **assignment operator**.
- These default versions perform **shallow copies**, which is usually sufficient for classes with only primitive or value-type members.

#### Example:

```cpp
struct Point {
    int x, y;
    // No copy constructor or assignment operator needed
};

Point p1 = {1, 2};
Point p2 = p1;       // Calls default copy constructor
p2 = p1;             // Calls default copy assignment
```

### 4. **When to Explicitly Define Copy Semantics**

Explicitly define the copy constructor and copy assignment operator when:

- The class **manages resources** (e.g., memory, file descriptors, handles).
- You need **custom behavior** (e.g., logging, shared ownership, or reference counting).
- You need to enforce **deep copy** semantics.

### 5. **Const Correctness in Copying**

Both the copy constructor and copy assignment operator should take their argument as `const ClassName&`. Otherwise:

- Copying from a `const` object will fail.
- Example of incorrect usage:

```cpp
class MyClass {
public:
    MyClass(MyClass& other); // Non-const: cannot copy from const MyClass
};
```

This will cause compilation errors when passing a `const MyClass` to the constructor.

### 6. **Virtual Base Considerations**

When a class has **virtual base classes**, either user-defined or compiler-generated copy constructors initialize the virtual base **only once**, at the most-derived level. This behavior avoids multiple copies of the base class and follows the diamond inheritance rules.

### 7. **Rule of Three (Pre-C++11)**

If your class needs a **user-defined** copy constructor, copy assignment operator, or destructor, **define all three**. This is because the need to manage resources typically implies non-trivial behavior for all of them.

```cpp
class MyClass {
public:
    MyClass(const MyClass&);            // Copy constructor
    MyClass& operator=(const MyClass&); // Copy assignment operator
    ~MyClass();                         // Destructor
};
```

### 8. **Example: Copy Semantics in Practice**

```cpp
#include <iostream>
#include <cstring>

class String {
public:
    String(const char* src = "") {
        data = new char[std::strlen(src) + 1];
        std::strcpy(data, src);
    }

    // Copy constructor
    String(const String& other) {
        data = new char[std::strlen(other.data) + 1];
        std::strcpy(data, other.data);
    }

    // Copy assignment operator
    String& operator=(const String& other) {
        if (this == &other)
            return *this;
        delete[] data;
        data = new char[std::strlen(other.data) + 1];
        std::strcpy(data, other.data);
        return *this;
    }

    ~String() {
        delete[] data;
    }

    void print() const { std::cout << data << std::endl; }

private:
    char* data;
};

int main() {
    String a("Hello");
    String b = a; // copy constructor
    String c;
    c = b;        // copy assignment
    c.print();    // Output: Hello
}
```

### Summary

| Feature                  | Purpose                       | Best Practice                                   |
| ------------------------ | ----------------------------- | ----------------------------------------------- |
| Copy Constructor         | Initializes a new object      | Accept argument as `const ClassName&`           |
| Copy Assignment Operator | Assigns to an existing object | Handle self-assignment, release old resources   |
| Default Behavior         | Member-wise shallow copy      | Override when managing resources                |
| Rule of Three            | Define all 3 if defining any  | Ensures safe and consistent resource management |

For modern C++ (C++11 and beyond), consider the **Rule of Five** (which includes move semantics), and use `= default` or `= delete` to express intent explicitly.
