# Constructor

## 1. **What is a Constructor?**

A **constructor** is a special member function of a class that is automatically invoked when an object is created. It has the same name as the class and **does not return a value**.

### Key Characteristics:

- Automatically invoked at object creation.
- Can be **overloaded** to allow different initialization scenarios.
- Can use **initializer lists** to initialize members efficiently.
- Supports **default, copy, move, delegating**, and **inheriting** forms.

## 2. **Types of Constructors**

### a. **Default Constructor**

A constructor that takes no parameters (or only default parameters).

```cpp
class A {
public:
    A() {}  // default constructor
};
```

If no constructor is defined, the compiler will generate one implicitly, provided no other constructors are declared.

**Best Practice**: Always initialize data members, either in-class or using initializer lists.

### b. **Parameterized Constructor**

Allows object creation with specific values:

```cpp
class Box {
public:
    Box(int w, int h, int d) : width(w), height(h), depth(d) {}
private:
    int width, height, depth;
};
```

Overloaded versions allow flexibility in object creation.

### c. **Member Initializer Lists**

More efficient than assignments in the constructor body, especially for:

- `const` members
- Reference members
- Class-type members with no default constructor

```cpp
Box(int w, int h) : width(w), height(h) {}
```

**Tip**: Initialize members in the same order they are declared to avoid compiler warnings.

### d. **Copy Constructor**

Used when creating a new object as a copy of an existing one.

```cpp
Box(const Box& other);  // recommended form
```

The compiler will generate a shallow copy constructor unless a user-defined one is provided.

Use **deep copies** when managing resources (e.g., heap memory or file handles).

```cpp
Box(const Box& other) {
    data = new int[*other.data]; // Deep copy
}
```

You must also define the copy **assignment operator** in this case.

### e. **Move Constructor (C++11+)**

Used to efficiently transfer ownership of resources:

```cpp
Box(Box&& other) noexcept
    : data(std::move(other.data)) {}
```

Move semantics avoid expensive copies, especially for large objects like vectors or strings.

**Rule of Five**: If you define any of the following, you should define all of them:

- Destructor
- Copy constructor
- Copy assignment
- Move constructor
- Move assignment

### f. **Delegating Constructor (C++11+)**

Allows one constructor to invoke another:

```cpp
Box(int size) : Box(size, size, size) {}  // Delegates to 3-param constructor
```

Avoids duplication and centralizes initialization logic.

### g. **Inheriting Constructors (C++11+)**

A derived class can inherit constructors from a base class using `using`:

```cpp
class Base {
public:
    Base(int) {}
};

class Derived : public Base {
public:
    using Base::Base;  // Inherit constructors
};
```

Caution: Does not initialize derived-specific members.

### h. **Explicit Constructors**

Prevent implicit conversions from types:

```cpp
explicit Box(int size);
```

```cpp
Box b = 5;      // Error if constructor is explicit
Box b(5);       // OK
```

**Best Practice**: Prefer `explicit` unless implicit conversion is intentional.

### i. **constexpr Constructor (C++11+)**

Enables compile-time constant initialization:

```cpp
class Point {
public:
    constexpr Point(int x, int y) : x_(x), y_(y) {}
private:
    int x_, y_;
};
```

Constraints:

- All fields must be `constexpr`-compatible
- No dynamic memory allocation

### j. **Deleted Constructors**

Prevent object creation or copying:

```cpp
Box() = delete;
Box(const Box&) = delete;
```

Useful for singleton patterns, resource-locked objects, or types not meant to be copied.

## 3. **Construction Order**

### Initialization proceeds in this strict order:

1. **Base class constructors** (from topmost in hierarchy)
2. **Non-static data members**, in the order declared
3. **Derived class constructor body**

```cpp
class Base {
public:
    Base() { std::cout << "Base\n"; }
};

class Derived : public Base {
public:
    Derived() { std::cout << "Derived\n"; }
};
```

**Output**:

```
Base
Derived
```

**Best Practice**: Always match initialization list order with member declaration order.

## 4. **Composite and Aggregated Classes**

If a member is of a class type that lacks a default constructor, you **must initialize it** explicitly via the initializer list:

```cpp
class Label {
public:
    Label(std::string s) : text(std::move(s)) {}
private:
    std::string text;
};

class Box {
    Label label;
public:
    Box(std::string labelText) : label(std::move(labelText)) {}
};
```

## 5. **Aggregate Initialization & C++17 Considerations**

In C++17, aggregate initialization rules have changed, especially with private base class constructors. Using `{}` initialization may attempt direct member initialization even across access boundaries.

```cpp
struct Base {
private:
    Base() {}
    friend struct Derived;
};

struct Derived : Base {
    Derived() {} // Required to allow `{}` init in C++17
};
```

## 6. **Constructor Overload Resolution**

Constructors participate in overload resolution based on:

- Argument count
- Argument types
- `explicit` specifier
- Default arguments
- Presence of initializer_list constructor

```cpp
Box b1{1};              // May prefer initializer_list constructor
Box b2(1, 2, 3);         // Matches specific constructor
Box b3 = 42;             // Invokes implicit conversion unless constructor is explicit
```

## 7. **Most Vexing Parse**

```cpp
Box b(); // Declares a function, not an object
```

To create a default-initialized object, prefer:

```cpp
Box b;        // OK
Box b{};      // Preferred in modern C++
```

## 8. **Common Pitfalls**

- Not initializing `const` or reference members in the initializer list (compile error).
- Forgetting to define a destructor when handling raw resources (memory leaks).
- Using default constructor syntax that triggers the Most Vexing Parse.
- Not marking single-parameter constructors as `explicit`, leading to unwanted conversions.
- Declaring a move constructor disables the implicit copy constructor.

## Summary: Best Practices

- Prefer member initializer lists over assignments in constructor body.
- Use `explicit` for single-parameter constructors unless implicit conversion is needed.
- Always define move/copy constructors and assignment operators when managing resources.
- Use `=default` and `=delete` for clarity and intent.
- Use `constexpr` constructors for compile-time constants when applicable.
- Favor delegating constructors to reduce duplication.
- Always match initialization order with declaration order for safety and clarity.

Constructors are the foundation of class design in C++, and mastering their subtleties ensures correctness, clarity, and performance across your codebase. Whether designing RAII-style resource holders, composite objects, or high-performance containers, effective constructor use distinguishes high-quality engineering.
