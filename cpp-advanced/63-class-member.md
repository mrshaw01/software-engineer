### Class Member

In C++, a `class` or `struct` is a user-defined type that encapsulates data (state) and behavior (functions). The internal components of a class—its _members_—define both what data the class holds and how it operates on that data. This encapsulation is central to object-oriented design and is a foundation for principles like abstraction, inheritance, and polymorphism.

#### Categories of Class Members

The key categories of members in a class are:

1. **Data Members**
2. **Member Functions**
3. **Special Member Functions**
4. **Static Members**
5. **Mutable Members**
6. **Operators**
7. **Nested Types (Classes, Unions, Enums)**
8. **Bit Fields**
9. **Friends**
10. **Type Aliases and Typedefs**

Each category plays a unique role in enabling class functionality and maintaining a clean, modular design.

### 1. **Data Members**

These are the variables defined inside a class. Each object (instance) of the class holds its own copy of non-static data members.

```cpp
class Example {
private:
    int value;           // Instance data member
    std::string name;
};
```

Since C++11, data members can be initialized in-place:

```cpp
int value = 42;          // Default initialization
std::string name = "John";
```

### 2. **Member Functions**

These are functions defined within the class that operate on the class’s data members.

```cpp
class Example {
public:
    void setValue(int v) { value = v; }
    int getValue() const { return value; }

private:
    int value;
};
```

Member functions can be:

- **const-qualified**, ensuring they don’t modify the object
- **virtual**, allowing for runtime polymorphism
- **inline**, enabling compiler inlining (if defined in-class)

### 3. **Special Member Functions**

The compiler automatically generates these if not explicitly defined:

| Function                 | Purpose                                 |
| ------------------------ | --------------------------------------- |
| Default constructor      | `Example()`                             |
| Copy constructor         | `Example(const Example&)`               |
| Move constructor (C++11) | `Example(Example&&)`                    |
| Copy assignment operator | `Example& operator=(const Example&)`    |
| Move assignment operator | `Example& operator=(Example&&)` (C++11) |
| Destructor               | `~Example()`                            |

Best practice: explicitly define or delete these if your class manages resources.

```cpp
Example() = default;
Example(const Example&) = delete;
Example(Example&&) noexcept = default;
~Example();
```

### 4. **Static Members**

Static members are shared across all instances of the class. They must be defined outside the class body.

```cpp
class Counter {
public:
    static int instanceCount;

    Counter() { ++instanceCount; }
};

int Counter::instanceCount = 0;
```

A static member function can only access other static members.

### 5. **Mutable Members**

The `mutable` keyword allows modification even in `const` member functions.

```cpp
class Logger {
public:
    void log() const { ++counter; }

private:
    mutable int counter = 0;
};
```

### 6. **Operators**

Operators can be overloaded as member or non-member functions to provide custom behavior.

```cpp
class Vector {
public:
    Vector operator+(const Vector& other) const;
};
```

### 7. **Nested Types**

Classes can contain nested types such as:

- **Nested Classes**
- **Unions**
- **Enumerations**

```cpp
class Outer {
public:
    enum class Status { OK, Error };

    class Inner {
        // ...
    };
};
```

### 8. **Bit Fields**

Bit fields allow the packing of data into smaller storage units.

```cpp
struct Flags {
    unsigned int isVisible : 1;
    unsigned int isEnabled : 1;
};
```

### 9. **Friends**

Friend declarations grant non-members access to private or protected members.

```cpp
class Secret {
    friend void reveal(const Secret&);
private:
    int code = 42;
};
```

Although friends are declared _within_ the class, they are not members and are not subject to access control.

### 10. **Type Aliases and Typedefs**

Used to create synonyms for types.

```cpp
class Network {
public:
    using IP = std::string;
    typedef unsigned short Port;
};
```

### Example: Comprehensive Class

```cpp
class TestRun {
public:
    TestRun() = default;
    TestRun(const TestRun&) = delete;
    TestRun(std::string name);
    void DoSomething();
    int Calculate(int a, double d);
    virtual ~TestRun();

    enum class State { Active, Suspended };

protected:
    virtual void Initialize();
    virtual void Suspend();
    State GetState();

private:
    State _state{ State::Suspended };
    std::string _testName{ "" };
    int _index{ 0 };

    static int _instances;
};

// Definition of static member
int TestRun::_instances = 0;
```

**Explanation:**

- Demonstrates constructors, access specifiers, static members, default member initialization, enum types, and a virtual destructor.
- This design is extensible, encapsulated, and correctly resource-safe.

### Summary

Understanding class members in depth is crucial for writing robust, idiomatic, and maintainable C++. Key takeaways include:

- Favor initialization at point-of-declaration (C++11+).
- Explicitly manage special member functions for RAII or rule-of-five compliance.
- Use `static` members judiciously when shared state is truly needed.
- Isolate implementation details using access specifiers (`private`, `protected`, `public`).
- Prefer `enum class` over plain `enum` for scoped and type-safe enumerations.

This structure lays the foundation for more advanced topics such as inheritance, polymorphism, templates, and metaprogramming.
