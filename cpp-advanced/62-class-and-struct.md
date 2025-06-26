# C++ Classes and Structs

In C++, both `class` and `struct` are mechanisms for defining user-defined types that encapsulate data (members) and associated behavior (member functions). Although syntactically similar and functionally equivalent, they differ in **default access control** and **usage idioms** rooted in historical and conventional design patterns.

## 1. **Class vs Struct: Core Similarity and Key Difference**

| Feature             | `class`                           | `struct`                        |
| ------------------- | --------------------------------- | ------------------------------- |
| Keyword             | `class`                           | `struct`                        |
| Default Access      | `private`                         | `public`                        |
| Inheritance Default | `private`                         | `public`                        |
| Use Case Convention | Used for encapsulated OOP designs | Used for simple data aggregates |
| Functionality       | Full-featured user-defined type   | Identical to class in C++       |

> Technically, in C++, `class` and `struct` behave the same apart from default access levels. However, the distinction in usage improves code readability and communicates intent.

## 2. **Class Declaration Syntax and Example**

### Syntax

```cpp
class ClassName [: access BaseClass] {
    // members: data and functions
};
```

### Example: Basic Class with Encapsulation and Inheritance

```cpp
#include <iostream>
#include <string>

class Dog {
public:
    Dog() : legs(4), barks(true) {}

    void setSize(const std::string& size) { dogSize = size; }

    virtual void setEars(const std::string& type) { earType = type; }

protected:
    std::string dogSize, earType;

private:
    int legs;
    bool barks;
};

class Breed : public Dog {
public:
    Breed(const std::string& color, const std::string& size)
        : color(color) {
        setSize(size);
    }

    std::string getColor() const { return color; }

    void setEars(const std::string& length, const std::string& type) {
        earLength = length;
        earType = type;
    }

private:
    std::string color, earLength;
};
```

### Output Usage:

```cpp
int main() {
    Breed labrador("yellow", "large");
    labrador.setEars("long", "floppy");
    std::cout << "Cody is a " << labrador.getColor() << " labrador\n";
}
```

> **Expected Output:**

```
Cody is a yellow labrador
```

## 3. **Struct Declaration Syntax and Example**

### Syntax

```cpp
struct StructName {
    // members (default public)
};
```

### Example: Struct with Bit Fields and Aggregate Initialization

```cpp
#include <iostream>

struct Person {
    int age;
    long ssn;
    float weight;
    char name[25];
};

struct Cell {
    unsigned short character  : 8;
    unsigned short foreground : 3;
    unsigned short intensity  : 1;
    unsigned short background : 3;
    unsigned short blink      : 1;
};

int main() {
    Person sister = {13, 123456789, 50.5f, "Alice"};
    Person brother = {7, 987654321, 40.0f, "Bob"};

    std::cout << "sister.age = " << sister.age << '\n';
    std::cout << "brother.age = " << brother.age << '\n';

    Cell myCell = {};
    myCell.character = 1;

    std::cout << "myCell.character = " << myCell.character << '\n';
}
```

> **Expected Output:**

```
sister.age = 13
brother.age = 7
myCell.character = 1
```

## 4. **Advanced Topics**

### a. **Member Access Control**

- `private`: Accessible only inside the class.
- `protected`: Accessible inside the class and derived classes.
- `public`: Accessible from anywhere.

### b. **Inheritance**

```cpp
class Base {};
class Derived : public Base {};        // public inheritance
class Derived2 : protected Base {};    // protected inheritance
class Derived3 : private Base {};      // private inheritance
```

### c. **Static Members**

Static members belong to the class rather than any instance.

```cpp
class Counter {
public:
    static int count;
};

int Counter::count = 0;
```

### d. **User-Defined Type Conversions**

Allow implicit or explicit conversion to other types.

```cpp
class Fraction {
public:
    operator double() const { return numerator / (double)denominator; }

private:
    int numerator = 1, denominator = 2;
};
```

### e. **Mutable Members**

Allows a member to be modified even if the containing object is `const`.

```cpp
class Logger {
public:
    void log() const { ++call_count; }

private:
    mutable int call_count = 0;
};
```

### f. **Nested Classes**

```cpp
class Outer {
    class Inner {
        // Only accessible through Outer
    };
};
```

### g. **Pointers to Members and `this` Pointer**

```cpp
class Sample {
public:
    int value = 5;
    void print() {
        std::cout << "Value: " << this->value << '\n';
    }
};
```

## 5. **Unions and Anonymous Classes**

- `union` allows storage of different data types in the same memory location.
- Anonymous class types can be defined without naming the class explicitly. This is rarely used in modern C++ due to maintainability concerns.

## 6. **Best Practices**

- Use `struct` for **simple data aggregates** (PODs) or value types (like vectors, coordinates).
- Use `class` for **encapsulated types**, **object-oriented behavior**, or anything that requires **encapsulation**, **abstraction**, or **polymorphism**.
- Always specify access levels explicitly to improve clarity.
- Avoid overusing inheritance. Prefer composition over inheritance unless polymorphism is needed.

## Conclusion

C++ `class` and `struct` are powerful constructs that underpin user-defined types. While functionally similar, their **default access control** and **conventional usage** make each suited for different scenarios. Understanding when and how to use each, along with best practices around encapsulation, inheritance, and resource management, is essential to designing robust and maintainable C++ software systems.
