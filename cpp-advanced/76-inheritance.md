### Inheritance

Inheritance is a core pillar of object-oriented programming (OOP) in C++, enabling a class (called the **derived class**) to inherit properties and behaviors (data members and member functions) from another class (called the **base class**). This promotes **code reuse**, **modularity**, and **extensibility**, especially in large-scale systems.

## 1. **Syntax of Inheritance**

```cpp
class Base {
    // Members of the base class
};

class Derived : access-specifier Base {
    // Additional members or overrides
};
```

- **Access Specifier**: Can be `public`, `protected`, or `private`.
- **Default Access**: In classes, inheritance is **private** by default if not explicitly specified.

## 2. **Types of Inheritance**

### a. **Single Inheritance**

```cpp
class Animal {
public:
    void speak() const { std::cout << "Animal sound\n"; }
};

class Dog : public Animal {
public:
    void bark() const { std::cout << "Woof\n"; }
};
```

#### Example Usage:

```cpp
Dog d;
d.speak();  // Inherited
d.bark();   // Defined in Dog
```

### b. **Multiple Inheritance**

```cpp
class Swimmer {
public:
    void swim() const { std::cout << "Swimming\n"; }
};

class Flyer {
public:
    void fly() const { std::cout << "Flying\n"; }
};

class Duck : public Swimmer, public Flyer {};
```

#### Example Usage:

```cpp
Duck d;
d.swim();
d.fly();
```

### c. **Multilevel Inheritance**

```cpp
class Vehicle {
public:
    void start() const { std::cout << "Starting\n"; }
};

class Car : public Vehicle {};

class SportsCar : public Car {};
```

### d. **Hierarchical Inheritance**

Multiple derived classes from the same base class.

## 3. **Access Control in Inheritance**

| Base Member Access | `public` Inheritance | `protected` Inheritance | `private` Inheritance |
| ------------------ | -------------------- | ----------------------- | --------------------- |
| `public`           | `public`             | `protected`             | `private`             |
| `protected`        | `protected`          | `protected`             | `private`             |
| `private`          | Inaccessible         | Inaccessible            | Inaccessible          |

**Note**: Private members are never directly inherited; they are only accessible through public or protected member functions of the base.

## 4. **Virtual Inheritance**

Used to solve the **"diamond problem"** in multiple inheritance where a base class is inherited multiple times through different paths.

### Diamond Problem Example:

```cpp
class A { public: void f() const { std::cout << "A::f\n"; } };
class B : virtual public A {};
class C : virtual public A {};
class D : public B, public C {};
```

Without virtual inheritance, class `A` would be duplicated. Using `virtual` ensures only one shared instance of `A` is present.

### Usage:

```cpp
D d;
d.f(); // OK: Ambiguity resolved via virtual inheritance
```

## 5. **Overriding and Polymorphism**

To override base class behavior:

```cpp
class Base {
public:
    virtual void log() const { std::cout << "Base log\n"; }
};

class Derived : public Base {
public:
    void log() const override { std::cout << "Derived log\n"; }
};
```

### Usage:

```cpp
Base* b = new Derived();
b->log(); // Outputs "Derived log"
```

- **`virtual`** enables dynamic dispatch.
- Use `override` to explicitly mark overriding functions.
- Use `final` to prevent further overriding.

## 6. **Best Practices**

- Prefer **public inheritance** to model **"is-a"** relationships.

- Avoid multiple inheritance unless necessary; prefer **composition**.

- Use **virtual destructors** in base classes intended for polymorphic use:

  ```cpp
  class Base {
  public:
      virtual ~Base() {}
  };
  ```

- Limit `protected` inheritance—it suggests tight coupling and poor encapsulation.

- Use `final` for classes or methods you don't want to be derived or overridden:

  ```cpp
  class FinalClass final {};
  ```

## 7. **Example: Practical Design**

```cpp
class Logger {
public:
    virtual void log(const std::string& msg) const {
        std::cout << "Log: " << msg << '\n';
    }
    virtual ~Logger() = default;
};

class FileLogger : public Logger {
public:
    void log(const std::string& msg) const override {
        std::cout << "File Log: " << msg << '\n';
    }
};
```

### Usage:

```cpp
std::unique_ptr<Logger> logger = std::make_unique<FileLogger>();
logger->log("Hello");
```

## 8. **Conclusion**

Inheritance in C++ provides a structured way to promote **code reuse**, enable **polymorphism**, and define **hierarchies** of related types. However, it should be used judiciously:

- Use inheritance when there's a clear "is-a" relationship.
- Combine with **composition** for flexible designs.
- Virtual inheritance should be reserved for scenarios involving shared base classes in complex hierarchies.

Understanding and applying the appropriate inheritance model is essential for designing robust, maintainable, and extensible C++ systems.
