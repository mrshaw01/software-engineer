# Overview of Member Functions

Member functions are functions that operate on instances of a class and have access to that class’s data members. In C++, member functions fall into two categories:

- **Non-static member functions**, which operate on specific instances of a class and have access to the `this` pointer.
- **Static member functions**, which do not require an object instance and do not have access to `this`.

### 1. **Defining Member Functions**

#### a. _Inside the Class Declaration_

When a member function is defined inside the class body, it is **implicitly inline**, meaning its definition is inserted at the point of call (if the compiler chooses to do so), potentially reducing function call overhead.

```cpp
class Account {
public:
    double Deposit(double amount) {
        balance += amount;
        return balance;
    }
private:
    double balance = 0.0;
};
```

**Key Point:** There's no need to prefix the function name with the class name when defined inline.

#### b. _Outside the Class Declaration_

When defined outside the class, the function must be **qualified** with the class name using the scope resolution operator `::`. It is _not_ implicitly inline unless explicitly marked with the `inline` keyword.

```cpp
class Account {
public:
    double Deposit(double amount);
private:
    double balance = 0.0;
};

inline double Account::Deposit(double amount) {
    balance += amount;
    return balance;
}
```

### 2. **Static Member Functions**

Static functions belong to the class rather than to any object instance. They cannot access non-static members or `this`.

```cpp
class Logger {
public:
    static void Log(const std::string& msg) {
        std::cout << "[LOG]: " << msg << "\n";
    }
};

int main() {
    Logger::Log("Static call");
}
```

### 3. **Virtual Member Functions**

The `virtual` keyword enables **dynamic dispatch**, where the function to invoke is determined at runtime based on the actual type of the object.

```cpp
class Shape {
public:
    virtual void Draw() const {
        std::cout << "Drawing a generic shape\n";
    }
};

class Circle : public Shape {
public:
    void Draw() const override {
        std::cout << "Drawing a circle\n";
    }
};

void Render(const Shape& s) {
    s.Draw();  // Calls Circle::Draw if s is a Circle
}
```

**Best Practice:** Always use `virtual` for polymorphic base class methods that you intend to override.

### 4. **`override` Specifier**

C++11 introduced `override` to indicate that a function is intended to override a base class method. This allows the compiler to catch errors such as signature mismatches or missing `virtual`.

#### Without `override` – Silent Bugs:

```cpp
class Base {
public:
    virtual void func(int) const {}
};

class Derived : public Base {
    void func(int) {}  // Different signature (non-const), no compiler error
};
```

#### With `override` – Compile-time Check:

```cpp
class Derived : public Base {
    void func(int) override; // Error: does not override Base::func(int) const
};
```

**Best Practice:** Always use `override` for clarity and safety when overriding virtual functions.

### 5. **`final` Specifier**

Use `final` to prevent further overrides or inheritance:

#### a. _Prevent Overriding a Virtual Function_

```cpp
class Base {
public:
    virtual void Run() final;
};

class Derived : public Base {
    void Run(); // Error: cannot override a final function
};
```

#### b. _Prevent Inheriting from a Class_

```cpp
class Utility final {};

class ExtendedUtility : public Utility {}; // Error: Utility is final
```

**Use Case:** Apply `final` to lock down APIs or internal implementations.

### 6. **Important Constraints**

- **One Definition Rule (ODR):** A member function must have only one definition per translation unit. If declared inline, all definitions across translation units must be identical.
- **Member functions must be defined within the lifetime of the class definition.** You cannot append new member functions to a class once it has been declared.

### Summary Table

| Specifier  | Purpose                                  | Applies To                   | Notes                                              |
| ---------- | ---------------------------------------- | ---------------------------- | -------------------------------------------------- |
| `inline`   | Suggest inlining at call site            | Member functions             | Implicit for in-class definitions                  |
| `static`   | Belongs to class, not object             | Member functions             | No `this` pointer; cannot access non-static fields |
| `virtual`  | Enables runtime polymorphism             | Non-static member functions  | Must be used in base class                         |
| `override` | Asserts function is overriding virtual   | Overriding functions         | Helps catch signature mismatches                   |
| `final`    | Prevents further override or inheritance | Virtual functions or classes | Enhances safety and API control                    |

### Final Thoughts

A thorough understanding of member function behavior, declaration styles, and object-oriented tools like `virtual`, `override`, and `final` is essential for writing safe, maintainable, and extensible C++ code. Proper use of these features allows for expressive design patterns, robust inheritance hierarchies, and efficient software abstractions.
