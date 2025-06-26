# Virtual Functions

### Overview

A _virtual function_ in C++ is a member function that is declared with the `virtual` keyword in a base class and is meant to be overridden in derived classes. The primary purpose of virtual functions is to support **runtime polymorphism**, enabling dynamic dispatch—i.e., function calls that are resolved at runtime based on the actual type of the object being pointed to, rather than the static type of the pointer or reference.

This capability is central to designing **extensible**, **interface-driven**, and **loosely coupled** object-oriented systems.

### Declaration and Behavior

A virtual function is declared in a base class using the `virtual` keyword:

```cpp
class Base {
public:
    virtual void Print();  // virtual function
};
```

When a derived class overrides this function, it provides its own implementation:

```cpp
class Derived : public Base {
public:
    void Print() override;  // override is optional but recommended
};
```

### Example: Dynamic Dispatch

```cpp
#include <iostream>
using namespace std;

class Account {
public:
    Account(double amount) : _balance(amount) {}
    virtual ~Account() {}

    virtual void PrintBalance() {
        cerr << "Base Account: Balance not available." << endl;
    }

protected:
    double _balance;
};

class CheckingAccount : public Account {
public:
    CheckingAccount(double amount) : Account(amount) {}

    void PrintBalance() override {
        cout << "Checking account balance: " << _balance << endl;
    }
};

class SavingsAccount : public Account {
public:
    SavingsAccount(double amount) : Account(amount) {}

    void PrintBalance() override {
        cout << "Savings account balance: " << _balance << endl;
    }
};

int main() {
    Account* acct1 = new CheckingAccount(500.0);
    Account* acct2 = new SavingsAccount(1000.0);

    acct1->PrintBalance();  // Calls CheckingAccount::PrintBalance
    acct2->PrintBalance();  // Calls SavingsAccount::PrintBalance

    delete acct1;
    delete acct2;
}
```

**Output:**

```
Checking account balance: 500
Savings account balance: 1000
```

Despite both pointers being of type `Account*`, the correct derived implementation is called, demonstrating **runtime dispatch**.

### Key Characteristics

#### 1. **Binding**

- **Virtual functions** use **late (dynamic) binding**.
- **Non-virtual functions** use **early (static) binding**.

#### 2. **Virtual Table (vtable)**

- When a class declares a virtual function, the compiler generates a **vtable**—a table of function pointers per class.
- Each object carries a hidden pointer to this table, enabling runtime resolution of the correct method.

#### 3. **Function Matching**

- A derived class overrides a virtual function **only if** the function signatures (return type + parameters) **exactly match**.
- A mismatch results in function hiding, not overriding.

### Suppressing Dynamic Dispatch

You can **explicitly bypass** virtual dispatch using the **scope resolution operator**:

```cpp
Account* p = new CheckingAccount(200);
p->Account::PrintBalance();  // Calls base class implementation
```

This is generally discouraged unless you're intentionally invoking base functionality (e.g., inside an overriding method).

### Best Practices

#### Use `override`

Always mark overriding functions in derived classes with `override`:

```cpp
void PrintBalance() override;
```

This allows the compiler to catch mistakes such as signature mismatches or accidental hiding.

#### Use `final` to prevent further overrides:

```cpp
class CheckingAccount : public Account {
public:
    void PrintBalance() override final;
};
```

#### Use virtual destructors

If a class is meant to be a base class, declare its destructor as virtual:

```cpp
virtual ~Account() {}
```

Failing to do so results in **undefined behavior** when deleting derived objects through a base pointer.

### Comparison: Virtual vs. Non-Virtual Function

```cpp
class Base {
public:
    virtual void VirtualFunc() { cout << "Base::VirtualFunc\n"; }
    void NonVirtualFunc() { cout << "Base::NonVirtualFunc\n"; }
};

class Derived : public Base {
public:
    void VirtualFunc() override { cout << "Derived::VirtualFunc\n"; }
    void NonVirtualFunc() { cout << "Derived::NonVirtualFunc\n"; }
};

int main() {
    Derived d;
    Base* ptr = &d;

    ptr->VirtualFunc();      // Calls Derived::VirtualFunc
    ptr->NonVirtualFunc();   // Calls Base::NonVirtualFunc
}
```

**Output:**

```
Derived::VirtualFunc
Base::NonVirtualFunc
```

This illustrates the importance of the `virtual` keyword in enabling polymorphic behavior.

### Limitations

- **Global and static member functions** cannot be virtual.
- **Constructors** cannot be virtual (although destructors should be).
- **Virtual functions introduce runtime overhead**, typically via a vtable and an indirection cost during invocation.

### Summary

| Feature      | Virtual Function      | Non-Virtual Function        |
| ------------ | --------------------- | --------------------------- |
| Binding Time | Runtime (late)        | Compile-time (early)        |
| Polymorphism | Supported             | Not supported               |
| Overhead     | Slightly higher       | Minimal                     |
| Use Case     | Interface abstraction | Utility or default behavior |

### Advanced: Pure Virtual and Abstract Classes

To make a class abstract, define a pure virtual function:

```cpp
class AbstractAccount {
public:
    virtual void PrintBalance() = 0; // pure virtual
};
```

Such classes **cannot be instantiated** and serve as **interfaces or contracts** for derived types.

### Conclusion

Virtual functions are a fundamental feature in C++ enabling **dynamic polymorphism**. By allowing function calls to be resolved at runtime based on the actual object type, they enable flexible, extensible, and modular system design. Proper use of `virtual`, `override`, and `final` ensures correctness, clarity, and maintainability in object-oriented C++ code.
