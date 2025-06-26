# Member Access Control

Access control is one of the foundational pillars of encapsulation in C++. It allows a class to hide internal implementation details, expose public APIs, and selectively share internals with derived types or trusted functions/classes (friends). It is enforced at compile time and helps build robust, maintainable, and secure abstractions.

## Access Specifiers

C++ provides three access specifiers:

| Specifier   | Access Level                                              |
| ----------- | --------------------------------------------------------- |
| `private`   | Accessible only by the class itself and its friends       |
| `protected` | Accessible by the class, its friends, and derived classes |
| `public`    | Accessible by all                                         |

### Example: Basic Usage

```cpp
class Point {
public:
    Point(int x, int y) : _x(x), _y(y) {}
    int& x() { return _x; }
    int& y() { return _y; }

protected:
    void transformToScreenSpace();  // For derived classes

private:
    int _x, _y;                     // Implementation details
};
```

### Key Characteristics

- Access specifiers affect all subsequent declarations until another specifier is encountered.
- C++ classes default to `private` access; `struct`s default to `public`.

## Access in Derived Classes

### Overview of Access Inheritance Rules

Let’s consider a base class member and how its access translates in derived classes:

| Base Member Access | `public` Inheritance | `protected` Inheritance | `private` Inheritance |
| ------------------ | -------------------- | ----------------------- | --------------------- |
| `public`           | public               | protected               | private               |
| `protected`        | protected            | protected               | private               |
| `private`          | inaccessible         | inaccessible            | inaccessible          |

### Example

```cpp
class Base {
public:
    void publicMethod();
protected:
    void protectedMethod();
private:
    void privateMethod();
};

class PublicDerived : public Base {
    void foo() {
        publicMethod();     // OK
        protectedMethod();  // OK
        // privateMethod(); // Error
    }
};

class PrivateDerived : private Base {
    void foo() {
        publicMethod();     // OK but becomes private
        protectedMethod();  // OK but becomes private
    }
};

void test() {
    PublicDerived pd;
    pd.publicMethod();      // OK

    PrivateDerived prd;
    // prd.publicMethod();  // Error
}
```

## Inheritance Defaults and Implications

- `class Derived : Base` → private inheritance by default
- `struct Derived : Base` → public inheritance by default

**Best Practice:** Explicitly specify inheritance visibility to avoid confusion.

## Friends and Access Control

`friend` declarations allow otherwise inaccessible members to be accessed by the specified friend functions or classes. This breaks encapsulation but is sometimes necessary, e.g., for operator overloading or builder patterns.

```cpp
class A {
    friend class B;  // Class B can access A's private/protected members
private:
    int secret = 42;
};
```

## Access Control with Static Members

Static members obey access rules for lookup, but the context of usage may bypass restrictions.

```cpp
class Base {
public:
    static int StaticFunc();
};

class Derived : private Base {};

void example(Derived* d) {
    Base::StaticFunc();     // OK, explicit scope avoids access restrictions
    // d->StaticFunc();     // Error: Base is a private base of Derived
}
```

**Insight:** Access control applies to the _conversion_, not the function itself. The implicit `this` pointer needs conversion, which may be disallowed.

## Access Control and Virtual Functions

Access control applies to the _static type_ used in the call, not the override itself.

```cpp
class VBase {
public:
    virtual int getValue() { return 1; }
};

class VDerived : public VBase {
private:
    int getValue() override { return 2; }
};

int main() {
    VDerived d;
    VBase* basePtr = &d;
    int x = basePtr->getValue();  // OK: access granted via VBase
    // d.getValue();              // Error: getValue is private in VDerived
}
```

**Note:** Despite calling `VDerived::getValue`, the access check is on `VBase::getValue`.

## Multiple Inheritance and Ambiguous Access

When a base class is inherited via different access specifiers (e.g., one path public, another private), the _most accessible path_ is used.

```cpp
class VBase {};

class LeftPath : virtual private VBase {};
class RightPath : virtual public VBase {};

class Derived : public LeftPath, public RightPath {
    // VBase is reachable via RightPath → access is public
};
```

**Recommendation:** Avoid ambiguous inheritance paths when possible, or use clear virtual base access and consistent specifiers.

## Best Practices for Access Control

1. **Use `private` by default**: Expose only what is necessary.
2. **Use `protected` sparingly**: It can expose internals to subclasses, violating encapsulation.
3. **Avoid `friend` unless justified**: Overuse indicates flawed design.
4. **Explicitly state base class access**: Improves code clarity.
5. **Leverage composition over inheritance** when access complexity becomes unmanageable.
6. **Keep virtual overrides consistent in access levels** to avoid confusion.

## Summary

Access control in C++ enforces encapsulation and helps maintain a clean separation between interface and implementation. It extends across class members, inheritance hierarchies, static context, and virtual dispatch, and interacts subtly with type conversion and multiple inheritance. Proper use of access specifiers promotes modular, testable, and maintainable code.
