# C++ OOP - Access Specifiers

## What Are Access Specifiers?

**Access specifiers** control the visibility and accessibility of class members (attributes and methods).
They help **protect data** and ensure that code is used correctly and safely.

## Types of Access Specifiers in C++

| Specifier   | Description                                                                |
| ----------- | -------------------------------------------------------------------------- |
| `public`    | Members are accessible from outside the class                              |
| `private`   | Members **cannot** be accessed from outside the class                      |
| `protected` | Members cannot be accessed from outside, but can be accessed in subclasses |

## Using `public`

When class members are declared `public`, they can be accessed and modified from outside the class.

### Example:

```cpp
class MyClass {
  public:
    int x;
};

int main() {
  MyClass obj;
  obj.x = 25;   // Allowed
  cout << obj.x;
  return 0;
}
```

## Using `private`

When members are `private`, they are hidden from outside access.

### Example:

```cpp
class MyClass {
  public:
    int x;   // Public
  private:
    int y;   // Private
};

int main() {
  MyClass obj;
  obj.x = 25;    // Allowed
  obj.y = 50;    // Error: 'y' is private
  return 0;
}
```

### Result:

```
error: 'int MyClass::y' is private within this context
```

> ✅ Best practice: Declare attributes as `private` and control access through `public` methods. This principle is called **Encapsulation** (covered in the next part).

## Default Access Specifier

If no access specifier is provided, members are considered `private` by default in a class.

### Example:

```cpp
class MyClass {
  int x;   // Private by default
};
```

## Using `protected`

Protected members are **not accessible from outside** the class, but **can be accessed by derived (child) classes**.

## Real-Life Analogy

- **Public**: Like the front door of your house – anyone can come in.
- **Private**: Like a locked drawer – only you (or authorized methods) can open it.
- **Protected**: Like a family-only room – only children (subclasses) have access.
