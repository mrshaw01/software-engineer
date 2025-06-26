# `friend`

In C++, encapsulation ensures that the internal state and implementation details of a class are shielded from unauthorized access. However, in some scenarios, certain functions or other classes may require special access to a class’s `private` or `protected` members. C++ provides the `friend` keyword to grant such access selectively. This mechanism is a controlled breach of encapsulation, justified in well-reasoned design scenarios.

## 1. **What Is a Friend in C++?**

A **friend** is a function, method, or entire class that is explicitly allowed access to another class’s `private` and `protected` members. Importantly:

- Friendship is **not mutual** (A can be friend to B without the inverse).
- Friendship is **not inherited** (a derived class doesn't inherit its base’s friends).
- Friendship is **not transitive** (a friend of a friend is not automatically a friend).

## 2. **Friend Function**

A **friend function** is a non-member function granted access to a class's internal state.

### Example:

```cpp
#include <iostream>
class Point {
    friend void ChangePrivate(Point&);  // Declare friend function

public:
    Point() : x(0) {}
    void Print() const { std::cout << x << '\n'; }

private:
    int x;
};

void ChangePrivate(Point& p) {
    p.x += 1;  // Accessing private member
}

int main() {
    Point p;
    p.Print();         // Output: 0
    ChangePrivate(p);
    p.Print();         // Output: 1
}
```

**Explanation:**
The `ChangePrivate` function is not a member of `Point` but has access to its private data member `x`.

## 3. **Friend Class**

A **friend class** allows all its member functions to access the private/protected members of another class.

### Example:

```cpp
class SecretHolder {
    friend class Key;  // Declare friend class

private:
    int secret = 42;
};

class Key {
public:
    int unlock(const SecretHolder& s) {
        return s.secret;  // Accessing private data
    }
};
```

**Explanation:**
Every member of `Key` can access `SecretHolder`'s private and protected members.

## 4. **Friend Member Function**

Instead of making an entire class a friend, you can declare **only specific member functions** of another class as friends.

### Example:

```cpp
class B;  // Forward declaration

class A {
public:
    int expose(B& b);

private:
    int hidden(B& b);
};

class B {
private:
    int value = 10;
    friend int A::expose(B&);  // Only expose is a friend
};

int A::expose(B& b) { return b.value; }     // Allowed
int A::hidden(B& b) { return b.value + 1; } // Error: not a friend
```

## 5. **Friend Templates**

In template classes, you can declare type parameters as friends or define friend functions/templates within templates.

### Example:

```cpp
template <typename T>
class MyClass {
    friend T;  // T must be a class with access needs
};
```

You can also define a friend function template:

```cpp
template <typename T>
class Container {
    friend void inspect<>(const Container<T>&);
private:
    T value;
};

template <typename T>
void inspect(const Container<T>& c) {
    std::cout << c.value << '\n';
}
```

## 6. **Friend Typedefs and Aliases**

You can declare a typedef or type alias as a friend:

```cpp
class Foo {};
typedef Foo Bar;

class G {
    friend Bar;      // OK
    friend class Bar; // Error: redefinition
};
```

## 7. **Friendship Rules Summary**

| Rule                       | Explanation                                                     |
| -------------------------- | --------------------------------------------------------------- |
| **Explicit Only**          | A class must explicitly declare who its friends are.            |
| **Not Mutual**             | If A is a friend of B, B is not automatically a friend of A.    |
| **Not Transitive**         | A friend of a friend is not a friend.                           |
| **Not Inherited**          | Base class friends are not inherited by derived classes.        |
| **Inline Friend Function** | Can be defined within the class declaration; treated as inline. |

## 8. **Best Practices and Recommendations**

### When to Use `friend`:

- **Operator Overloads**: Especially binary operators like `operator+` that need access to private members of both operands.
- **Encapsulation Across Types**: When a tightly-coupled helper or controller class needs access.
- **Serialization**: Friend access to internal state can be useful for serializers/deserializers.
- **Performance-Critical Code**: Exposing internals via friend can avoid unnecessary accessor/mutator overhead.

### Avoid Overuse:

- Excessive use of `friend` breaks encapsulation and increases coupling.
- Prefer member functions or public/protected interfaces unless access truly cannot be otherwise managed.
- Document the rationale for `friend` usage to aid maintainability.

## 9. **Common Pitfalls**

- **Declaring before full definition**: Attempting to friend a function or class not yet fully defined leads to compilation errors.
- **Wrong friend syntax**: `friend class F;` introduces `F` if not already declared, while `friend F;` requires an existing declaration.

## 10. **Conclusion**

The `friend` keyword is a powerful tool in C++ that allows you to selectively grant access to private and protected members of a class. When used judiciously, it facilitates cleaner interfaces and better encapsulation management in complex systems. However, its misuse can lead to tightly coupled and hard-to-maintain codebases. Always consider alternatives and document your use of friendship to maintain clarity and discipline in your design.
