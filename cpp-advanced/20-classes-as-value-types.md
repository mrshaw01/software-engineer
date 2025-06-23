# C++ Classes as Value Types

## Overview

C++ classes are, by default, **value types**. This means they support value semantics such as copying and assignment. However, C++ also allows classes to be configured as **reference types**, enabling polymorphism and object-oriented design patterns through virtual functions and inheritance.

Understanding the distinction between value and reference types is essential for designing efficient and correct software systems in modern C++.

## Value Types vs Reference Types

### Value Types

- Copyable by default: C++ automatically provides a copy constructor and a copy assignment operator.
- Focused on **content**: Each copy is independent; changes to one do not affect the other.
- Typically used when objects represent **data**.
- Move operations (via `&&`) can optimize performance and eliminate unnecessary deep copies.

### Reference Types (Polymorphic Types)

- Identity-centric: Behavior is defined by the **type hierarchy**, not just data content.
- Require:

  - A virtual destructor.
  - Disabled copy constructor and copy assignment operator to avoid slicing.

- Enable polymorphism through inheritance and virtual functions.

## Example: Preventing Copy for Reference-like Type

```cpp
class MyRefType {
private:
    MyRefType& operator=(const MyRefType&);
    MyRefType(const MyRefType&);
public:
    MyRefType() {}
};

int main() {
    MyRefType a, b;
    a = b;  // Compilation error
}
```

**Compilation Error:**

```text
error C2248: 'MyRefType::operator=' : cannot access private member
```

This is the intended behavior for reference types: disallow copying to enforce identity semantics.

## Value Types and Move Efficiency

C++11 introduced **move semantics** to optimize value operations, avoiding costly deep copies when possible. This is particularly useful when working with:

- Containers (e.g., inserting strings into vectors)
- Temporary objects (e.g., return-by-value)
- Resource-managing classes

### Example: Efficient Insertion and Return

```cpp
#include <vector>
#include <string>
#include <set>
using namespace std;

set<widget> LoadHugeData() {
    set<widget> ret;
    // Load data...
    return ret;  // Efficient: no deep copy due to move
}

vector<string> v = IfIHadAMillionStrings();
v.insert(begin(v) + v.size() / 2, "scott");   // Efficient: move, not copy
```

### Operator Overloads with Move Support

```cpp
HugeMatrix operator+(const HugeMatrix&, const HugeMatrix&);
HugeMatrix operator+(const HugeMatrix&, HugeMatrix&&);
HugeMatrix operator+(HugeMatrix&&, const HugeMatrix&);
HugeMatrix operator+(HugeMatrix&&, HugeMatrix&&);

// Enables chained move-optimized additions
hm5 = hm1 + hm2 + hm3 + hm4 + hm5;
```

## Enabling Move Operations

To opt into move semantics, explicitly declare move constructors and move assignment operators:

```cpp
#include <memory>
#include <stdexcept>
using namespace std;

class my_class {
    unique_ptr<BigHugeData> data;

public:
    my_class(my_class&& other) noexcept
        : data(move(other.data)) {}

    my_class& operator=(my_class&& other) noexcept {
        data = move(other.data);
        return *this;
    }

    void method() {
        if (!data)
            throw runtime_error("RUNTIME ERROR: Insufficient resources!");
    }
};
```

### Guidelines

- If you enable copy construction/assignment, also enable move if it’s more efficient.
- If the object is non-copyable but transferable, make it **move-only** (like `unique_ptr`).

## Summary

| Aspect           | Value Type               | Reference Type                     |
| ---------------- | ------------------------ | ---------------------------------- |
| Default behavior | Copyable                 | Non-copyable                       |
| Semantics        | Content-based            | Identity-based                     |
| Polymorphism     | Not supported by default | Enabled via virtual functions      |
| Use case         | Data encapsulation       | Interface and behavior abstraction |
| Optimization     | Move semantics           | Disabled copying, use pointers     |

Move semantics provide performance gains without compromising readability or correctness. By default, C++ favors value semantics, but the language provides rich support for both paradigms, enabling fine-grained control over object behavior and lifetime.
