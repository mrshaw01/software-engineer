# weak_ptr Class Documentation

## Overview

`weak_ptr` is a smart pointer in C++ that provides a non-owning reference to an object managed by a `shared_ptr`. It is useful for breaking cyclic references that could otherwise cause memory leaks.

## Syntax

```cpp
template <class T>
class weak_ptr;
```

## Key Features

- Non-owning observer of resources managed by `shared_ptr`.
- Does not contribute to the reference count.
- Can detect if the resource it observes has expired.

## Use Cases

- Breaking cyclic references.
- Observing objects without extending their lifetime.

## Constructors

| Constructor                      | Description                                       |
| -------------------------------- | ------------------------------------------------- |
| `weak_ptr()`                     | Default constructor, creates an empty `weak_ptr`. |
| `weak_ptr(const weak_ptr&)`      | Copy constructor, observes the same resource.     |
| `weak_ptr(const shared_ptr<T>&)` | Constructs from a `shared_ptr`.                   |

## Member Functions

| Function         | Description                                                                                |
| ---------------- | ------------------------------------------------------------------------------------------ |
| `lock()`         | Creates a `shared_ptr` if the resource still exists; otherwise returns empty `shared_ptr`. |
| `expired()`      | Checks if the observed resource has been deleted.                                          |
| `reset()`        | Resets the `weak_ptr` to empty state.                                                      |
| `use_count()`    | Returns the number of `shared_ptr` instances sharing ownership of the resource.            |
| `swap()`         | Exchanges resources with another `weak_ptr`.                                               |
| `owner_before()` | Compares the order of two pointers for ordering purposes.                                  |

## Operators

| Operator         | Description                        |
| ---------------- | ---------------------------------- |
| `operator=(...)` | Assigns a new resource to observe. |

## Examples

### Basic Usage

```cpp
#include <iostream>
#include <memory>

int main() {
    std::shared_ptr<int> sp = std::make_shared<int>(10);
    std::weak_ptr<int> wp(sp);

    if (auto locked = wp.lock()) {
        std::cout << "Value: " << *locked << "\n";
    }

    sp.reset();

    if (wp.expired()) {
        std::cout << "Resource expired.\n";
    }

    return 0;
}

// Output:
// Value: 10
// Resource expired.
```

### Breaking Cycles

```cpp
#include <memory>
#include <iostream>

struct Node {
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> prev;  // weak_ptr to break the cycle
};

int main() {
    auto node1 = std::make_shared<Node>();
    auto node2 = std::make_shared<Node>();

    node1->next = node2;
    node2->prev = node1;  // No cyclic reference due to weak_ptr

    std::cout << "Node created and cycle avoided." << std::endl;

    return 0;
}
```

## Remarks

- `weak_ptr` must use `lock()` to access the underlying object safely.
- Use `expired()` to check resource validity before using `lock()`.
- Does not manage lifetime; purely observational.
