# shared_ptr Class Documentation

## Overview

`shared_ptr` is a smart pointer class in C++ that manages resources through reference counting, allowing multiple pointers to share ownership of a dynamically allocated object. The resource is automatically deleted when the last `shared_ptr` pointing to it is destroyed.

## Syntax

```cpp
template <class T>
class shared_ptr;
```

## Key Features

- Manages shared ownership of resources via reference counting.
- Automatically releases resources when reference count reaches zero.
- Thread-safe for concurrent reads and writes to separate instances.
- Custom deleters and allocators can be provided.

## Constructors

| Constructor                           | Description                              |
| ------------------------------------- | ---------------------------------------- |
| `shared_ptr()`                        | Constructs an empty `shared_ptr`.        |
| `shared_ptr(nullptr_t)`               | Constructs an empty `shared_ptr`.        |
| `explicit shared_ptr(pointer ptr)`    | Constructs and takes ownership of `ptr`. |
| `shared_ptr(pointer ptr, Deleter d)`  | Constructs with custom deleter `d`.      |
| `shared_ptr(shared_ptr&& other)`      | Move constructor, transfers ownership.   |
| `shared_ptr(const shared_ptr& other)` | Copy constructor, shares ownership.      |

## Member Functions

| Function            | Description                                |
| ------------------- | ------------------------------------------ |
| `get()`             | Returns the pointer to the managed object. |
| `use_count()`       | Returns the number of shared owners.       |
| `reset(pointer)`    | Replaces the managed resource.             |
| `swap(shared_ptr&)` | Swaps resources with another `shared_ptr`. |

## Operators

| Operator                       | Description                                 |
| ------------------------------ | ------------------------------------------- |
| `operator bool()`              | Checks if the `shared_ptr` owns a resource. |
| `operator*()`                  | Dereferences to the managed object.         |
| `operator->()`                 | Accesses members of the managed object.     |
| `operator=(shared_ptr&&)`      | Move assignment.                            |
| `operator=(const shared_ptr&)` | Copy assignment.                            |

## Thread Safety

- Concurrent access and modification of separate `shared_ptr` instances are thread-safe.

## Examples

### Basic Usage

```cpp
#include <iostream>
#include <memory>

int main() {
    auto sp1 = std::make_shared<int>(10);
    std::shared_ptr<int> sp2 = sp1;

    std::cout << "Value: " << *sp1 << ", Use count: " << sp1.use_count() << std::endl;
}

// Output:
// Value: 10, Use count: 2
```

### Custom Deleter

```cpp
#include <iostream>
#include <memory>

void customDelete(int* p) {
    std::cout << "Deleting resource." << std::endl;
    delete p;
}

int main() {
    std::shared_ptr<int> sp(new int(5), customDelete);
    std::cout << *sp << std::endl;
}

// Output:
// 5
// Deleting resource.
```

## Remarks

- `shared_ptr` maintains a control block that stores reference counts and optional custom deleters.
- When `use_count()` reaches zero, the resource and control block are released.
- Prefer `std::make_shared` for efficient memory allocation.
