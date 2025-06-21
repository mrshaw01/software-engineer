# unique_ptr Class Documentation

## Overview

`unique_ptr` is a smart pointer class in C++ that manages the lifetime of an object or an array, ensuring automatic deletion of the managed resource when the `unique_ptr` goes out of scope. It supersedes the deprecated `auto_ptr`, providing robust ownership semantics.

## Syntax

```cpp
template <class T, class Deleter = std::default_delete<T>>
class unique_ptr;

// Specialization for arrays
template <class T, class Deleter>
class unique_ptr<T[], Deleter>;
```

## Key Features

- Ensures unique ownership (non-copyable, only movable).
- Automatically deletes the owned resource upon destruction.
- Custom deleters can be specified for flexible resource management.
- Provides specialization for array management.

## Constructors

| Constructor                          | Description                                         |
| ------------------------------------ | --------------------------------------------------- |
| `unique_ptr()`                       | Default constructor, initializes with `nullptr`.    |
| `unique_ptr(nullptr_t)`              | Explicitly initializes with `nullptr`.              |
| `explicit unique_ptr(pointer ptr)`   | Takes ownership of `ptr`.                           |
| `unique_ptr(pointer ptr, Deleter d)` | Takes ownership of `ptr` with a custom deleter `d`. |
| `unique_ptr(unique_ptr&& other)`     | Move constructor, transfers ownership.              |
| Deleted copy constructor             | Prevents copying to enforce unique ownership.       |

## Member Functions

| Function            | Description                                    |
| ------------------- | ---------------------------------------------- |
| `get()`             | Returns the stored pointer.                    |
| `get_deleter()`     | Accesses the stored deleter object.            |
| `release()`         | Releases ownership and returns pointer.        |
| `reset(pointer)`    | Replaces managed object with a new pointer.    |
| `swap(unique_ptr&)` | Exchanges resources with another `unique_ptr`. |

## Operators

| Operator                  | Description                                   |
| ------------------------- | --------------------------------------------- |
| `operator bool()`         | Checks if the `unique_ptr` manages an object. |
| `operator*()`             | Dereferences to access the managed object.    |
| `operator->()`            | Accesses members of the managed object.       |
| `operator=(unique_ptr&&)` | Move assignment, transfers ownership.         |

## Specialization for Arrays

Specialized syntax and functions for managing arrays:

```cpp
template <class T, class Deleter>
class unique_ptr<T[], Deleter>;
```

### Array-specific Members

| Function             | Description            |
| -------------------- | ---------------------- |
| `operator[](size_t)` | Access array elements. |

## Remarks

- A `unique_ptr` object manages a single resource exclusively.
- Use `std::make_unique` for efficient and safe allocation.
- Default deleter (`default_delete`) uses `delete` or `delete[]` appropriately.

## Example

```cpp
#include <iostream>
#include <memory>

struct Sample {
   int content;
   Sample(int c) : content(c) { std::cout << "Constructing " << content << '\n'; }
   ~Sample() { std::cout << "Deleting " << content << '\n'; }
};

int main() {
    auto up1 = std::make_unique<Sample>(10);
    auto rawPtr = up1.release(); // Transfers ownership, up1 becomes empty
    delete rawPtr;               // Manually delete resource
}

// Output:
// Constructing 10
// Deleting 10
```
