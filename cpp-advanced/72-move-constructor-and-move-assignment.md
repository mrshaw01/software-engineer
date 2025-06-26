# Move Constructors and Move Assignment Operators

Understanding and applying **move semantics** is essential for writing high-performance, resource-efficient C++ applications. Move constructors and move assignment operators enable the **transfer of ownership** of dynamically allocated or expensive-to-copy resources from temporary (rvalue) objects to new or existing objects, eliminating unnecessary copies.

## 1. **Why Move Semantics?**

Before C++11, object transfer always involved deep copying via copy constructors or assignment operators, which could be costly. With C++11, **move semantics** allow us to "steal" resources from rvalues, drastically improving performance by avoiding heap allocations and memory copies.

## 2. **Move Constructor**

### Purpose:

Constructs an object by **taking ownership** of the resources from an rvalue object.

### Signature:

```cpp
MemoryBlock(MemoryBlock&& other) noexcept;
```

### Implementation:

```cpp
MemoryBlock(MemoryBlock&& other) noexcept
    : _data(nullptr), _length(0)  // Initialize members
{
    std::cout << "In MemoryBlock(MemoryBlock&&). length = "
              << other._length << ". Moving resource." << std::endl;

    _data = other._data;
    _length = other._length;

    other._data = nullptr;
    other._length = 0;
}
```

### Explanation:

- Transfers `_data` and `_length` from `other`.
- Sets `other._data` to `nullptr` and `other._length` to `0` to prevent double deletion.

## 3. **Move Assignment Operator**

### Purpose:

Assigns an rvalue object to an existing object, again by transferring ownership of resources.

### Signature:

```cpp
MemoryBlock& operator=(MemoryBlock&& other) noexcept;
```

### Implementation:

```cpp
MemoryBlock& operator=(MemoryBlock&& other) noexcept
{
    std::cout << "In operator=(MemoryBlock&&). length = "
              << other._length << "." << std::endl;

    if (this != &other)
    {
        delete[] _data; // Free current resource

        _data = other._data;
        _length = other._length;

        other._data = nullptr;
        other._length = 0;
    }
    return *this;
}
```

### Best Practices:

- **Self-assignment check**: `if (this != &other)` is crucial even in move semantics.
- **Resource safety**: Always release current resources before acquiring new ones.
- **noexcept**: Marking move operations `noexcept` enables optimizations (e.g., for `std::vector`).

## 4. **Example Usage with `std::vector`**

```cpp
#include "MemoryBlock.h"
#include <vector>

int main()
{
    std::vector<MemoryBlock> v;
    v.push_back(MemoryBlock(25));
    v.push_back(MemoryBlock(75));
    v.insert(v.begin() + 1, MemoryBlock(50));
}
```

### Output (Move Semantics Enabled):

```
In MemoryBlock(size_t). length = 25.
In MemoryBlock(MemoryBlock&&). length = 25. Moving resource.
In ~MemoryBlock(). length = 0.
...
In ~MemoryBlock(). length = 25. Deleting resource.
```

Here, instead of deep copies (as would happen with only copy constructors), the vector uses move operations for efficient reallocation and insertion.

## 5. **Optional Optimization: Delegating Move Constructor**

To avoid duplicating code between the move constructor and move assignment operator:

```cpp
MemoryBlock(MemoryBlock&& other) noexcept
    : _data(nullptr), _length(0)
{
    *this = std::move(other); // Delegates to move assignment
}
```

This ensures consistency and reduces maintenance effort. However, be aware that **assigning to an uninitialized object** is unconventional, and this pattern may not be suitable for all types, especially polymorphic ones.

## 6. **When to Implement Move Semantics Manually**

Implement move operations when:

- Your class manages **dynamic resources** (memory, files, sockets).
- **Performance** is critical (e.g., large data structures).
- You want to enable **efficient use in STL containers** (e.g., `std::vector`, `std::map`).

Avoid manual implementation when:

- The default move constructor/operator suffices (C++11 onwards).
- You rely only on types that are themselves moveable and trivially constructible.

## 7. **Rule of Five**

If you implement any of:

- Destructor
- Copy constructor
- Copy assignment
- Move constructor
- Move assignment

Then you likely need to implement all five to manage the resource lifecycle explicitly.

## 8. **Summary Table**

| Operation        | When Called                       | Key Behavior       |
| ---------------- | --------------------------------- | ------------------ |
| Copy Constructor | `MemoryBlock b2 = b1;`            | Deep copy          |
| Move Constructor | `MemoryBlock b2 = std::move(b1);` | Transfer ownership |
| Copy Assignment  | `b2 = b1;`                        | Deep copy          |
| Move Assignment  | `b2 = std::move(b1);`             | Transfer ownership |

## 9. **Conclusion**

Move constructors and move assignment operators are cornerstones of modern C++ performance optimization. They allow developers to build resource-efficient, safe, and fast systems without incurring unnecessary copies. When designing resource-managing classes, always prefer move semantics when possible, and ensure correctness through proper reset of the moved-from state and exception safety guarantees (`noexcept`).

If your class follows RAII principles and you write correct move logic, your objects will behave predictably and integrate seamlessly with the STL and other modern C++ libraries.
