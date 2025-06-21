# Object Lifetime and Resource Management (RAII)

## Overview

Resource Acquisition Is Initialization (RAII) is a key programming technique in C++ that binds the lifecycle of resources (heap memory, file handles, sockets, mutexes, etc.) directly to object lifetimes. This approach ensures deterministic and safe resource management.

## Core Concepts of RAII

RAII encapsulates resources within objects whose lifetimes are tied to the scope in which they're declared. The technique ensures that resources:

- Are acquired during object initialization (construction).
- Are automatically released during object cleanup (destruction).

This approach eliminates common issues such as resource leaks and dangling pointers.

## Key Benefits of RAII

- **Deterministic resource management**: Resources are released at predictable points.
- **Exception safety**: Resources are reliably cleaned up even when exceptions occur.
- **No manual cleanup**: Developers don't explicitly call resource release methods.

## Examples

### Basic RAII Example (No Dynamic Resource)

```cpp
class Widget {
private:
    Gadget g;  // automatic lifetime tied to Widget
public:
    void draw();
};

void functionUsingWidget() {
    Widget w;   // automatic construction
    w.draw();
} // automatic destruction of w and w.g
```

### RAII with Dynamic Memory

Without RAII:

```cpp
class Widget {
private:
    int* data;
public:
    Widget(int size) { data = new int[size]; }
    ~Widget() { delete[] data; }
};
```

With RAII (using smart pointers):

```cpp
#include <memory>

class Widget {
private:
    std::unique_ptr<int[]> data;
public:
    Widget(int size) : data(std::make_unique<int[]>(size)) {}
};
```

## RAII in Standard Library

C++ standard library extensively uses RAII principles in classes such as:

- Containers (`std::vector`, `std::string`, etc.)
- Smart pointers (`std::unique_ptr`, `std::shared_ptr`)
- Synchronization primitives (`std::lock_guard`, `std::unique_lock`)

### Mutex Management Example

```cpp
#include <mutex>

std::mutex m;

void good() {
    std::lock_guard<std::mutex> lock(m); // mutex acquired here
    f();                                // mutex released automatically
}
```

## Limitations

RAII does **not** manage resources not acquirable at initialization, such as CPU time, network bandwidth, or stack memory.

## Best Practices

- Always encapsulate resources within RAII-compliant classes.
- Prefer standard library RAII wrappers over custom implementations.
- Ensure constructors establish resource invariants or fail clearly via exceptions.
