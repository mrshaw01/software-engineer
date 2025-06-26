### Pimpl Idiom for Compile-Time Encapsulation in Modern C++

The **Pimpl idiom** (Pointer to IMPLementation) is a robust C++ design technique to decouple a class's interface from its implementation. It is particularly effective in large systems where compilation times, ABI stability, and encapsulation are major concerns. Known also as the _Cheshire Cat_ or _Compiler Firewall_, it achieves _compile-time encapsulation_ by hiding implementation details from header files.

## Goals and Motivation

The primary reasons for employing the Pimpl idiom in modern C++ are:

1. **Reduce Compilation Dependencies**
   By moving private members and complex types into a separately compiled `.cpp` file, Pimpl reduces the ripple effect of header changes on dependent compilation units.

2. **Encapsulation and ABI Stability**
   The public interface remains stable even if internal representations change, making this idiom ideal for shared libraries or evolving APIs.

3. **Faster Build Times**
   Since only the source file where the implementation resides must be recompiled when implementation changes, Pimpl reduces build overhead.

4. **Binary Compatibility**
   Modifications to the implementation do not affect client code, helping with versioning and library distribution.

## Core Structure of the Pimpl Idiom

### Header File (`my_class.h`)

```cpp
#pragma once
#include <memory>

class my_class {
public:
    my_class();
    ~my_class();
    my_class(my_class&&) noexcept;
    my_class& operator=(my_class&&) noexcept;

    void do_something() const;

private:
    class impl;
    std::unique_ptr<impl> pimpl;
};
```

- **`impl` is an incomplete type** at this point.
- **`std::unique_ptr<impl>`** is used for ownership with efficient heap allocation.
- Use of `noexcept` ensures compatibility with standard containers and optimizations.

### Source File (`my_class.cpp`)

```cpp
#include "my_class.h"
#include <iostream>

class my_class::impl {
public:
    void do_something() const {
        std::cout << "Doing something inside impl\n";
    }

    // Add private data and helper methods here.
};

my_class::my_class() : pimpl(std::make_unique<impl>()) {}

my_class::~my_class() = default;

my_class::my_class(my_class&&) noexcept = default;
my_class& my_class::operator=(my_class&&) noexcept = default;

void my_class::do_something() const {
    pimpl->do_something();
}
```

- All implementation details (data, algorithms, dependencies) are _fully hidden_ in the `.cpp` file.
- The user only sees the clean, stable interface in the header.

## Best Practices and Modern Enhancements

1. **Rule of Five**

   - Implement or default the move constructor and move assignment for resource-safe, efficient transfer.
   - Avoid copy semantics unless you explicitly support deep copies (which is costly and complex with `unique_ptr`).

2. **Non-Throwing Swap**

   - Consider a custom `swap()` function to support exception-safe swap idioms.

   ```cpp
   friend void swap(my_class& a, my_class& b) noexcept {
       using std::swap;
       swap(a.pimpl, b.pimpl);
   }
   ```

3. **Const-Correctness**

   - Declare `impl` methods as `const` where applicable to enable `const` methods on the public interface.

4. **Avoid Over-Abstraction**

   - Don’t use Pimpl for everything; it’s most useful when:

     - Implementation involves heavyweight headers
     - Binary interface stability is needed
     - You have long-lived libraries or modules

5. **Performance Considerations**

   - There's a slight **runtime indirection** cost due to heap allocation and pointer dereferencing. For performance-critical inner loops or small trivial types, this may be prohibitive.

## Output Example

```cpp
my_class obj;
obj.do_something();
```

**Output:**

```
Doing something inside impl
```

## Summary

| Aspect                  | Pimpl Idiom Benefit                                          |
| ----------------------- | ------------------------------------------------------------ |
| Compile-time dependency | Eliminated by opaque pointer                                 |
| Encapsulation           | Implementation details are fully hidden from consumers       |
| ABI Stability           | Changes in private data do not affect the public ABI         |
| Build performance       | Reduces unnecessary recompilation                            |
| Copy/Move semantics     | Move is preferred; copy is complex unless explicitly managed |

The Pimpl idiom is a powerful abstraction mechanism in modern C++ for large-scale, maintainable software. Use it selectively, balancing abstraction cost with the need for flexibility and build isolation.
