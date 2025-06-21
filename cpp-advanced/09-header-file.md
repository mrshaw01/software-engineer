# Header Files in C++

## Overview

In C++, every name (e.g., variable, function, class) must be declared before it is used. This is a requirement because each `.cpp` file is compiled independently. The compiler has no visibility into declarations from other translation units unless explicitly included. To manage and reuse declarations across files without duplication or inconsistency, C++ relies on _header files_.

Header files help maintain consistency, improve modularity, and reduce maintenance costs. This document outlines the purpose, structure, and best practices for writing and using header files in modern C++.

## Why Use Header Files?

Consider this example:

```cpp
int x;     // declaration
x = 42;    // usage
```

Without the declaration, the compiler cannot deduce the type or behavior of `x`.

When multiple source files need to share the same class, function, or variable, redundant declarations in each file can cause inconsistencies. Instead, declarations should be centralized in a header file and pulled into each translation unit using `#include`.

```cpp
#include "my_class.h"
```

The `#include` directive inserts the content of the header file during preprocessing. This ensures consistent declarations across compilation units.

## Example: Class Declaration and Usage

### Header File (`my_class.h`)

```cpp
#ifndef MY_CLASS_H
#define MY_CLASS_H

namespace N {
    class my_class {
    public:
        void do_something(); // Declaration only
    };
}

#endif // MY_CLASS_H
```

### Implementation File (`my_class.cpp`)

```cpp
#include "my_class.h"
#include <iostream>

using namespace N;

void my_class::do_something() {
    std::cout << "Doing something!" << std::endl;
}
```

### Usage (`main.cpp`)

```cpp
#include "my_class.h"

using namespace N;

int main() {
    my_class mc;
    mc.do_something();
    return 0;
}
```

## Compilation and Linking

Each `.cpp` file is compiled into an object file independently. The linker then merges all object files. The class `my_class` is defined only once in `my_class.cpp`, and all other `.cpp` files refer to it via the header file.

## Include Guards

To avoid multiple inclusion errors, header files should use **include guards** or `#pragma once`.

### With include guards:

```cpp
#ifndef MY_HEADER_H
#define MY_HEADER_H
// ... declarations
#endif // MY_HEADER_H
```

### Or use:

```cpp
#pragma once
```

## What to Put in a Header File

Header files should contain _declarations_, not _definitions_. Avoid anything that could lead to multiple definition errors.

**Allowed:**

- Class, struct, and enum declarations
- `inline` and `constexpr` function definitions
- `extern` variable declarations
- Type aliases (`using`, `typedef`)
- Template declarations and definitions
- Macros and conditional compilation

**Avoid:**

- Non-`inline` function definitions
- Non-`const` global variables
- `using namespace` directives (especially in global scope)
- Unnamed namespaces

## Sample Header File

```cpp
#pragma once
#include <vector>
#include <string>

namespace N {

    inline namespace P {}

    enum class colors : short { red, blue, purple, azure };

    constexpr int MeaningOfLife = 42;
    constexpr int get_meaning() {
        static_assert(MeaningOfLife == 42, "unexpected!");
        return MeaningOfLife;
    }

    using vstr = std::vector<int>;
    extern double d;

#define LOG

#ifdef LOG
    void print_to_log();
#endif

    class my_class {
        friend class other_class;
    public:
        void do_something();
        inline void put_value(int i) { vals.push_back(i); }
    private:
        vstr vals;
        int i;
    };

    struct RGB {
        short r{0}, g{0}, b{0};
    };

    template <typename T>
    class value_store {
    public:
        value_store<T>() = default;
        void write_value(T val) {
            // ...
        }
    private:
        std::vector<T> vals;
    };

    template <typename T>
    class value_widget; // Forward declaration

}
```

## Summary

Header files are a core mechanism in C++ for modular and maintainable code. By properly using header files:

- You ensure declarations are consistent across translation units.
- You reduce duplication and errors at link time.
- You adhere to best practices for performance, readability, and maintainability.

Transition to C++20 modules is encouraged for large-scale projects, but headers remain fundamental and widely supported.
