# Memory: Stack vs Heap

## Overview

In languages like C and C++, memory allocation occurs either on the **stack** or the **heap**. Understanding their differences is essential for efficient and robust software development.

## The Stack

The stack is a contiguous memory region automatically managed in a Last-In-First-Out (LIFO) manner.

### Characteristics

- **Automatic Management:** Allocation and deallocation handled automatically upon function calls and exits.
- **Fast Access:** Efficient CPU utilization due to contiguous allocation.
- **Limited Size:** Fixed limit based on system and compiler settings.
- **Local Scope:** Variables only exist during the execution of their creating function.
- **Thread Safety:** Each thread maintains its own stack.

### Example

```c
int main() {
    int count = 10;
    double scores[5]; // stack-allocated array
    return 0;
}
```

## The Heap

The heap provides dynamically allocated memory with manual management.

### Characteristics

- **Manual Management:** Explicit allocation (`malloc`, `calloc`, `realloc`) and deallocation (`free`).
- **Dynamic Size:** Memory allocation can be adjusted at runtime.
- **Global Access:** Accessible across different functions using pointers.
- **Fragmentation:** Potential for memory fragmentation over time.
- **Slower Access:** Requires indirection via pointers.
- **Less Thread-safe:** Shared across threads unless explicitly managed.

### Example

```c
#include <stdlib.h>

int main() {
    int *data = malloc(20 * sizeof(int));
    // use data
    free(data);
    return 0;
}
```

## Stack vs Heap Comparison

| Feature       | Stack                      | Heap                           |
| ------------- | -------------------------- | ------------------------------ |
| Allocation    | Automatic (compiler)       | Manual (`malloc`, `free`)      |
| Access Speed  | Fast                       | Slower                         |
| Scope         | Local (function only)      | Global (via pointers)          |
| Memory Size   | Limited (system-dependent) | Large (system memory limit)    |
| Resizable     | No                         | Yes (`realloc`)                |
| Thread Safety | Yes                        | No (unless managed explicitly) |
| Fragmentation | None                       | Possible                       |

## Usage Recommendations

### Use Stack when:

- Variables are small, temporary, and have local function scope.
- Performance is critical.
- Automatic memory management is beneficial.

### Use Heap when:

- Large data structures or dynamic resizing is required.
- Variables must persist beyond a single function.
- Complex data structures (trees, linked lists, dynamic arrays) are needed.
