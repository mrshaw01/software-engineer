## Resources and Smart Pointers

One common issue in C-style programming is **memory leaks**, often caused by forgetting to release memory allocated with `new`. Modern C++ advocates using the principle of **Resource Acquisition Is Initialization (RAII)**. According to RAII, resources such as heap memory, file handles, and network sockets should be managed by objects. The object acquires the resource upon construction and releases it upon destruction, ensuring automatic cleanup when the object goes out of scope.

### Smart Pointers in the Standard Library

To simplify RAII adoption, C++ provides three smart pointers:

- `std::unique_ptr`
- `std::shared_ptr`
- `std::weak_ptr`

These smart pointers automate resource management by handling memory allocation and deallocation.

### Examples

#### Example 1: Using `std::unique_ptr`

```cpp
#include <memory>
#include <iostream>

class Widget {
private:
    std::unique_ptr<int[]> data;
public:
    Widget(size_t size) : data(std::make_unique<int[]>(size)) {
        std::cout << "Widget created with size " << size << std::endl;
    }
    ~Widget() {
        std::cout << "Widget destroyed." << std::endl;
    }
    void do_something() {
        std::cout << "Doing something..." << std::endl;
    }
};

void functionUsingWidget() {
    Widget w(1000);  // Automatically managed lifetime
    w.do_something();
} // Widget destructor is automatically called here

int main() {
    functionUsingWidget();
    return 0;
}

// Output:
// Widget created with size 1000
// Doing something...
// Widget destroyed.
```

#### Example 2: Using `std::shared_ptr` and `std::weak_ptr` to Avoid Cycles

```cpp
#include <memory>
#include <iostream>

struct Node {
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> prev; // weak_ptr breaks potential cycle

    ~Node() { std::cout << "Node destroyed." << std::endl; }
};

int main() {
    auto node1 = std::make_shared<Node>();
    auto node2 = std::make_shared<Node>();

    node1->next = node2;
    node2->prev = node1; // No cycle due to weak_ptr usage

    std::cout << "Nodes created." << std::endl;

    return 0;

} // Both nodes automatically destroyed here

// Output:
// Nodes created.
// Node destroyed.
// Node destroyed.
```

### Best Practices

- Always prefer smart pointers (`unique_ptr`, `shared_ptr`, `weak_ptr`) over raw pointers.
- When explicit `new` and `delete` must be used, adhere strictly to RAII principles to avoid memory leaks and dangling pointers.
