# How to create and use `weak_ptr` instances

In modern C++, `std::shared_ptr` is used to represent shared ownership of dynamically allocated objects. However, in scenarios involving **cyclic references**, `shared_ptr` can create **memory leaks** by keeping objects alive unintentionally. This is where `std::weak_ptr` becomes essential—it breaks the cycle by offering **non-owning** access to an object managed by `shared_ptr`, without contributing to the reference count.

## **1. What Is `std::weak_ptr`?**

A `weak_ptr<T>` is a smart pointer that:

- Holds a reference to a `shared_ptr<T>`-managed object.
- **Does not affect the reference count**.
- Allows one to **check** whether the managed object still exists via `expired()` or `lock()`.

> Best practice: use `weak_ptr` when an object needs to observe or access another object without extending its lifetime.

## **2. Use Case: Breaking Cyclic Dependencies**

Suppose you have two or more objects that **mutually reference each other** using `shared_ptr`. This leads to **cyclic ownership**, which prevents their destructors from being called, even when no external reference exists.

### Example Setup: Controllers with Mutual Awareness

```cpp
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

class Controller {
public:
    int Num;
    std::wstring Status;
    std::vector<std::weak_ptr<Controller>> others;

    explicit Controller(int i) : Num(i), Status(L"On") {
        std::wcout << L"Creating Controller" << Num << std::endl;
    }

    ~Controller() {
        std::wcout << L"Destroying Controller" << Num << std::endl;
    }

    void CheckStatuses() const {
        for (const auto& wp : others) {
            if (auto sp = wp.lock()) {
                std::wcout << L"Status of " << sp->Num << L" = " << sp->Status << std::endl;
            } else {
                std::wcout << L"Null object" << std::endl;
            }
        }
    }
};
```

## **3. Building the Graph with `weak_ptr`**

```cpp
void RunTest() {
    std::vector<std::shared_ptr<Controller>> v;
    for (int i = 0; i < 5; ++i)
        v.emplace_back(std::make_shared<Controller>(i));

    // Setup weak_ptr references to all other controllers
    for (int i = 0; i < v.size(); ++i) {
        for (const auto& p : v) {
            if (p->Num != i) {
                v[i]->others.push_back(std::weak_ptr<Controller>(p));
                std::wcout << L"push_back to v[" << i << L"]: " << p->Num << std::endl;
            }
        }
    }

    for (auto& p : v) {
        std::wcout << L"use_count = " << p.use_count() << std::endl;
        p->CheckStatuses();
    }
}
```

### `main()`

```cpp
int main() {
    RunTest();
    std::wcout << L"Press any key" << std::endl;
    std::cin.get();
}
```

### Output (Truncated)

```
Creating Controller0
Creating Controller1
...
use_count = 1
Status of 1 = On
...
Destroying Controller0
Destroying Controller1
...
```

> Each `Controller` is properly destroyed when `RunTest()` exits—thanks to using `weak_ptr` to avoid cyclic references.

## **4. Experiment: Replacing `weak_ptr` with `shared_ptr`**

If you change:

```cpp
std::vector<std::weak_ptr<Controller>> others;
```

to:

```cpp
std::vector<std::shared_ptr<Controller>> others;
```

Each `Controller` will **own references to all other controllers**, forming **strong reference cycles**. This results in **none of the destructors being called**, as the reference count for each controller **never drops to zero**.

### Consequence:

```txt
Creating Controller0
Creating Controller1
...
(no destructors invoked)
```

This confirms that cyclic `shared_ptr` usage without `weak_ptr` leads to memory leaks.

## **5. Best Practices for `weak_ptr`**

- Use `weak_ptr` only when observing shared objects without participating in their ownership.
- Call `.lock()` to promote a `weak_ptr` to a `shared_ptr` (if still valid).
- Always check `.expired()` or the result of `.lock()` before use.
- Prefer **composition** over **cyclic dependencies**. But if cycles are necessary, mitigate them with `weak_ptr`.

## **6. Summary**

| Pointer Type | Owns Object | Increases Ref Count | Prevents Destruction |
| ------------ | ----------- | ------------------- | -------------------- |
| `shared_ptr` | Yes         | Yes                 | Yes                  |
| `weak_ptr`   | No          | No                  | No                   |

Using `weak_ptr` allows you to safely reference shared resources **without preventing their cleanup**, thus maintaining memory correctness and avoiding leaks due to cyclic graphs. It is an essential tool in any advanced C++ resource management strategy.
