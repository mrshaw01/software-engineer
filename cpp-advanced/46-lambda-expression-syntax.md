### Lambda Expression Syntax in C++

Understanding the syntax and semantics of lambda expressions is essential to leveraging modern C++ effectively. Lambda expressions unify the flexibility of function objects with the conciseness of inline function definitions, streamlining code especially in algorithmic contexts and asynchronous workflows.

## **1. Motivation: Function Objects vs. Lambdas**

In traditional C++, operations like custom sorting or filtering required either:

- **Function pointers** – concise, but stateless.
- **Function objects (functors)** – stateful, but syntactically verbose.

**Lambdas**, introduced in C++11 and enhanced in later standards, combine the best of both:

- They can **retain state** (like functors),
- And offer a **compact, readable syntax** (like function pointers),
- Without needing a named class or separate function.

## **2. Lambda Expression Syntax Overview**

```cpp
[capture](parameters) -> return_type {
    // lambda body
}
```

| Component             | Description                                                       |
| --------------------- | ----------------------------------------------------------------- |
| `[capture]`           | Captures variables from the enclosing scope by value or reference |
| `(parameters)`        | Optional parameter list, like a regular function                  |
| `mutable` (optional)  | Allows mutation of captured-by-value variables inside the lambda  |
| `noexcept` (optional) | Specifies the lambda does not throw exceptions                    |
| `-> return_type`      | Optional; required if return type can't be deduced                |
| `{ ... }`             | The lambda body; can access captured variables and parameters     |

## **3. Example: Using a Lambda with `std::for_each`**

```cpp
#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<int> v;
    for (int i = 1; i < 10; ++i) v.push_back(i);

    int evenCount = 0;
    for_each(v.begin(), v.end(), [&evenCount](int n) {
        cout << n;
        if (n % 2 == 0) {
            cout << " is even" << endl;
            ++evenCount;
        } else {
            cout << " is odd" << endl;
        }
    });

    cout << "There are " << evenCount << " even numbers in the vector." << endl;
}
```

### **Output**

```
1 is odd
2 is even
3 is odd
4 is even
5 is odd
6 is even
7 is odd
8 is even
9 is odd
There are 4 even numbers in the vector.
```

### **Explanation**

- `&evenCount`: captured by reference to allow mutation inside the lambda.
- `(int n)`: parameter list.
- Body contains logic to identify and count even numbers.

## **4. Equivalent: Using a Function Object**

```cpp
#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

class FunctorClass {
public:
    explicit FunctorClass(int& evenCount) : m_evenCount(evenCount) {}

    void operator()(int n) const {
        cout << n;
        if (n % 2 == 0) {
            cout << " is even" << endl;
            ++m_evenCount;
        } else {
            cout << " is odd" << endl;
        }
    }

private:
    FunctorClass& operator=(const FunctorClass&) = delete;
    mutable int& m_evenCount;
};

int main() {
    vector<int> v;
    for (int i = 1; i < 10; ++i) v.push_back(i);

    int evenCount = 0;
    for_each(v.begin(), v.end(), FunctorClass(evenCount));

    cout << "There are " << evenCount << " even numbers in the vector." << endl;
}
```

### **Output**

Same as lambda example.

### **Key Takeaways**

- Functor is verbose and requires boilerplate.
- Lambda improves code locality and readability.
- Use functors for **reusable**, **extensible** components.
- Use lambdas for **one-off**, **inline** logic.

## **5. Best Practices and Advanced Notes**

### **When to Use Lambdas**

- When logic is small, inline, and doesn’t justify a named class.
- When passing callbacks to algorithms (`std::sort`, `std::find_if`, etc.).
- When capturing scope-local variables is more readable than parameter passing.

### **Capture Strategies**

- `[=]`: capture everything by value.
- `[&]`: capture everything by reference.
- `[=, &x]`: capture most by value, `x` by reference.
- C++14 and later: `[value = std::move(ptr)]` enables move-capture.

### **Constness**

- By default, lambdas are `const` unless marked `mutable`.

### **Return Types**

- Explicit return types are useful when using conditional returns or braced-init lists:

```cpp
auto f = [](bool b) -> int {
    if (b) return 1;
    else return 2;
};
```

## **Conclusion**

Lambda expressions are a cornerstone of modern C++, striking a balance between power and brevity. They encapsulate behavior with minimal overhead and integrate seamlessly with STL algorithms and concurrency constructs. Mastering their syntax and capture semantics enables cleaner, more expressive code that remains both performant and maintainable.
