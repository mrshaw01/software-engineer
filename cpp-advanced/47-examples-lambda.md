# **Examples of Lambda Expressions in Modern C++**

## **1. Declaring Lambda Expressions**

### **Example 1: Assignment to `auto` and `std::function`**

```cpp
#include <functional>
#include <iostream>

int main() {
    auto f1 = [](int x, int y) { return x + y; };
    std::cout << f1(2, 3) << std::endl;

    std::function<int(int, int)> f2 = [](int x, int y) { return x + y; };
    std::cout << f2(3, 4) << std::endl;
}
```

**Output:**

```
5
7
```

**Explanation:**

- `f1` uses type deduction via `auto`, enabling lightweight usage.
- `f2` uses `std::function` for type-erased storage and dispatch. Useful when type uniformity or polymorphism is needed.

### **Example 2: Value vs. Reference Capture**

```cpp
#include <functional>
#include <iostream>

int main() {
    int i = 3;
    int j = 5;

    std::function<int()> f = [i, &j] { return i + j; };

    i = 22;
    j = 44;

    std::cout << f() << std::endl;
}
```

**Output:**

```
47
```

**Explanation:**

- `i` is captured **by value**, so `f` uses `i = 3`.
- `j` is captured **by reference**, so `f` uses updated `j = 44`.

## **2. Calling Lambda Expressions**

### **Example 1: Immediate Invocation**

```cpp
int result = [](int x, int y) { return x + y; }(5, 4);
std::cout << result << std::endl;
```

**Output:**

```
9
```

**Use case:** Quick, one-off evaluations.

### **Example 2: Lambda in Algorithm**

```cpp
#include <list>
#include <algorithm>
#include <iostream>

int main() {
    std::list<int> numbers = {13, 17, 42, 46, 99};

    auto it = std::find_if(numbers.begin(), numbers.end(), [](int n) {
        return n % 2 == 0;
    });

    if (it != numbers.end()) {
        std::cout << "The first even number in the list is " << *it << "." << std::endl;
    } else {
        std::cout << "The list contains no even numbers." << std::endl;
    }
}
```

**Output:**

```
The first even number in the list is 42.
```

## **3. Nesting Lambda Expressions**

```cpp
int result = [](int x) {
    return [](int y) { return y * 2; }(x) + 3;
}(5);

std::cout << result << std::endl;
```

**Output:**

```
13
```

**Explanation:**

- The inner lambda multiplies by 2 → `5 * 2 = 10`
- Outer lambda adds 3 → `10 + 3 = 13`

## **4. Higher-Order Lambda Functions**

```cpp
#include <functional>
#include <iostream>

int main() {
    auto addtwointegers = [](int x) -> std::function<int(int)> {
        return [=](int y) { return x + y; };
    };

    auto higherorder = [](const std::function<int(int)>& f, int z) {
        return f(z) * 2;
    };

    int result = higherorder(addtwointegers(7), 8);
    std::cout << result << std::endl;
}
```

**Output:**

```
30
```

**Explanation:**

- `addtwointegers(7)` returns a closure that adds 7.
- `higherorder` applies it to 8 → `7 + 8 = 15`, then multiplies by 2.

## **5. Using Lambdas in Class Member Functions**

```cpp
#include <vector>
#include <algorithm>
#include <iostream>

class Scale {
public:
    explicit Scale(int scale) : _scale(scale) {}

    void ApplyScale(const std::vector<int>& v) const {
        std::for_each(v.begin(), v.end(), [=](int n) {
            std::cout << n * _scale << std::endl;
        });
    }

private:
    int _scale;
};

int main() {
    std::vector<int> values = {1, 2, 3, 4};
    Scale s(3);
    s.ApplyScale(values);
}
```

**Output:**

```
3
6
9
12
```

**Explanation:**
Capturing `this` (implicitly via `[=]`) allows access to `_scale` inside the lambda.

## **6. Using Lambdas with Templates**

```cpp
#include <vector>
#include <algorithm>
#include <iostream>

template <typename T>
void negate_all(std::vector<T>& v) {
    std::for_each(v.begin(), v.end(), [](T& n) { n = -n; });
}

template <typename T>
void print_all(const std::vector<T>& v) {
    std::for_each(v.begin(), v.end(), [](const T& n) { std::cout << n << std::endl; });
}

int main() {
    std::vector<int> v = {34, -43, 56};

    print_all(v);
    negate_all(v);
    std::cout << "After negate_all():" << std::endl;
    print_all(v);
}
```

**Output:**

```
34
-43
56
After negate_all():
-34
43
-56
```

## **7. Exception Handling in Lambdas**

```cpp
#include <vector>
#include <algorithm>
#include <iostream>

int main() {
    std::vector<int> elements(3);
    std::vector<int> indices = {0, -1, 2}; // -1 is invalid

    try {
        std::for_each(indices.begin(), indices.end(), [&](int index) {
            elements.at(index) = index;
        });
    } catch (const std::out_of_range& e) {
        std::cerr << "Caught '" << e.what() << "'." << std::endl;
    }
}
```

**Output:**

```
Caught 'invalid vector<T> subscript'.
```

**Explanation:**
Lambda uses `std::vector::at()` which throws if out of bounds. The exception is caught at the outer scope.

## **8. Lambda Expressions with Managed Types (C++/CLI)**

```cpp
// Compile with: /clr
using namespace System;

int main() {
    char ch = '!';

    [=](String^ s) {
        Console::WriteLine(s + Convert::ToChar(ch));
    }("Hello");
}
```

**Output:**

```
Hello!
```

**Explanation:**
The lambda captures unmanaged variable `ch` by value and takes a managed `System::String^` parameter.

## **Best Practices and Notes**

- Use **`auto` lambdas** where appropriate to minimize verbosity.
- Prefer **value capture** (`[=]`) for thread safety and lifetime guarantees unless mutation is required.
- Use **`mutable`** if modifying captured-by-value variables.
- Avoid capturing by reference in asynchronous or deferred execution contexts.
- Leverage lambdas in **STL algorithms** to avoid verbose function objects.
- Combine lambdas with **`std::function`** for higher-order function patterns.

## **Conclusion**

These examples demonstrate the versatility and expressive power of lambda expressions in C++. Whether you're processing containers, creating higher-order functions, or integrating with templates and class members, lambdas provide a concise and type-safe mechanism for defining behavior inline. Used correctly, they can significantly improve code clarity, modularity, and maintainability.
