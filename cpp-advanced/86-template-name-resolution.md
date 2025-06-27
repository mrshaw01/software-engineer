# Templates and Name Resolution

In C++ template programming, name resolution involves identifying the correct meanings of identifiers within templates, especially distinguishing between dependent and non-dependent names. Understanding this distinction is crucial for writing robust and standards-compliant template code.

### Three Categories of Names in Templates

In template definitions, there are three primary categories of names:

1. **Locally declared names**:

   - Names defined within the template itself, including template parameters and member declarations.

2. **Non-dependent names (names from enclosing scope)**:

   - Identifiers defined outside of the template that do **not depend** on template parameters. These names are resolved once at the point of template definition.

3. **Dependent names**:

   - Identifiers that **depend** on one or more template parameters. Their actual meaning can only be determined at template instantiation.

### Dependent Names and Non-dependent Names

- **Non-dependent names**:

  - Resolved at the point of template definition.
  - Example: Standard library functions or types from global scope.

- **Dependent names**:

  - Resolved at template instantiation.
  - Example: Types or expressions involving template parameters.

### Examples of Dependent Types

Here are several cases illustrating what makes a type dependent:

```cpp
template <typename T>
class MyTemplate {
public:
    T a;                      // T itself is dependent.
    typename T::NestedType b; // Nested dependent type, requires typename.
    const T c;                // cv-qualified dependent type.
    T* ptr;                   // Pointer to dependent type.
    T array[5];               // Array of dependent type.
};
```

### Type Dependence vs. Value Dependence

Expressions and names in templates can also exhibit:

- **Type dependence**: The type itself depends on template parameters.
- **Value dependence**: The value of an expression depends on template parameters.

Example of value-dependent type:

```cpp
template <int N>
class Array {
    int data[N]; // size N is value-dependent
};
```

## Name Resolution for Dependent Types (`typename` keyword)

When using dependent types inside templates, C++ requires explicit indication using the `typename` keyword to inform the compiler that an identifier represents a type:

### Example:

```cpp
#include <iostream>

template <class T>
class Wrapper {
public:
    void show(typename T::Type value) {
        std::cout << "Value: " << value << '\n';
    }
};

struct MyStruct {
    using Type = int;
};

int main() {
    Wrapper<MyStruct> w;
    w.show(42);
    return 0;
}
```

**Expected Output:**

```
Value: 42
```

Explanation:
Without the `typename` keyword, the compiler cannot know whether `T::Type` is a type or a static member. Using `typename` explicitly clarifies this ambiguity.

## Name Resolution at Instantiation Context

When resolving dependent names, C++ uses two contexts:

1. **Template Definition Context**: for non-dependent names.
2. **Instantiation Context**: for dependent names.

Example:

```cpp
#include <iostream>

void display(char) {
    std::cout << "display(char)\n";
}

namespace Special {
    void display(int) {
        std::cout << "Special::display(int)\n";
    }
}

using namespace Special;

template <typename T>
void callDisplay(T value) {
    display(value); // Dependent, resolved at instantiation.
}

int main() {
    callDisplay(10);   // Calls Special::display(int)
    callDisplay('a');  // Calls display(char)
    return 0;
}
```

**Expected Output:**

```
Special::display(int)
display(char)
```

Explanation:
Dependent name `display` is resolved at template instantiation. The best matching function is chosen based on the actual argument type at instantiation time.

## Template Disambiguation (`template` keyword for nested templates)

When referring to nested templates, the keyword `template` must explicitly indicate nested template members.

### Example:

```cpp
template <typename T>
struct Allocator {
    template <typename U>
    struct Rebind {
        using type = Allocator<U>;
    };
};

template <typename X, typename AY>
struct Container {
    // Correct: explicitly indicates AY::Rebind is a template
    using AX = typename AY::template Rebind<X>::type;
};

int main() {
    Container<int, Allocator<double>> obj;
    return 0;
}
```

Explanation:
Without the keyword `template`, the compiler interprets `<` as less-than instead of the start of template arguments.

## Resolution of Locally Declared Names and Specializations

Within a template class:

- Unqualified class names refer to the current specialization.
- Qualified class names can explicitly refer to other specializations.

### Example of specialization:

```cpp
template <typename T>
class Node {
    Node* self;       // refers to Node<T>
    Node<int>* other; // explicitly refers to Node<int>
};

template <>
class Node<int> {
    Node* self;       // refers specifically to Node<int>
};
```

## Overload Resolution of Function Template Calls

Function templates can overload regular (non-template) functions:

- Compiler prefers non-template functions if equally good match exists.
- Exact matches are preferred over those requiring conversions.

### Example:

```cpp
#include <iostream>

void func(int, int) {
    std::cout << "func(int, int)\n";
}

template <typename T1, typename T2>
void func(T1 a, T2 b) {
    std::cout << "func(T1, T2)\n";
}

int main() {
    func(1, 2);        // Matches func(int, int)
    func('a', 1);      // Matches template func(T1, T2)
    func<long, int>(3L, 4); // Explicitly template func(T1, T2)
}
```

**Expected Output:**

```
func(int, int)
func(T1, T2)
func(T1, T2)
```

## Expert-level Best Practices Summary:

- **Use `typename`** explicitly for dependent nested types.
- **Clearly distinguish dependent vs non-dependent names** to avoid ambiguity.
- **Prefer fully qualified names** where ambiguity may arise.
- **Use the `template` keyword** explicitly to disambiguate nested templates.
- Understand that dependent names are resolved at **instantiation time**, impacting template behavior significantly.
- Explicitly specify template arguments only when necessary to control overload resolution.
