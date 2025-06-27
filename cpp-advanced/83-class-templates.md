# Class Templates: Comprehensive Overview

### 1. **What are Class Templates?**

Class templates enable **generic programming**, allowing you to define a blueprint for a class where types can be parameterized. This eliminates code duplication for classes that perform the same logic on different types.

**Syntax Example:**

```cpp
template<typename T>
class MyClass {
  T data;
public:
  void setData(const T& value) { data = value; }
  T getData() const { return data; }
};
```

**Usage:**

```cpp
MyClass<int> objInt;
objInt.setData(10);
std::cout << objInt.getData() << std::endl; // Output: 10

MyClass<std::string> objStr;
objStr.setData("Hello");
std::cout << objStr.getData() << std::endl; // Output: Hello
```

### 2. **Member Functions of Class Templates**

#### **Inside or Outside Definition**

Member functions of class templates can be defined **inside** or **outside** the class.

- **Inside:** Acts as an implicit inline definition.
- **Outside:** Requires repeating the full template parameter list.

**Example:**

```cpp
template<class T>
class Stack {
  std::vector<T> elements;
public:
  void push(const T& item);
  T pop();
};

template<class T>
void Stack<T>::push(const T& item) {
  elements.push_back(item);
}

template<class T>
T Stack<T>::pop() {
  T item = elements.back();
  elements.pop_back();
  return item;
}
```

**Expected Output:**

If used with `int`, pushing `1,2,3` and popping will yield `3,2,1` in order.

### 3. **Nested Class Templates**

Nested class templates are **templates declared within class templates or classes**, acting as member templates.

**Key points:**

- Inner template is scoped within the outer class.
- Enables hierarchical generic structures.

**Example:**

```cpp
template<class T>
class Outer {
public:
  template<class U>
  class Inner {
    U value;
  public:
    Inner(U v) : value(v) {}
    void print() { std::cout << value << std::endl; }
  };
};

int main() {
  Outer<int>::Inner<std::string> obj("Nested Template");
  obj.print(); // Output: Nested Template
}
```

### 4. **Member Function Templates**

A member function within a class template can itself be a template with **its own type parameters**.

**Example:**

```cpp
template<typename T>
class Container {
public:
  template<typename U>
  void display(const U& val) {
    std::cout << "Value: " << val << std::endl;
  }
};

int main() {
  Container<int> c;
  c.display(3.14); // Output: Value: 3.14
}
```

### 5. **Template Friends**

Class templates can declare:

- **Function templates** as friends.
- **Specific function specializations** as friends.
- **Other class templates** as friends.

#### **Example: Function Template Friend**

```cpp
template <class T>
class Box;

template <class T>
void printBox(Box<T>& b);

template <class T>
class Box {
  T item;
public:
  Box(T i): item(i) {}
  friend void printBox<>(Box<T>& b);
};

template <class T>
void printBox(Box<T>& b) {
  std::cout << "Box contains: " << b.item << std::endl;
}

int main() {
  Box<int> b(42);
  printBox(b); // Output: Box contains: 42
}
```

#### **Example: Friend Class Template**

```cpp
template <class T>
class Store;

template <class T>
class Manager {
public:
  void update(Store<T>& s, const T& val);
};

template <class T>
class Store {
  T data;
  friend class Manager<T>;
};

template <class T>
void Manager<T>::update(Store<T>& s, const T& val) {
  s.data = val;
}
```

### 6. **Template Parameter Reuse**

You can reuse template parameters within the parameter list to enforce type relationships.

**Example:**

```cpp
template<class T1, class T2 = T1>
class Pair {
  T1 first;
  T2 second;
};
```

- **`T2 = T1`** sets a default of `T2` to be the same type as `T1`.

### 7. **Best Practices**

- **Define small functions inline** within templates for performance-critical code.
- Use **`typename` keyword for dependent types** in templates to avoid compiler errors.
- For class templates with large logic, **split definition and implementation** into `.hpp` files included at the end of the header to avoid linker errors (due to templates needing full definitions).
- Prefer **template specialization** over partial specialization when needing distinct behavior for specific types.
- Be cautious with **friend declarations** in templates to avoid unintended tight coupling.
- Use **static_assert** for better compile-time diagnostics on template constraints.

### 8. **Summary**

Class templates in C++ are powerful tools for **generic, reusable, and type-safe code**, forming the backbone of STL containers and modern C++ libraries. Understanding their syntax, member functions, nested templates, and friend mechanics enables robust system design and advanced template metaprogramming.

If you need deeper template metaprogramming examples (e.g. SFINAE, enable_if, or concepts for constraints on class templates), let me know.
