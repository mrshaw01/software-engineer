# C++ OOP - Templates

## What Are Templates?

**Templates** allow you to write functions and classes that work with **any data type**.
They promote **code reuse**, reduce duplication, and support **generic programming**.

## Why Use Templates?

Templates help you:

- Avoid repeating logic for different types
- Write flexible and reusable code
- Implement type-independent algorithms and data structures

## Function Templates

Use function templates when you want to write the same function logic for multiple types.

### Syntax:

```cpp
template <typename T>
T functionName(T param) {
  // code using T
}
```

> `T` is a placeholder for any data type (e.g., `int`, `float`, `string`). You can use other names like `U`, `Type`, etc., but `T` is common.

### Example:

```cpp
template <typename T>
T add(T a, T b) {
  return a + b;
}

int main() {
  cout << add<int>(5, 3) << "\n";       // Outputs: 8
  cout << add<double>(2.5, 1.5) << "\n"; // Outputs: 4.0
  return 0;
}
```

## Class Templates

Use class templates to build type-independent classes like containers, boxes, or wrappers.

### Syntax:

```cpp
template <typename T>
class ClassName {
  // members and methods using T
};
```

### Example 1: Single Type Template Class

```cpp
template <typename T>
class Box {
  public:
    T value;

    Box(T v) {
      value = v;
    }

    void show() {
      cout << "Value: " << value << "\n";
    }
};

int main() {
  Box<int> intBox(50);
  Box<string> strBox("Hello");

  intBox.show();  // Outputs: Value: 50
  strBox.show();  // Outputs: Value: Hello
  return 0;
}
```

### Example 2: Multiple Type Template Class

```cpp
template <typename T1, typename T2>
class Pair {
  public:
    T1 first;
    T2 second;

    Pair(T1 a, T2 b) {
      first = a;
      second = b;
    }

    void display() {
      cout << "First: " << first << ", Second: " << second << "\n";
    }
};

int main() {
  Pair<string, int> person("John", 30);
  Pair<int, double> score(51, 9.5);

  person.display();  // Outputs: First: John, Second: 30
  score.display();   // Outputs: First: 51, Second: 9.5
  return 0;
}
```

## Important Notes

- Templates are typically defined in **header files (`.h`)** because the compiler needs the full definition during compilation.
- Templates are **type-safe**—you get compiler errors if used incorrectly with a type.
- C++ Standard Library heavily uses templates (e.g., `std::vector`, `std::map`).
