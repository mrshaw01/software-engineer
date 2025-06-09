# C++ OOP - Class Methods

## What are Class Methods?

In C++, **methods** are functions that belong to a class. They define behaviors or actions that class objects can perform.

You can define a method in two ways:

1. **Inside the class definition**
2. **Outside the class definition** (recommended for larger programs)

## 1. Method Defined Inside the Class

You can define a method directly inside the class body.

### Example:

```cpp
class MyClass {
  public:
    void myMethod() {
      cout << "Hello World!";
    }
};

int main() {
  MyClass myObj;
  myObj.myMethod();  // Call the method
  return 0;
}
```

### Explanation:

- `myMethod()` is a method of the class.
- We use the dot `.` operator on an object (`myObj`) to call the method.

## 2. Method Defined Outside the Class

In larger projects, it is common to separate the method definition from its declaration.

Use the **scope resolution operator** `::` to define the method outside the class.

### Example:

```cpp
class MyClass {
  public:
    void myMethod();  // Method declaration
};

// Method definition outside the class
void MyClass::myMethod() {
  cout << "Hello World!";
}

int main() {
  MyClass myObj;
  myObj.myMethod();  // Call the method
  return 0;
}
```

## Method with Parameters

You can pass values to class methods just like you do with regular functions.

### Example:

```cpp
#include <iostream>
using namespace std;

class Car {
  public:
    int speed(int maxSpeed);  // Declaration
};

int Car::speed(int maxSpeed) {  // Definition
  return maxSpeed;
}

int main() {
  Car myObj;
  cout << myObj.speed(200);  // Output: 200
  return 0;
}
```
