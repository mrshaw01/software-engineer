# C++ OOP - Introduction to Classes and Objects

## What is OOP?

**OOP** stands for _Object-Oriented Programming_. It’s a programming paradigm based on the concept of "objects" — containers that group data and functions together.

### Benefits of OOP:

- Provides a clear structure to programs
- Makes code easier to maintain, reuse, and debug
- Follows the DRY principle (Don't Repeat Yourself)
- Enables building reusable components and applications

> DRY: Avoid writing the same code more than once. Extract repeated code into functions or classes.

## Classes vs. Objects

- A **class** defines the _structure_ and _behavior_ of data.
- An **object** is an _instance_ of a class.

| Class | Objects (Examples)   |
| ----- | -------------------- |
| Fruit | Apple, Banana, Mango |
| Car   | Volvo, Audi, Toyota  |

When you create an object from a class, it inherits all the class’s variables (attributes) and functions (methods).

## Procedural vs Object-Oriented

- **Procedural programming** focuses on writing functions that operate on data.
- **OOP** focuses on creating objects that _combine_ data and behavior.

## Creating a Class

To define a class, use the `class` keyword:

```cpp
class MyClass {
  public:
    int myNum;
    string myString;
};
```

### Explanation:

- `public`: an _access specifier_ making members accessible outside the class.
- `myNum` and `myString`: class attributes.

> Note: Always end a class definition with a semicolon `;`.

## Creating an Object

Objects are created from a class:

```cpp
int main() {
  MyClass myObj;              // Object creation
  myObj.myNum = 15;
  myObj.myString = "Some text";

  cout << myObj.myNum << "\n";
  cout << myObj.myString;
  return 0;
}
```

## Multiple Objects

You can create many objects from the same class:

```cpp
class Car {
  public:
    string brand;
    string model;
    int year;
};

int main() {
  Car carObj1, carObj2;

  carObj1.brand = "BMW";
  carObj1.model = "X5";
  carObj1.year = 1999;

  carObj2.brand = "Ford";
  carObj2.model = "Mustang";
  carObj2.year = 1969;

  cout << carObj1.brand << " " << carObj1.model << " " << carObj1.year << "\n";
  cout << carObj2.brand << " " << carObj2.model << " " << carObj2.year << "\n";
  return 0;
}
```
