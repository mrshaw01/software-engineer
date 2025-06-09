# C++ OOP - Constructors

## What is a Constructor?

A **constructor** is a special method that is **automatically called** when an object of a class is created. It is commonly used to initialize class attributes.

### Basic Rules:

- A constructor has the **same name** as the class.
- It has **no return type** (not even `void`).
- It is usually declared as `public`.
- It runs **automatically** when an object is created.

## Defining a Constructor

### Example:

```cpp
class MyClass {
  public:
    MyClass() {
      cout << "Hello World!";
    }
};

int main() {
  MyClass myObj;  // Constructor is called automatically
  return 0;
}
```

## Constructor with Parameters

Constructors can accept parameters to initialize class attributes when an object is created.

### Example:

```cpp
class Car {
  public:
    string brand;
    string model;
    int year;

    Car(string x, string y, int z) {
      brand = x;
      model = y;
      year = z;
    }
};

int main() {
  Car carObj1("BMW", "X5", 1999);
  Car carObj2("Ford", "Mustang", 1969);

  cout << carObj1.brand << " " << carObj1.model << " " << carObj1.year << "\n";
  cout << carObj2.brand << " " << carObj2.model << " " << carObj2.year << "\n";
  return 0;
}
```

## Defining a Constructor Outside the Class

You can define the constructor outside the class using the scope resolution operator `::`.

### Example:

```cpp
class Car {
  public:
    string brand;
    string model;
    int year;

    Car(string x, string y, int z);  // Constructor declaration
};

// Constructor definition outside the class
Car::Car(string x, string y, int z) {
  brand = x;
  model = y;
  year = z;
}
```

## Why Use Constructors?

Constructors make sure your object is ready to use as soon as it's created.

> Think of a constructor like a pizza chef — they prepare everything before delivering the pizza (object), so you don’t have to do it yourself.

## Constructor Overloading

You can have **multiple constructors** in a class with different sets of parameters. This is known as **constructor overloading**.

### Why overload constructors?

- Provide default or custom initialization
- Reduce repetitive code
- Increase flexibility when creating objects

### Example with Two Constructors:

```cpp
class Car {
  public:
    string brand;
    string model;

    Car() {
      brand = "Unknown";
      model = "Unknown";
    }

    Car(string b, string m) {
      brand = b;
      model = m;
    }
};

int main() {
  Car car1;
  Car car2("BMW", "X5");
  Car car3("Ford", "Mustang");

  cout << "Car1: " << car1.brand << " " << car1.model << "\n";
  cout << "Car2: " << car2.brand << " " << car2.model << "\n";
  cout << "Car3: " << car3.brand << " " << car3.model;
  return 0;
}
```

### Output:

```
Car1: Unknown Unknown
Car2: BMW X5
Car3: Ford Mustang
```
