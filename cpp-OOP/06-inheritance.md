# C++ OOP - Inheritance

## What is Inheritance?

**Inheritance** is a feature of object-oriented programming that allows one class to inherit the attributes and methods of another. It promotes **code reuse**, reduces redundancy, and supports better program structure.

## Terminology

- **Base class (parent)**: The class being inherited from.
- **Derived class (child)**: The class that inherits from the base class.

## Basic Inheritance

Use the `:` symbol to indicate that a class is derived from another.

### Example:

```cpp
// Base class
class Vehicle {
  public:
    string brand = "Ford";
    void honk() {
      cout << "Tuut, tuut!\n";
    }
};

// Derived class
class Car: public Vehicle {
  public:
    string model = "Mustang";
};

int main() {
  Car myCar;
  myCar.honk();  // Inherited method
  cout << myCar.brand + " " + myCar.model;
  return 0;
}
```

## Why Use Inheritance?

- Reuse existing code instead of rewriting
- Organize and group related classes
- Extend base functionality in child classes

## Multilevel Inheritance

A derived class can itself serve as a base class for another class.

### Example:

```cpp
// Base class
class MyClass {
  public:
    void myFunction() {
      cout << "Some content in parent class.";
    }
};

// Derived class
class MyChild: public MyClass {
};

// Derived from MyChild
class MyGrandChild: public MyChild {
};

int main() {
  MyGrandChild myObj;
  myObj.myFunction();  // Inherited from grandparent
  return 0;
}
```

## Multiple Inheritance

A class can inherit from **more than one** base class.

### Example:

```cpp
// First base class
class MyClass {
  public:
    void myFunction() {
      cout << "Some content in parent class.";
    }
};

// Second base class
class MyOtherClass {
  public:
    void myOtherFunction() {
      cout << "Some content in another class.";
    }
};

// Derived class from both
class MyChildClass: public MyClass, public MyOtherClass {
};

int main() {
  MyChildClass myObj;
  myObj.myFunction();
  myObj.myOtherFunction();
  return 0;
}
```

## Inheritance and Access Specifiers

You’ve already learned about `public`, `private`, and `protected` access specifiers.

- `public`: Accessible everywhere
- `private`: Only accessible within the class
- `protected`: Not accessible outside the class, **but available to derived classes**

### Example Using `protected`:

```cpp
// Base class
class Employee {
  protected:
    int salary;
};

// Derived class
class Programmer: public Employee {
  public:
    int bonus;

    void setSalary(int s) {
      salary = s;  // Accessible via protected
    }

    int getSalary() {
      return salary;
    }
};

int main() {
  Programmer myObj;
  myObj.setSalary(50000);
  myObj.bonus = 15000;

  cout << "Salary: " << myObj.getSalary() << "\n";
  cout << "Bonus: " << myObj.bonus << "\n";
  return 0;
}
```
