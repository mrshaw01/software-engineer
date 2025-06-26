# C++ OOP - Polymorphism

## What is Polymorphism?

**Polymorphism** means “many forms.” In C++, it refers to the ability of different classes related by inheritance to respond differently to the same function call.

Polymorphism allows a **single interface** to represent different underlying data types or behaviors. This makes code more flexible and extensible.

## Example Scenario

Imagine a base class `Animal` with a method `animalSound()`. Each derived class (e.g., `Pig`, `Dog`) provides its own version of the method:

### Example:

```cpp
// Base class
class Animal {
  public:
    void animalSound() {
      cout << "The animal makes a sound\n";
    }
};

// Derived class
class Pig : public Animal {
  public:
    void animalSound() {
      cout << "The pig says: wee wee\n";
    }
};

// Derived class
class Dog : public Animal {
  public:
    void animalSound() {
      cout << "The dog says: bow wow\n";
    }
};

int main() {
  Animal myAnimal;
  Pig myPig;
  Dog myDog;

  myAnimal.animalSound();
  myPig.animalSound();
  myDog.animalSound();
  return 0;
}
```

This is **compile-time polymorphism** (method overriding with the same name).
However, for **runtime polymorphism**, we need **virtual functions**.

## C++ Virtual Functions

A **virtual function** is declared in the base class using the `virtual` keyword and can be **overridden** in derived classes.

### Why Use Virtual Functions?

- Without `virtual`: function call is based on the **pointer type**.
- With `virtual`: function call is based on the **actual object type**.

## Without Virtual

### Example:

```cpp
class Animal {
  public:
    void sound() {
      cout << "Animal sound\n";
    }
};

class Dog : public Animal {
  public:
    void sound() {
      cout << "Dog barks\n";
    }
};

int main() {
  Animal* a;
  Dog d;
  a = &d;
  a->sound();  // Outputs: Animal sound
  return 0;
}
```

Even though `a` points to a `Dog`, it calls the base class version. Why? Because `sound()` is not virtual.

## With Virtual

### Example:

```cpp
class Animal {
  public:
    virtual void sound() {
      cout << "Animal sound\n";
    }
};

class Dog : public Animal {
  public:
    void sound() override {
      cout << "Dog barks\n";
    }
};

int main() {
  Animal* a;
  Dog d;
  a = &d;
  a->sound();  // Outputs: Dog barks
  return 0;
}
```

Now the correct function is called based on the actual object type (`Dog`), thanks to `virtual`.

## Key Rules

- Mark base class methods as `virtual`.
- Use `override` in the derived class (recommended for clarity, but optional).
- Use pointers or references to enable runtime polymorphism.

## About the `->` Operator

In `a->sound();`, the `->` operator is used because `a` is a pointer.

```cpp
a->sound();     // Same as (*a).sound();
```

> Use `->` to call methods or access members through a pointer.
