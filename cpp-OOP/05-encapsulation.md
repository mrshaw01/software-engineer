# C++ OOP - Encapsulation

## What is Encapsulation?

**Encapsulation** is the practice of hiding sensitive data from users and providing controlled access through public methods.

It helps protect the internal state of an object and ensures data integrity by only allowing modifications through trusted interfaces.

## How to Achieve Encapsulation

1. **Declare attributes as `private`** — they cannot be accessed directly from outside the class.
2. **Provide `public` getter and setter methods** — to read or modify the private data safely.

## Real-Life Analogy

Think of an employee's salary:

- The salary is **private** — the employee can’t change it directly.
- Only the **manager** (trusted methods) can update or reveal it.

Encapsulation works the same way — it protects internal data and allows controlled interaction.

## Accessing Private Members

Use `public` methods to access and update private attributes.

### Example:

```cpp
#include <iostream>
using namespace std;

class Employee {
  private:
    int salary;  // Private attribute

  public:
    void setSalary(int s) {
      salary = s;
    }

    int getSalary() {
      return salary;
    }
};

int main() {
  Employee myObj;
  myObj.setSalary(50000);          // Set salary using setter
  cout << myObj.getSalary();       // Get salary using getter
  return 0;
}
```

### Explanation:

- `salary` is declared **private** — it cannot be accessed directly.
- `setSalary()` is a **setter** — it assigns a value.
- `getSalary()` is a **getter** — it retrieves the value.

## Why Use Encapsulation?

- **Better control** over your data
- **Hides complexity** from the user
- **Improves maintainability** by separating internal logic from external access
- **Increases security** by restricting direct access

> Best practice: Declare class attributes as `private` and provide controlled access through `public` methods.
