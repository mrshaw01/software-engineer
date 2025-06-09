# C++ `new` and `delete`

## Introduction

C++ allows you to **manually manage memory** using the `new` and `delete` keywords.

- `new` is used to allocate memory dynamically (during runtime).
- `delete` is used to release memory previously allocated with `new`.

## Using `new` for Single Variables

You can allocate memory for a single variable using `new`:

```cpp
int* ptr = new int;
*ptr = 35;
cout << *ptr;  // Outputs 35
```

### Explanation:

- `new int` creates memory space for one integer.
- `ptr` stores the memory address.
- `*ptr = 35` assigns a value to that memory.
- `cout << *ptr` prints the value.

## Releasing Memory with `delete`

After you're done using the memory, **free it** using `delete`:

```cpp
delete ptr;
```

If you forget to delete, the memory is not returned to the system. This leads to a **memory leak**.

## Using `new` and `delete` with Arrays

You can also allocate memory for arrays using `new[]`, and release it using `delete[]`:

```cpp
int* arr = new int[5];
// Use arr...
delete[] arr;
```

## Example: Dynamic Guest List

```cpp
#include <iostream>
#include <string>
using namespace std;

int main() {
  int numGuests;
  cout << "How many guests? ";
  cin >> numGuests;

  if (numGuests <= 0) {
    cout << "Number of guests must be at least 1.\n";
    return 0;
  }

  string* guests = new string[numGuests];

  for (int i = 0; i < numGuests; i++) {
    cout << "Enter name for guest " << (i + 1) << ": ";
    cin >> guests[i];
  }

  cout << "\nGuests checked in:\n";
  for (int i = 0; i < numGuests; i++) {
    cout << guests[i] << "\n";
  }

  delete[] guests;
  return 0;
}
```

### Sample Output:

```
How many guests? 3
Enter name for guest 1: John Doe
Enter name for guest 2: Liam Spurs
Enter name for guest 3: Jenny Kasp
Guests checked in:
John Doe
Liam Spurs
Jenny Kasp
```

## When to Use `new`

Most of the time, C++ handles memory for you:

```cpp
int age = 35;
string name = "John";
```

However, you should use `new` when:

- You don’t know the amount of memory needed until runtime.
- You need flexible or large storage.
- You want explicit control over memory.

## Best Practice

> **If you use `new`, always remember to use `delete`.**
> For arrays: use `delete[]`.

Proper memory management is essential to avoid performance issues and crashes due to memory leaks.
