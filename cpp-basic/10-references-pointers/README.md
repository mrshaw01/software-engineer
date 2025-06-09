# C++ References, Pointers, and Memory Management

## References

A **reference** is an alias for another variable. It is declared using the `&` symbol:

```cpp
string food = "Pizza";
string &meal = food;

cout << food << "\n";  // Outputs Pizza
cout << meal << "\n";  // Outputs Pizza
```

- Both `food` and `meal` refer to the **same memory location**.
- Changing one affects the other.

## Pointers

### What is a Pointer?

A **pointer** is a variable that stores the memory address of another variable.

### Creating a Pointer

```cpp
string food = "Pizza";
string* ptr = &food;

cout << food << "\n";   // Outputs Pizza
cout << &food << "\n";  // Outputs memory address
cout << ptr << "\n";    // Outputs memory address (same as above)
```

- Use `*` to declare a pointer.
- Use `&` to assign the address of a variable.

### Declaration Styles

```cpp
string* ptr;  // Preferred
string *ptr;
string * ptr;
```

## Dereferencing

You can use the `*` operator to get the value stored at the pointer’s address:

```cpp
string food = "Pizza";
string* ptr = &food;

cout << ptr << "\n";   // Outputs memory address
cout << *ptr << "\n";  // Outputs "Pizza"
```

- In declaration: `*` creates a pointer.
- In usage: `*` dereferences the pointer (gets the value).

## Modifying Values Through Pointers

```cpp
string food = "Pizza";
string* ptr = &food;

*ptr = "Hamburger";

cout << *ptr << "\n";  // Outputs Hamburger
cout << food << "\n";  // Also outputs Hamburger
```

- Modifying `*ptr` updates the original variable.

## Memory Management

### What is Memory Management?

It is the process of allocating, using, and freeing memory in your program.

### Automatic Memory Allocation

```cpp
int myNumber = 10;
```

C++ automatically reserves memory for simple variable declarations like the above.

### Checking Memory Size

Use `sizeof` to see how much memory a variable or type uses:

```cpp
int main() {
  int myInt;
  float myFloat;
  double myDouble;
  char myChar;

  cout << sizeof(myInt) << "\n";     // Typically 4 bytes
  cout << sizeof(myFloat) << "\n";   // 4 bytes
  cout << sizeof(myDouble) << "\n";  // 8 bytes
  cout << sizeof(myChar) << "\n";    // 1 byte
  return 0;
}
```

Understanding size helps optimize memory usage.

## Do You Need to Manage Memory?

- **No**, for regular variables: C++ handles memory automatically.
- **Yes**, for dynamically allocated memory (e.g., via `new`): You must release it manually using `delete`.

### Why It Matters

Poor memory management can cause:

- Memory leaks
- Performance issues
- Program crashes
