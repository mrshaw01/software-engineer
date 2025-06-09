# C++ Structures (`struct`)

## Introduction

Structures (also called **structs**) are a user-defined data type in C++ that group related variables into a single unit. Unlike arrays, structures can store **different data types** like `int`, `string`, `bool`, etc., in a single object.

Each variable in a structure is known as a **member**.

## Create a Structure

To create a structure, use the `struct` keyword:

```cpp
struct {
  int myNum;
  string myString;
} myStructure;
```

Here:

- `myNum` and `myString` are members of the structure.
- `myStructure` is the structure variable.

## Access Structure Members

Use the dot operator (`.`) to access members of the structure:

```cpp
myStructure.myNum = 1;
myStructure.myString = "Hello World!";

cout << myStructure.myNum << "\n";
cout << myStructure.myString << "\n";
```

## One Structure in Multiple Variables

You can declare multiple structure variables at once using a comma:

```cpp
struct {
  int myNum;
  string myString;
} myStruct1, myStruct2, myStruct3;
```

### Example

```cpp
struct {
  string brand;
  string model;
  int year;
} myCar1, myCar2;

myCar1.brand = "BMW";
myCar1.model = "X5";
myCar1.year = 1999;

myCar2.brand = "Ford";
myCar2.model = "Mustang";
myCar2.year = 1969;

cout << myCar1.brand << " " << myCar1.model << " " << myCar1.year << "\n";
cout << myCar2.brand << " " << myCar2.model << " " << myCar2.year << "\n";
```

## Named Structures

You can give a structure a name, allowing you to reuse it as a custom data type:

```cpp
struct car {
  string brand;
  string model;
  int year;
};
```

Now you can declare structure variables using `car`:

```cpp
car myCar1;
myCar1.brand = "BMW";
myCar1.model = "X5";
myCar1.year = 1999;
```

### Full Example

```cpp
#include <iostream>
#include <string>
using namespace std;

struct car {
  string brand;
  string model;
  int year;
};

int main() {
  car myCar1;
  myCar1.brand = "BMW";
  myCar1.model = "X5";
  myCar1.year = 1999;

  car myCar2;
  myCar2.brand = "Ford";
  myCar2.model = "Mustang";
  myCar2.year = 1969;

  cout << myCar1.brand << " " << myCar1.model << " " << myCar1.year << "\n";
  cout << myCar2.brand << " " << myCar2.model << " " << myCar2.year << "\n";

  return 0;
}
```

## Challenge Task

Create a structure named `student` with the following members:

- `name` (string)
- `age` (int)
- `grade` (char)

### Steps:

1. Create one `student` variable.
2. Assign values to its members.
3. Print the values.

### Expected Output

```
Name: Liam
Age: 35
Grade: A
```

### Solution

```cpp
#include <iostream>
#include <string>
using namespace std;

struct student {
  string name;
  int age;
  char grade;
};

int main() {
  student s1;
  s1.name = "Liam";
  s1.age = 35;
  s1.grade = 'A';

  cout << "Name: " << s1.name << "\n";
  cout << "Age: " << s1.age << "\n";
  cout << "Grade: " << s1.grade << "\n";

  return 0;
}
```
