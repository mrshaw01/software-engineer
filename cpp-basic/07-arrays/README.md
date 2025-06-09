# C++: Arrays

## C++ Arrays

Arrays are used to store multiple values in a single variable, instead of declaring separate variables for each value.

### Declare and Initialize

```cpp
string cars[4] = {"Volvo", "BMW", "Ford", "Mazda"};
int myNum[3] = {10, 20, 30};
```

### Access Elements

```cpp
cout << cars[0]; // Outputs "Volvo"
```

> Note: Array indexes start at 0.

### Change Elements

```cpp
cars[0] = "Opel";
```

## Exercise

What is the output?

```cpp
string names[4] = {"Liam", "Jenny", "Angie", "Kasper"};
cout << names[2];
```

Correct answer: `Angie`

## C++ Arrays and Loops

### Loop with `for`

```cpp
string cars[5] = {"Volvo", "BMW", "Ford", "Mazda", "Tesla"};
for (int i = 0; i < 5; i++) {
  cout << cars[i] << "\n";
}
```

### Index and Value

```cpp
for (int i = 0; i < 5; i++) {
  cout << i << " = " << cars[i] << "\n";
}
```

### Integer Array Loop

```cpp
int myNumbers[5] = {10, 20, 30, 40, 50};
for (int i = 0; i < 5; i++) {
  cout << myNumbers[i] << "\n";
}
```

### For-Each Loop (C++11+)

```cpp
for (int i : myNumbers) {
  cout << i << "\n";
}
```

```cpp
for (string car : cars) {
  cout << car << "\n";
}
```

## Omit Array Size

```cpp
string cars[] = {"Volvo", "BMW", "Ford"};
```

Good practice (explicit size):

```cpp
string cars[3] = {"Volvo", "BMW", "Ford"};
```

### Add Elements Later

```cpp
string cars[5];
cars[0] = "Volvo";
// ...
```

> Cannot omit size in this case: `string cars[];` will cause a compile error.

## Fixed vs. Dynamic Size

### Fixed-Size Array

```cpp
string cars[3] = {"Volvo", "BMW", "Ford"};
// cars[3] = "Tesla"; // Error
```

### Dynamic Array (Vector)

```cpp
#include <vector>
vector<string> cars = {"Volvo", "BMW", "Ford"};
cars.push_back("Tesla");
```

> Vectors will be covered in a later chapter.

## Get Array Size

```cpp
int myNumbers[5] = {10, 20, 30, 40, 50};
cout << sizeof(myNumbers); // Outputs 20 (5 elements * 4 bytes)
```

### Get Number of Elements

```cpp
int length = sizeof(myNumbers) / sizeof(myNumbers[0]);
```

### Loop with Computed Size

```cpp
for (int i = 0; i < sizeof(myNumbers)/sizeof(myNumbers[0]); i++) {
  cout << myNumbers[i] << "\n";
}
```

### Cleaner with For-Each

```cpp
for (int i : myNumbers) {
  cout << i << "\n";
}
```

## Real-Life Examples

### Average Age

```cpp
int ages[8] = {20, 22, 18, 35, 48, 26, 87, 70};
float avg, sum = 0;
int length = sizeof(ages) / sizeof(ages[0]);
for (int age : ages) sum += age;
avg = sum / length;
cout << "The average age is: " << avg << "\n";
```

### Lowest Age

```cpp
int lowestAge = ages[0];
for (int age : ages) {
  if (lowestAge > age) lowestAge = age;
}
cout << "The lowest age is: " << lowestAge << "\n";
```

## Multi-Dimensional Arrays

```cpp
string letters[2][4] = {
  { "A", "B", "C", "D" },
  { "E", "F", "G", "H" }
};
```

### Three Dimensions

```cpp
string letters[2][2][2] = {
  {
    { "A", "B" },
    { "C", "D" }
  },
  {
    { "E", "F" },
    { "G", "H" }
  }
};
```

### Access and Modify

```cpp
cout << letters[0][2]; // Outputs "C"
letters[0][0] = "Z";
cout << letters[0][0]; // Outputs "Z"
```

### Loop 2D Array

```cpp
for (int i = 0; i < 2; i++) {
  for (int j = 0; j < 4; j++) {
    cout << letters[i][j] << "\n";
  }
}
```

### Loop 3D Array

```cpp
for (int i = 0; i < 2; i++) {
  for (int j = 0; j < 2; j++) {
    for (int k = 0; k < 2; k++) {
      cout << letters[i][j][k] << "\n";
    }
  }
}
```

## Real-World Example: Battleship Game

```cpp
bool ships[4][4] = {
  { 0, 1, 1, 0 },
  { 0, 0, 0, 0 },
  { 0, 0, 1, 0 },
  { 0, 0, 1, 0 }
};

int hits = 0;
int numberOfTurns = 0;

while (hits < 4) {
  int row, column;
  cout << "Choose a row (0-3): ";
  cin >> row;
  cout << "Choose a column (0-3): ";
  cin >> column;

  if (ships[row][column]) {
    ships[row][column] = 0;
    hits++;
    cout << "Hit! " << (4 - hits) << " left.\n\n";
  } else {
    cout << "Miss\n\n";
  }

  numberOfTurns++;
}

cout << "Victory!\n";
cout << "You won in " << numberOfTurns << " turns";
```
