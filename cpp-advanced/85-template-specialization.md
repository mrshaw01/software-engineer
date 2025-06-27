# Template Specialization

**Templates** enable generic programming in C++, allowing functions and classes to operate with generic types. However, sometimes specific types require customized behavior. **Template specialization** allows you to provide tailored implementations for particular types while retaining generality for other types.

There are two forms:

1. **Full Specialization** – specialization for a particular type or type combination.
2. **Partial Specialization** – available only for class templates, where only some template parameters are specialized, while the rest remain generic.

Function templates do not support partial specialization directly, but similar effects can be achieved using overloads with `enable_if` or concepts in C++20.

## **1. Full Specialization**

This fully specializes a template for a specific type.

### **Example: Full Specialization for `int`**

```cpp
#include <iostream>

template<typename T>
struct Printer {
    void print(T value) {
        std::cout << "Generic print: " << value << std::endl;
    }
};

// Full specialization for int
template<>
struct Printer<int> {
    void print(int value) {
        std::cout << "Specialized print for int: " << value << std::endl;
    }
};

int main() {
    Printer<double> pd;
    pd.print(3.14); // Generic print: 3.14

    Printer<int> pi;
    pi.print(42); // Specialized print for int: 42

    return 0;
}
```

**Explanation:**

- The general template handles all types.
- The specialized template handles `int` with a different message.

## **2. Partial Specialization of Class Templates**

Partial specialization allows **customizing behavior for a subset of types while keeping the class generic for others**. This is widely used in type traits, containers, and meta-programming.

### **Example: Detecting Pointer and Pointer-to-Member Types**

```cpp
#include <iostream>

template<typename T>
struct TypeTrait {
    static constexpr bool is_pointer = false;
    static constexpr bool is_member_pointer = false;
};

// Specialization for pointer types
template<typename T>
struct TypeTrait<T*> {
    static constexpr bool is_pointer = true;
    static constexpr bool is_member_pointer = false;
};

// Specialization for pointer-to-member types
template<typename T, typename U>
struct TypeTrait<T U::*> {
    static constexpr bool is_pointer = false;
    static constexpr bool is_member_pointer = true;
};

struct MyStruct {};

int main() {
    std::cout << "MyStruct: is_pointer = " << TypeTrait<MyStruct>::is_pointer
              << ", is_member_pointer = " << TypeTrait<MyStruct>::is_member_pointer << std::endl;

    std::cout << "MyStruct*: is_pointer = " << TypeTrait<MyStruct*>::is_pointer
              << ", is_member_pointer = " << TypeTrait<MyStruct*>::is_member_pointer << std::endl;

    std::cout << "int MyStruct::*: is_pointer = " << TypeTrait<int MyStruct::*>::is_pointer
              << ", is_member_pointer = " << TypeTrait<int MyStruct::*>::is_member_pointer << std::endl;

    return 0;
}
```

**Expected Output:**

```
MyStruct: is_pointer = 0, is_member_pointer = 0
MyStruct*: is_pointer = 1, is_member_pointer = 0
int MyStruct::*: is_pointer = 0, is_member_pointer = 1
```

**Explanation:**

- The base template sets default values to `false`.
- The partial specialization for `T*` sets `is_pointer = true`.
- The partial specialization for pointer-to-member `T U::*` sets `is_member_pointer = true`.

## **3. Partial Specialization for Pointer Types in Collections**

This practical example demonstrates **storing values pointed to rather than pointers themselves**, useful in container designs.

### **Example: Bag Collection with Partial Specialization**

```cpp
#include <iostream>

template<typename T>
class Bag {
    T* data;
    size_t size;
    size_t capacity;

    void resize() {
        capacity = capacity ? capacity * 2 : 1;
        T* new_data = new T[capacity];
        for (size_t i = 0; i < size; ++i)
            new_data[i] = data[i];
        delete[] data;
        data = new_data;
    }

public:
    Bag() : data(nullptr), size(0), capacity(0) {}

    void add(const T& value) {
        if (size >= capacity)
            resize();
        data[size++] = value;
    }

    void print() const {
        for (size_t i = 0; i < size; ++i)
            std::cout << data[i] << " ";
        std::cout << std::endl;
    }

    ~Bag() { delete[] data; }
};

// Partial specialization for pointer types
template<typename T>
class Bag<T*> {
    T* data;
    size_t size;
    size_t capacity;

    void resize() {
        capacity = capacity ? capacity * 2 : 1;
        T* new_data = new T[capacity];
        for (size_t i = 0; i < size; ++i)
            new_data[i] = data[i];
        delete[] data;
        data = new_data;
    }

public:
    Bag() : data(nullptr), size(0), capacity(0) {}

    void add(T* ptr) {
        if (!ptr) {
            std::cout << "Null pointer ignored." << std::endl;
            return;
        }
        if (size >= capacity)
            resize();
        data[size++] = *ptr; // Store dereferenced value
    }

    void print() const {
        for (size_t i = 0; i < size; ++i)
            std::cout << data[i] << " ";
        std::cout << std::endl;
    }

    ~Bag() { delete[] data; }
};

int main() {
    Bag<int> bi;
    bi.add(5);
    bi.add(10);
    bi.print(); // 5 10

    int x = 42, y = 100;
    Bag<int*> bip;
    bip.add(&x);
    bip.add(&y);
    bip.print(); // 42 100

    int* null_ptr = nullptr;
    bip.add(null_ptr); // Null pointer ignored.

    return 0;
}
```

**Explanation:**

- The generic `Bag` stores values directly.
- The specialized `Bag<T*>` stores dereferenced values and checks for null pointers to avoid runtime errors.

## **4. Partial Specialization with One Type Fixed**

Sometimes a class template takes two types, but we want a specialized implementation when one of them is fixed to a specific type.

### **Example: Dictionary with Key Specialized to `int`**

```cpp
#include <iostream>
#include <algorithm>

template<typename Key, typename Value>
class Dictionary {
    Key* keys;
    Value* values;
    size_t size;
    size_t capacity;

    void resize() {
        capacity = capacity ? capacity * 2 : 1;
        Key* new_keys = new Key[capacity];
        Value* new_values = new Value[capacity];
        for (size_t i = 0; i < size; ++i) {
            new_keys[i] = keys[i];
            new_values[i] = values[i];
        }
        delete[] keys;
        delete[] values;
        keys = new_keys;
        values = new_values;
    }

public:
    Dictionary() : keys(nullptr), values(nullptr), size(0), capacity(0) {}

    void add(const Key& key, const Value& value) {
        if (size >= capacity)
            resize();
        keys[size] = key;
        values[size++] = value;
    }

    void print() const {
        for (size_t i = 0; i < size; ++i)
            std::cout << "{" << keys[i] << ", " << values[i] << "} ";
        std::cout << std::endl;
    }

    ~Dictionary() {
        delete[] keys;
        delete[] values;
    }
};

// Partial specialization where Key is int
template<typename Value>
class Dictionary<int, Value> {
    int* keys;
    Value* values;
    size_t size;
    size_t capacity;

    void resize() {
        capacity = capacity ? capacity * 2 : 1;
        int* new_keys = new int[capacity];
        Value* new_values = new Value[capacity];
        for (size_t i = 0; i < size; ++i) {
            new_keys[i] = keys[i];
            new_values[i] = values[i];
        }
        delete[] keys;
        delete[] values;
        keys = new_keys;
        values = new_values;
    }

public:
    Dictionary() : keys(nullptr), values(nullptr), size(0), capacity(0) {}

    void add(int key, const Value& value) {
        if (size >= capacity)
            resize();
        keys[size] = key;
        values[size++] = value;
    }

    void sort_by_key() {
        for (size_t i = 0; i < size - 1; ++i) {
            for (size_t j = i + 1; j < size; ++j) {
                if (keys[j] < keys[i]) {
                    std::swap(keys[i], keys[j]);
                    std::swap(values[i], values[j]);
                }
            }
        }
    }

    void print() const {
        for (size_t i = 0; i < size; ++i)
            std::cout << "{" << keys[i] << ", " << values[i] << "} ";
        std::cout << std::endl;
    }

    ~Dictionary() {
        delete[] keys;
        delete[] values;
    }
};

int main() {
    Dictionary<const char*, const char*> dict;
    dict.add("apple", "fruit");
    dict.add("dog", "animal");
    dict.print(); // {apple, fruit} {dog, animal}

    Dictionary<int, const char*> int_dict;
    int_dict.add(42, "answer");
    int_dict.add(7, "lucky");
    int_dict.print(); // {42, answer} {7, lucky}
    int_dict.sort_by_key();
    int_dict.print(); // {7, lucky} {42, answer}

    return 0;
}
```

**Explanation:**

- The general `Dictionary` handles any key-value pair.
- The specialized `Dictionary<int, Value>` adds a `sort_by_key` method and provides specialized storage for `int` keys.

## **Best Practices for Template Specialization**

1. **Minimize specializations** to maintain code generality and avoid maintenance burden.
2. Use **`static_assert`** in templates to provide meaningful compile-time error messages for unsupported types.
3. Prefer **type traits and SFINAE (or C++20 concepts)** for function template specialization needs.
4. Ensure **resource management** (e.g. proper destructors) in specialized classes to avoid leaks.
5. Always **test specializations thoroughly** to ensure correctness across general and specialized implementations.

### **Summary**

Template specialization, both full and partial, is a powerful technique in C++ to customize generic code behavior for specific types or type patterns while preserving general solutions. Mastery of this feature is essential for designing reusable and efficient libraries in high-performance or system-level C++ projects.
