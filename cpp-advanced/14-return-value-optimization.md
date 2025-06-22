# Return Value Optimization (RVO) in C++

## Introduction

Return Value Optimization (RVO) is a compiler optimization that eliminates the need to create temporary objects when returning values from functions. This technique minimizes unnecessary copy and move operations, particularly for large objects, improving performance and reducing the risk of resource leaks. RVO falls under the broader category of copy elision.

Since C++17, compilers are **required** to perform copy elision for temporary (unnamed) objects returned from functions, but RVO for named objects remains optional and implementation-dependent.

## 1. Terminology

- **RVO**: General term for copy elision during return, applicable to both named and unnamed return values.
- **URVO (Unnamed RVO)**: Copy elision when a function returns a temporary (unnamed) object.
- **NRVO (Named RVO)**: Copy elision when a function returns a named local object.

> The C++ standard does not officially define RVO or NRVO, but they are widely used terms in the C++ community.

## 2. Demonstrating RVO with a Test Struct

```cpp
static int counter;

struct S {
    int i{ 0 };
    int id;

    S() : id{ ++counter } { std::cout << "default ctor " << id << "\n"; }
    S(const S& s) : i{ s.i }, id{ ++counter } { std::cout << "copy ctor " << id << "\n"; }
    S& operator=(const S& s) {
        i = s.i;
        std::cout << "assign " << s.id << " to " << id << "\n";
        return *this;
    }
    ~S() { std::cout << "dtor " << id << "\n"; }
};
```

## 3. Unnamed RVO (URVO)

URVO applies when a function directly returns a temporary object:

```cpp
S get_B() {
    return S(); // C++17 requires copy elision here
}
```

### Behavior:

- **With URVO**: One object created, no copy constructor called.
- **Pre-C++17**: URVO is optional, but most compilers supported it.

## 4. Named RVO (NRVO)

NRVO occurs when returning a local named object:

```cpp
S get_C() {
    S s;
    s.i = 8;
    return s;
}
```

### Behavior:

- **With NRVO**: One object created, copy elision performed.
- **Without NRVO**: Two objects created, copy constructor called.

Compiler support:

- **GCC**: NRVO enabled by default.
- **MSVC**: Requires `/O2` optimization flag.

## 5. Compiler Limitations

### Conditional Returns

```cpp
S get_D1(int x) {
    S s;
    if (x % 2 == 0)
        return s;
    else
        return s;
}

S get_D2(int x) {
    if (x % 2 == 0) {
        S s1; return s1;
    } else {
        S s2; return s2;
    }
}
```

- `get_D1`: NRVO possible.
- `get_D2`: NRVO unlikely due to multiple return objects.

## 6. Calling Function's Role

RVO can be defeated by the way return values are received:

```cpp
S s;            // default ctor
s = get_E();    // copy or move assignment, RVO lost
```

Instead, write:

```cpp
S s = get_E();  // enables RVO/NRVO
```

## 7. Workarounds for Missing RVO

### Pass-by-Reference

```cpp
void get_F1(S& s) { s.i = 8; }
```

### Dynamic Allocation (Not Recommended)

```cpp
S* get_F2() {
    S* ps = new S;
    ps->i = 8;
    return ps;
}
```

> The caller must delete `ps` manually, risking memory leaks.

## 8. When to Forego RVO

In some cases, readability or scoping may require foregoing RVO:

```cpp
S s;
try {
    s = get(); // assignment disables RVO
    use(s);
} catch (...) {
    // ...
}
```

### Alternative that retains RVO:

```cpp
try {
    S s = get(); // RVO possible
    use(s);
    // ...
} catch (...) {
    // ...
}
```

## 9. Summary

- RVO avoids unnecessary copies when returning objects.
- C++17 mandates URVO but not NRVO.
- Always prefer returning unnamed temporaries or initializing directly.
- Watch for pitfalls in control flow or assignment that disable RVO.
- Measure and test RVO effects using constructors, destructors, and copy log outputs.
- Use NRVO-friendly patterns for performance-critical code, especially with large objects.
