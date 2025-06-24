### Explicitly Defaulted and Deleted Functions in C++11 and Beyond

C++11 introduced a critical enhancement to the language's object model with _explicitly defaulted_ (`= default`) and _explicitly deleted_ (`= delete`) functions. These tools provide precise control over the automatic generation of _special member functions_ (SMFs), enable safer and clearer design patterns, and help avoid subtle and error-prone behavior stemming from implicit compiler actions.

## Background: Special Member Functions

In C++, the compiler will implicitly declare and define the following _special member functions_ for a class if they are not explicitly provided:

1. Default constructor
2. Destructor
3. Copy constructor
4. Copy assignment operator
5. Move constructor (C++11)
6. Move assignment operator (C++11)

The generation of these functions follows a set of interdependent rules. Declaring one can suppress the generation of others. This can lead to unintuitive situations where a seemingly innocent declaration unexpectedly prevents other operations from being available or efficient.

## Explicitly Defaulted Functions (`= default`)

Defaulted functions provide a mechanism to **explicitly state that a special member function should be compiler-generated**.

### Syntax

You can declare a function as defaulted either inside or outside the class body:

```cpp
struct Widget {
    Widget() = default;                            // inline defaulted
    Widget& operator=(const Widget&);              // declared only
};

Widget& Widget::operator=(const Widget&) = default; // defined later as default
```

### Use Cases

1. **Restoring default generation after suppression**
   If you declare another SMF (e.g., destructor), you may unintentionally suppress the default constructor or move constructor. `= default` restores them explicitly.

2. **Changing access levels**
   A default constructor or assignment operator can be made `private` or `protected` to control instantiation or assignment externally.

3. **Preserving triviality and POD status**
   Defaulted functions are treated as _trivial_ if the default implementation would have been trivial. This preserves optimizations like aggregate initialization and memory layout compatibility with C.

4. **Documenting intent**
   Declaring a function as defaulted clarifies that the default semantics are intentional, rather than an oversight.

### Best Practice

Prefer defaulted special member functions over empty user-defined ones. This maintains compiler optimizations, reduces binary size, and clearly communicates intent.

## Explicitly Deleted Functions (`= delete`)

Deleted functions prevent certain operations from being compiled, even though they might otherwise be valid or implicitly generated.

### Syntax

You must delete a function at the point of declaration:

```cpp
struct NonCopyable {
    NonCopyable() = default;
    NonCopyable(const NonCopyable&) = delete;
    NonCopyable& operator=(const NonCopyable&) = delete;
};
```

### Use Cases

1. **Prevent copy or move semantics**
   Making a type non-copyable or non-movable is idiomatically expressed using `= delete`.

2. **Prohibit heap allocation or deallocation**

   ```cpp
   void* operator new(std::size_t) = delete;  // prohibit dynamic allocation
   void operator delete(void*) = delete;      // prohibit dynamic deallocation
   ```

3. **Disable function overloads to prevent implicit type conversions**

   ```cpp
   void process(double);                      // valid
   void process(float) = delete;              // prevents float-to-double promotion
   ```

4. **Catch incorrect or unintended function calls during compilation**

   ```cpp
   template <typename T>
   void process(T) = delete;                  // disallow all non-explicit overloads
   ```

### Semantics

Deleted functions **participate in overload resolution**. If a deleted function is the best match for a given call, the program is ill-formed. This feature allows fine-grained control over overload selection.

## Practical Implications in Object-Oriented Design

### Modern Non-Copyable Base

Prior to C++11:

```cpp
struct NonCopyable {
private:
    NonCopyable(const NonCopyable&);
    NonCopyable& operator=(const NonCopyable&);
};
```

Issues:

- Compiler generates a non-trivial default constructor.
- Linker errors if accidentally used internally.
- Intent not immediately clear to readers.

In C++11 and later:

```cpp
struct NonCopyable {
    NonCopyable() = default;
    NonCopyable(const NonCopyable&) = delete;
    NonCopyable& operator=(const NonCopyable&) = delete;
};
```

Advantages:

- Intent is explicit and enforced at compile time.
- Default constructor remains trivial.
- Enables POD-compatible layout and initialization when applicable.

## Guidelines and Best Practices

1. **Use `= default`**:

   - When reinstating compiler-generated behavior that was suppressed.
   - To maintain triviality or clarity of design.
   - When customizing visibility or access control.

2. **Use `= delete`**:

   - To prevent undesired or dangerous usage patterns (e.g., copy, move, dynamic allocation).
   - To enforce design constraints through the type system.
   - To prevent unintended overload resolution paths.

3. **Avoid empty user-defined special member functions** unless necessary. Let the compiler generate them or mark them explicitly with `= default`.

4. **Don't delete functions after declaration.** `= delete` must appear at the point of declaration.

## Summary

Explicitly defaulted and deleted functions are powerful tools that modernize and clarify C++ class design. They:

- Replace brittle idioms with type-safe, compiler-enforced constructs.
- Clarify programmer intent and make class semantics more explicit.
- Enhance performance by retaining triviality where possible.
- Improve maintainability by simplifying special member function management.

Used properly, these features significantly reduce boilerplate, prevent subtle bugs, and elevate the clarity and correctness of C++ class interfaces. As a rule, lean on the compiler for correct, efficient behavior—and intervene only when your design demands it.
