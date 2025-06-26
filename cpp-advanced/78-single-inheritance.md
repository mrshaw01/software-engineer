### Single Inheritance

Single inheritance is a fundamental object-oriented programming (OOP) mechanism where a derived class inherits from a single base class. It provides a way to promote code reuse, establish an “is-a” relationship, and support polymorphism, encapsulation, and extensibility without the complications of multiple inheritance.

## 1. Concept and Hierarchy

In single inheritance:

- A class can inherit from one (and only one) base class.
- The derived class inherits both behavior (functions) and state (data members) from the base class.
- The inheritance chain may continue through multiple levels (base → derived → further derived), forming a directed acyclic graph (DAG).

**Example:**

```cpp
class PrintedDocument {};

class Book : public PrintedDocument {};         // Book is a kind of PrintedDocument

class PaperbackBook : public Book {};           // PaperbackBook is a kind of Book
```

In this hierarchy:

- `PrintedDocument` is the base class of `Book`.
- `Book` is both a derived class (from `PrintedDocument`) and a base class (to `PaperbackBook`).
- `PaperbackBook` indirectly inherits from `PrintedDocument`.

## 2. Access Specifiers in Inheritance

The access specifier (`public`, `protected`, `private`) after the colon (`:`) in inheritance controls the accessibility of the base class members in the derived class:

- `public`: Public and protected members of the base class retain their accessibility in the derived class.
- `protected`: Public and protected members become protected in the derived class.
- `private`: Public and protected members become private in the derived class.

**Best practice:** Use `public` inheritance when establishing a true “is-a” relationship.

## 3. Inherited Members

A derived class automatically contains the members (except constructors, destructors, and assignment operators) of its base class. If the base class member is not hidden (by redefining or access restrictions), it can be accessed directly.

**Example:**

```cpp
class Document {
public:
    const char* Name;
    void PrintNameOf() const {
        std::cout << Name << std::endl;
    }
};

class Book : public Document {
public:
    Book(const char* name, long pageCount)
        : PageCount(pageCount) {
        Name = name;
    }

private:
    long PageCount;
};
```

**Output:**

```plaintext
Programming Windows, 2nd Ed
```

Calling `LibraryBook.PrintNameOf()` will invoke the method from the base `Document` class.

## 4. Overriding and Scope Resolution

If a derived class redefines a method with the same name as in the base class (but not declared `virtual`), the base version is hidden.

To access the base version explicitly:

```cpp
void Book::PrintNameOf() {
    std::cout << "Name of book: ";
    Document::PrintNameOf();  // Calls base class implementation
}
```

This explicit qualification avoids ambiguity and gives control over behavior composition.

## 5. Pointers and Polymorphism

In single inheritance, a pointer or reference to a derived object can be implicitly converted to a pointer or reference to its base class, enabling polymorphic behavior—if virtual functions are used.

**Example:**

```cpp
struct Document {
    virtual void PrintNameOf() const {
        std::cout << "Generic Document" << std::endl;
    }
};

class PaperbackBook : public Document {
public:
    void PrintNameOf() const override {
        std::cout << "Paperback Book" << std::endl;
    }
};

int main() {
    Document* docLib[2];
    docLib[0] = new Document;
    docLib[1] = new PaperbackBook;

    for (int i = 0; i < 2; ++i)
        docLib[i]->PrintNameOf();

    // Clean-up omitted for brevity
}
```

**Output:**

```plaintext
Generic Document
Paperback Book
```

Without virtual functions, the call would be statically dispatched, and `Document::PrintNameOf()` would always be invoked.

## 6. Best Practices and Pitfalls

### Best Practices:

- Use single inheritance when a clear "is-a" relationship exists.
- Declare destructors as `virtual` in base classes if polymorphism is involved.
- Minimize public data members; use accessors/mutators instead.
- Prefer composition over inheritance when appropriate.

### Pitfalls:

- Avoid base class slicing when passing derived objects by value.
- Don't overload without virtual dispatch if polymorphic behavior is intended.
- Avoid deep inheritance chains that reduce maintainability and clarity.

## 7. Summary

Single inheritance offers a clean, intuitive model for establishing object hierarchies in C++. It:

- Promotes code reuse.
- Enables polymorphic substitution when used with virtual functions.
- Keeps complexity lower than multiple inheritance.

However, care must be taken to manage access control, object lifetimes, and behavioral overrides correctly. When combined with virtual dispatch, single inheritance forms the backbone of C++'s object-oriented design.
