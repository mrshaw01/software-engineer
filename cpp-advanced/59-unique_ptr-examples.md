# How to create and use `unique_ptr` instances

## 1. **What is `unique_ptr`?**

`std::unique_ptr<T>` is a smart pointer that **exclusively owns** a dynamically allocated object. When a `unique_ptr` goes out of scope, it **automatically deletes** the object it owns—ensuring no memory leaks.

### Key properties:

- **Non-copyable**: Ownership cannot be duplicated.
- **Moveable**: Ownership can be transferred using `std::move`.
- **Zero-overhead abstraction**: Same size and performance as raw pointer.
- **Defined in** `<memory>`

## 2. **Creation: Prefer `std::make_unique`**

Since C++14, the idiomatic way to create `unique_ptr` is:

```cpp
auto ptr = std::make_unique<MyType>(constructor_args...);
```

### Advantages:

- No risk of resource leaks due to temporary pointer lifetime.
- Exception-safe (unlike `unique_ptr<T>(new T(...))`).
- Cleaner syntax and clearer intent.

## 3. **Ownership Transfer via Move Semantics**

`unique_ptr` enforces single ownership. To transfer ownership, use `std::move`.

### Example:

```cpp
std::unique_ptr<Foo> a = std::make_unique<Foo>();
std::unique_ptr<Foo> b = std::move(a); // Ownership transferred
```

After the move:

- `a` becomes empty (i.e., `a.get() == nullptr`)
- `b` now owns the object

Attempting to copy will result in a **compiler error**.

## 4. **Use in Functions**

### Returning a `unique_ptr` from a factory:

```cpp
std::unique_ptr<Song> MakeSong(const std::wstring& artist, const std::wstring& title) {
    return std::make_unique<Song>(artist, title); // Implicit move
}
```

### Accepting a `unique_ptr`:

```cpp
void AcceptSong(std::unique_ptr<Song> song) {
    song->Play();
}

AcceptSong(std::make_unique<Song>(L"Adele", L"Hello"));
```

### Accepting by reference to modify the pointee but not transfer ownership:

```cpp
void UpdateTitle(Song& song) {
    song.title = L"Updated Title";
}
```

### Accepting by `const std::unique_ptr<T>&` for read-only access:

```cpp
void PrintSong(const std::unique_ptr<Song>& song) {
    std::wcout << song->title << std::endl;
}
```

## 5. **Use in Containers**

Standard containers like `std::vector` support move-only types.

### Example:

```cpp
std::vector<std::unique_ptr<Song>> playlist;

playlist.push_back(std::make_unique<Song>(L"Queen", L"Bohemian Rhapsody"));
playlist.push_back(std::make_unique<Song>(L"Elton John", L"Rocket Man"));
```

### Accessing safely:

```cpp
for (const auto& song : playlist) {
    std::wcout << song->title << std::endl;
}
```

Attempting `playlist.push_back(song)` will **fail** unless `song` is explicitly moved.

## 6. **Unique Pointers as Class Members**

`unique_ptr` is ideal for implementing **pImpl idioms**, resource management, or ownership semantics within a class.

### Example:

```cpp
class Controller {
    std::unique_ptr<Engine> engine;
public:
    Controller() : engine(std::make_unique<Engine>()) {}

    void Run() { engine->Start(); }
};
```

This pattern is safe, efficient, and reduces the need for manual destructor logic.

## 7. **Array Management with `unique_ptr`**

`std::make_unique<T[]>(size)` is used to manage arrays:

```cpp
auto arr = std::make_unique<int[]>(10);

for (int i = 0; i < 10; ++i)
    arr[i] = i * i;
```

Notes:

- You **must not** use `make_unique` to initialize each array element.
- Element access uses `operator[]`.

## 8. **Utility Functions and Methods**

| Method         | Description                                         |
| -------------- | --------------------------------------------------- |
| `.get()`       | Returns raw pointer without releasing ownership     |
| `.release()`   | Releases ownership and returns raw pointer          |
| `.reset(p)`    | Deletes old object and optionally takes new pointer |
| `.swap(other)` | Swaps ownership with another `unique_ptr`           |

### Example:

```cpp
std::unique_ptr<File> file = std::make_unique<File>();
File* raw = file.get();      // observe only
file.reset();                // destroy managed object
```

## 9. **Custom Deleters**

Useful when dealing with non-`delete` cleanup (e.g., file handles, sockets).

```cpp
std::unique_ptr<FILE, decltype(&fclose)> file(fopen("log.txt", "w"), &fclose);
```

Now `file` will automatically call `fclose` on destruction.

## 10. **Pitfalls to Avoid**

| Mistake                               | Why It's a Problem          | Better Approach                                       |
| ------------------------------------- | --------------------------- | ----------------------------------------------------- |
| Using `new` in function arguments     | May leak on exception       | Use `make_unique`                                     |
| Copying a `unique_ptr`                | Compilation error           | Use `std::move`                                       |
| Using `get()` without care            | May lead to double deletion | Avoid unless needed                                   |
| Resetting without releasing ownership | Destroys object             | Call `.release()` first if you want to retain pointer |

## 11. **Summary: Best Practices**

- **Prefer `std::make_unique<T>()`** over raw `new`.
- **Never copy** `unique_ptr`; always use `std::move`.
- Use **const references** for read-only access to the pointee.
- Use `unique_ptr<T[]>` for dynamic arrays.
- Use **custom deleters** for OS resources or C APIs.
- Ideal for RAII, encapsulated resources, factory patterns, and STL container integration.

## Final Thoughts

`std::unique_ptr` is a cornerstone of modern C++ memory safety. It gives precise control of ownership with zero runtime overhead and excellent compiler enforcement. By adopting `unique_ptr` rigorously in your codebase, you:

- Eliminate manual `delete` calls
- Prevent leaks and double frees
- Encode ownership semantics in the type system
- Simplify exception safety and maintenance

Use `unique_ptr` by default. Reach for `shared_ptr` only when shared ownership is truly required. Let the compiler work _with_ you—not _against_ you—to write correct and performant systems.
