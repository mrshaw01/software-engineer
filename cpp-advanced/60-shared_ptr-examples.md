# How to create and use `shared_ptr` instances

## 1. **What is `std::shared_ptr`?**

`std::shared_ptr<T>` is a smart pointer that **enables multiple owners** to share responsibility for a dynamically allocated object. It automatically manages reference counting via an internal **control block**. When the last `shared_ptr` referencing the resource is destroyed or reset, the resource is deleted.

### Characteristics:

- Copyable and moveable
- Thread-safe reference counting
- May lead to cycles — use `weak_ptr` to break them

Defined in `<memory>`.

## 2. **Prefer `std::make_shared`**

Use `make_shared<T>()` over `shared_ptr<T>(new T(...))`:

### Benefits:

- More efficient (single heap allocation for control block + object)
- Exception-safe
- Cleaner syntax

```cpp
auto sp = std::make_shared<Song>(L"Queen", L"Don't Stop Me Now");
```

Avoid:

```cpp
shared_ptr<Song> sp(new Song(...)); // Less efficient, not exception-safe
```

## 3. **Copying and Reference Counting**

Copying a `shared_ptr` increases the reference count. Destruction or reset decreases it. When count reaches 0, the object and control block are destroyed.

```cpp
auto sp1 = std::make_shared<Song>(L"Adele", L"Easy On Me");
auto sp2 = sp1; // ref count = 2

sp1.reset();    // ref count = 1
sp2.reset();    // ref count = 0 → object deleted
```

## 4. **Passing `shared_ptr` to Functions**

### Guidelines:

| Goal                            | How to Pass                         | Ref Count Changed? | Ownership?            |
| ------------------------------- | ----------------------------------- | ------------------ | --------------------- |
| Callee _must_ share ownership   | By value (`shared_ptr<T>`)          | Yes                | Yes                   |
| Callee only needs access        | By `const shared_ptr<T>&`           | No                 | No                    |
| Temporary access to object only | By raw pointer (`T*`) or ref (`T&`) | No                 | No                    |
| Callee takes over ownership     | By `std::move(shared_ptr)`          | No (transfers)     | Ownership transferred |

### Examples:

```cpp
void process(const std::shared_ptr<Song>& song); // observer
void take_ownership(std::shared_ptr<Song> song); // shared owner
```

## 5. **Use in Containers**

`shared_ptr` works seamlessly with STL containers like `std::vector`, even with copying algorithms.

```cpp
std::vector<std::shared_ptr<Song>> songs = {
    std::make_shared<Song>(L"Linkin Park", L"In The End"),
    std::make_shared<Song>(L"Coldplay", L"Yellow")
};

// Copy non-Linkin Park songs
std::vector<std::shared_ptr<Song>> others;
std::remove_copy_if(songs.begin(), songs.end(), std::back_inserter(others),
    [](const std::shared_ptr<Song>& s) {
        return s->artist == L"Linkin Park";
    });
```

The object remains valid as long as at least one `shared_ptr` instance exists.

## 6. **Type-Safe Casting Between `shared_ptr`**

Like raw pointers, `shared_ptr` supports safe casting using:

- `std::dynamic_pointer_cast<T>(p)`
- `std::static_pointer_cast<T>(p)`
- `std::const_pointer_cast<T>(p)`

```cpp
std::shared_ptr<MediaAsset> asset = std::make_shared<Photo>(...);

if (auto photo = std::dynamic_pointer_cast<Photo>(asset)) {
    std::wcout << photo->location << std::endl;
}
```

## 7. **Comparison and Ordering**

`shared_ptr` supports comparison based on underlying pointer:

```cpp
auto sp1 = std::make_shared<Song>(L"X", L"Y");
auto sp2 = sp1;               // same resource
auto sp3 = std::make_shared<Song>(L"X", L"Y"); // different object

sp1 == sp2 → true
sp1 == sp3 → false
sp1 < sp3 → pointer comparison (platform-specific)
```

Use `owner_before()` to compare ownership rather than pointee address.

## 8. **Resetting and Swapping**

```cpp
std::shared_ptr<Song> a = std::make_shared<Song>(...);
std::shared_ptr<Song> b = std::make_shared<Song>(...);

a.swap(b);    // swaps ownership
a.reset();    // a no longer owns the object
```

You can reset with a new pointer or leave it empty.

## 9. **Avoid Common Pitfalls**

### 9.1. **Double-Free from Raw Pointer Copy**

```cpp
auto sp = std::make_shared<Foo>();
Foo* raw = sp.get();

// Dangerous: don't do this
std::shared_ptr<Foo> another(raw); // undefined behavior
```

**Fix**: Never create a new shared*ptr from a raw pointer unless you're \_sure* it's not already owned.

### 9.2. **Circular References**

```cpp
struct Node {
    std::shared_ptr<Node> next;
    std::shared_ptr<Node> prev;
};
```

Creates a cycle → memory leak.

**Fix**: Break the cycle with `std::weak_ptr`.

```cpp
struct Node {
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> prev; // not owning
};
```

## 10. **Use Case Summary**

| Use Case                               | Preferred Smart Pointer   |
| -------------------------------------- | ------------------------- |
| Default ownership                      | `std::unique_ptr<T>`      |
| Shared lifetime (e.g., plugin graph)   | `std::shared_ptr<T>`      |
| Observing shared object without owning | `std::weak_ptr<T>`        |
| Managing OS resources (custom cleanup) | `shared_ptr<T>` + deleter |

## 11. **Custom Deleter**

```cpp
std::shared_ptr<FILE> file(
    fopen("file.txt", "r"),
    [](FILE* f) { if (f) fclose(f); }
);
```

Ensures `fclose()` is called when `file` goes out of scope.

## 12. **Thread Safety**

- **Reference count is thread-safe**
- **Access to the underlying object is not**

Use `std::mutex` or `std::atomic<T>` for shared data if needed.

## Final Thoughts

`std::shared_ptr` is a powerful abstraction when used judiciously. It enables shared ownership and integrates smoothly into standard C++ containers and algorithms. However, it should be reserved for cases where:

- Ownership truly needs to be shared
- Lifetime management across modules, tasks, or systems is required

For most single-owner resources, **prefer `unique_ptr`**. Overuse of `shared_ptr` introduces **performance overhead**, **potential leaks (via cycles)**, and **less clarity of ownership**.

### Best Practices:

- Use `make_shared` for creation
- Avoid passing raw pointers from `shared_ptr`
- Pass by reference when sharing ownership isn't needed
- Use `weak_ptr` to break cycles
- Do not use `shared_ptr<T>(raw_ptr)` unless raw_ptr is not already owned
