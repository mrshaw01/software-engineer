/**
 * Assignment: Implement a Unique Smart Pointer
 *
 * Implement a simplified smart pointer class analogous to std::unique_ptr<T>.
 * - Manages a dynamically allocated object with exclusive ownership.
 * - Deletes the object in the destructor (RAII).
 * - Copying is disallowed (copy constructor and assignment deleted).
 * - Moving is allowed using std::move.
 * - Supports pointer-like operations: operator* and operator->.
 *
 * Bonus:
 * - Support a custom deleter (e.g., lambda or function object).
 * - Discuss how shared_ptr differs: uses control block, supports copy, and thread-safe reference counting.
 */

#include <functional>
#include <iostream>
#include <utility>

template <typename T, typename Deleter = std::default_delete<T>> class UniquePtr {
  private:
    T *ptr;
    Deleter deleter;

  public:
    explicit UniquePtr(T *p = nullptr, Deleter d = Deleter()) noexcept : ptr(p), deleter(d) {}
    ~UniquePtr() { reset(); }
    UniquePtr(const UniquePtr &) = delete;
    UniquePtr &operator=(const UniquePtr &) = delete;
    UniquePtr(UniquePtr &&other) noexcept : ptr(other.ptr), deleter(std::move(other.deleter)) { other.ptr = nullptr; }
    UniquePtr &operator=(UniquePtr &&other) noexcept {
        if (this != &other) {
            reset();
            ptr = other.ptr;
            deleter = std::move(other.deleter);
            other.ptr = nullptr;
        }
        return *this;
    }
    T &operator*() const noexcept { return *ptr; }
    T *operator->() const noexcept { return ptr; }
    T *get() const noexcept { return ptr; }
    T *release() noexcept {
        T *temp = ptr;
        ptr = nullptr;
        return temp;
    }
    void reset(T *p = nullptr) noexcept {
        if (ptr)
            deleter(ptr);
        ptr = p;
    }
    explicit operator bool() const noexcept { return ptr != nullptr; }
};

struct Sample {
    void greet() { std::cout << "Hello from Sample!\n"; }
};

int main() {
    UniquePtr<Sample> sp(new Sample());
    sp->greet();

    UniquePtr<Sample> sp2 = std::move(sp);
    if (!sp)
        std::cout << "Ownership moved to sp2.\n";
    sp2->greet();

    UniquePtr<Sample, std::function<void(Sample *)>> sp3(new Sample(), [](Sample *p) {
        std::cout << "Custom deleting...\n";
        delete p;
    });
    sp3->greet();

    return 0;
}

/*
Hello from Sample!
Ownership moved to sp2.
Hello from Sample!
Hello from Sample!
Custom deleting...
*/
