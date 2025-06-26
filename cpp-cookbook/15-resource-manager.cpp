/**
 * Assignment: Implement a Move-Enabled Resource Manager
 *
 * Implement a class `Buffer` that manages a dynamically allocated array.
 * - Copy constructor/assignment: perform deep copy.
 * - Move constructor/assignment: transfer ownership (steal pointer + size).
 * - Destructor: releases the resource.
 *
 * Key Concepts:
 * - Rule of Five (copy/move constructor and assignment, destructor)
 * - Deep vs. shallow copy
 * - Move semantics and performance
 * - Memory safety (no leaks/double frees)
 *
 * Bonus:
 * - Use copy-and-swap idiom for exception-safe assignment.
 * - Compare with standard containers (Rule of Zero)
 * - Consider making the class a generic template for arbitrary resource types
 */

#include <algorithm>
#include <cstring> // for std::memcpy
#include <iostream>

class Buffer {
  private:
    size_t size;
    char *data;

  public:
    // Constructor
    explicit Buffer(size_t sz) : size(sz), data(new char[sz]) {
        std::fill(data, data + size, 0);
        std::cout << "Constructed Buffer of size " << size << "\n";
    }

    // Destructor
    ~Buffer() {
        delete[] data;
        std::cout << "Destroyed Buffer\n";
    }

    // Copy constructor
    Buffer(const Buffer &other) : size(other.size), data(new char[other.size]) {
        std::memcpy(data, other.data, size);
        std::cout << "Copied Buffer\n";
    }

    // Move constructor
    Buffer(Buffer &&other) noexcept : size(other.size), data(other.data) {
        other.data = nullptr;
        other.size = 0;
        std::cout << "Moved Buffer\n";
    }

    // Copy assignment operator
    Buffer &operator=(const Buffer &other) {
        if (this != &other) {
            char *newData = new char[other.size];
            std::memcpy(newData, other.data, other.size);
            delete[] data;
            data = newData;
            size = other.size;
            std::cout << "Copy-assigned Buffer\n";
        }
        return *this;
    }

    // Move assignment operator
    Buffer &operator=(Buffer &&other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
            std::cout << "Move-assigned Buffer\n";
        }
        return *this;
    }

    // Accessors for demonstration
    size_t getSize() const { return size; }
    char *getData() const { return data; }
};

// Example usage
int main() {
    Buffer buf1(10);
    Buffer buf2 = buf1;            // copy
    Buffer buf3 = std::move(buf1); // move

    Buffer buf4(20);
    buf4 = buf2;            // copy assignment
    buf4 = std::move(buf3); // move assignment

    return 0;
}

/*
Constructed Buffer of size 10
Copied Buffer
Moved Buffer
Constructed Buffer of size 20
Copy-assigned Buffer
Move-assigned Buffer
Destroyed Buffer
Destroyed Buffer
Destroyed Buffer
Destroyed Buffer
*/
