/**
 * Assignment: Implement a Thread-Safe Blocking Queue
 *
 * Implement a class template `ThreadSafeQueue<T>` that supports multiple producer/consumer threads.
 * - `push(T value)`: adds an element to the queue.
 * - `waitAndPop(T& value)`: blocks if the queue is empty, waits until an item is available.
 *
 * Key Concepts:
 * - Thread synchronization using `std::mutex`, `std::unique_lock`, and `std::condition_variable`
 * - Avoiding data races and ensuring exception safety
 * - Clean thread-safe design
 *
 * Bonus:
 * - Support `waitAndPop` with timeout
 * - Support bounded capacity with blocking push
 * - Graceful shutdown and cancellation handling
 */

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>

// Basic thread-safe blocking queue

template <typename T> class ThreadSafeQueue {
  private:
    std::queue<T> queue;
    mutable std::mutex mutex;
    std::condition_variable cond;
    bool shutdown = false;

  public:
    ThreadSafeQueue() = default;

    // Disable copy and assignment
    ThreadSafeQueue(const ThreadSafeQueue &) = delete;
    ThreadSafeQueue &operator=(const ThreadSafeQueue &) = delete;

    // Add an item to the queue
    void push(T value) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            queue.push(std::move(value));
        }
        cond.notify_one();
    }

    // Wait and pop an item
    bool waitAndPop(T &value) {
        std::unique_lock<std::mutex> lock(mutex);
        cond.wait(lock, [this]() { return !queue.empty() || shutdown; });

        if (shutdown && queue.empty())
            return false;

        value = std::move(queue.front());
        queue.pop();
        return true;
    }

    // Timed wait and pop
    bool waitAndPop(T &value, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex);
        if (!cond.wait_for(lock, timeout, [this]() { return !queue.empty() || shutdown; })) {
            return false; // timeout
        }

        if (shutdown && queue.empty())
            return false;

        value = std::move(queue.front());
        queue.pop();
        return true;
    }

    // Check if queue is empty
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.empty();
    }

    // Gracefully shutdown the queue
    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            shutdown = true;
        }
        cond.notify_all();
    }
};

// Example usage
#include <iostream>
#include <thread>

void producer(ThreadSafeQueue<int> &q) {
    for (int i = 0; i < 5; ++i) {
        std::cout << "Producing " << i << "\n";
        q.push(i);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    q.close();
}

void consumer(ThreadSafeQueue<int> &q) {
    int val;
    while (q.waitAndPop(val)) {
        std::cout << "Consumed " << val << "\n";
    }
    std::cout << "Consumer exiting\n";
}

int main() {
    ThreadSafeQueue<int> queue;

    std::thread t1(producer, std::ref(queue));
    std::thread t2(consumer, std::ref(queue));

    t1.join();
    t2.join();

    return 0;
}

/*
Producing 0
Consumed 0
Producing 1
Consumed 1
Producing 2
Consumed 2
Producing 3
Consumed 3
Producing 4
Consumed 4
Consumer exiting
*/
