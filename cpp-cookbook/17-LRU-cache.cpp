/**
 * Assignment: Implement an LRU Cache using STL Containers
 *
 * Implement an LRU cache with fixed capacity supporting:
 * - `get(key)`: returns value if present (marks as recently used), else returns -1.
 * - `put(key, value)`: inserts key-value pair or updates existing key.
 *   Evicts least-recently used item if capacity exceeded.
 *
 * Use:
 * - std::list for usage order (most recent at front, least at back).
 * - std::unordered_map for O(1) lookups from keys to list iterators.
 *
 * Bonus:
 * - Thread-safety with mutex.
 * - Generalize to a class template.
 * - Support smart pointer storage for large objects.
 * - Discuss why std::list is preferred here over std::deque.
 */

#include <iostream>
#include <list>
#include <unordered_map>

class LRUCache {
  private:
    using Key = int;
    using Value = int;
    using ListIt = std::list<std::pair<Key, Value>>::iterator;

    size_t capacity;
    std::list<std::pair<Key, Value>> itemList;
    std::unordered_map<Key, ListIt> itemMap;

  public:
    explicit LRUCache(size_t cap) : capacity(cap) {}

    int get(Key key) {
        auto it = itemMap.find(key);
        if (it == itemMap.end()) {
            return -1;
        }
        // Move the accessed item to the front
        itemList.splice(itemList.begin(), itemList, it->second);
        return it->second->second;
    }

    void put(Key key, Value value) {
        auto it = itemMap.find(key);
        if (it != itemMap.end()) {
            // Update value and move to front
            it->second->second = value;
            itemList.splice(itemList.begin(), itemList, it->second);
            return;
        }

        // Insert new key-value pair
        itemList.emplace_front(key, value);
        itemMap[key] = itemList.begin();

        // Evict if over capacity
        if (itemMap.size() > capacity) {
            auto lru = itemList.back();
            itemMap.erase(lru.first);
            itemList.pop_back();
        }
    }

    void printCache() const {
        std::cout << "Cache: ";
        for (const auto &[k, v] : itemList) {
            std::cout << "[" << k << ":" << v << "] ";
        }
        std::cout << "\n";
    }
};

int main() {
    LRUCache cache(3);

    cache.put(1, 10);
    cache.put(2, 20);
    cache.put(3, 30);
    cache.printCache();

    cache.get(2); // Access 2 (should move to front)
    cache.printCache();

    cache.put(4, 40); // Evict LRU (1)
    cache.printCache();

    std::cout << "Get key 1: " << cache.get(1) << "\n"; // -1 (not found)
    std::cout << "Get key 2: " << cache.get(2) << "\n"; // 20

    return 0;
}

/*
Cache: [3:30] [2:20] [1:10]
Cache: [2:20] [3:30] [1:10]
Cache: [4:40] [2:20] [3:30]
Get key 1: -1
Get key 2: 20
*/
