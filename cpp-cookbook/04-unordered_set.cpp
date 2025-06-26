#include <iostream>
#include <string>
#include <unordered_set>

int main() {
    std::unordered_set<std::string> fruits;

    fruits.insert("apple");
    fruits.insert("banana");
    fruits.insert("cherry");
    fruits.insert("banana");

    std::cout << "Fruits in the set:" << std::endl;
    for (const auto &fruit : fruits) {
        std::cout << fruit << std::endl;
    }

    std::string query = "banana";
    if (fruits.find(query) != fruits.end()) {
        std::cout << query << " is present in the set." << std::endl;
    } else {
        std::cout << query << " is not present in the set." << std::endl;
    }

    fruits.erase("apple");

    std::cout << "After erasing 'apple':" << std::endl;
    for (const auto &fruit : fruits) {
        std::cout << fruit << std::endl;
    }

    return 0;
}

/*
Fruits in the set:
cherry
banana
apple
banana is present in the set.
After erasing 'apple':
cherry
banana
*/
