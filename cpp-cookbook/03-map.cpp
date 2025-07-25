#include <iostream>
#include <map>
#include <string>

int main() {
    std::map<std::string, int> word_count;

    word_count["apple"] = 3;
    word_count["banana"] = 5;
    word_count["cherry"] = 2;

    word_count["banana"] += 1;

    word_count.insert({"date", 4});

    std::cout << "Word counts:" << std::endl;
    for (const auto &pair : word_count) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    std::string query = "apple";
    if (word_count.find(query) != word_count.end()) {
        std::cout << query << " exists with count = " << word_count[query] << std::endl;
    } else {
        std::cout << query << " not found." << std::endl;
    }

    word_count.erase("cherry");

    std::cout << "After erasing 'cherry':" << std::endl;
    for (const auto &pair : word_count) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    return 0;
}

/*
Word counts:
apple: 3
banana: 6
cherry: 2
date: 4
apple exists with count = 3
After erasing 'cherry':
apple: 3
banana: 6
date: 4
 */
