#include <iostream>
#include <set>

int main() {
    std::set<int> unique_numbers;

    unique_numbers.insert(30);
    unique_numbers.insert(10);
    unique_numbers.insert(20);
    unique_numbers.insert(20);
    unique_numbers.insert(40);

    std::cout << "Elements in set (sorted and unique): ";
    for (const auto &num : unique_numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    int query = 20;
    if (unique_numbers.find(query) != unique_numbers.end()) {
        std::cout << query << " is present in the set." << std::endl;
    } else {
        std::cout << query << " is not present in the set." << std::endl;
    }

    unique_numbers.erase(10);

    std::cout << "Set after erasing 10: ";
    for (const auto &num : unique_numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}

/*
Elements in set (sorted and unique): 10 20 30 40
20 is present in the set.
Set after erasing 10: 20 30 40
*/
