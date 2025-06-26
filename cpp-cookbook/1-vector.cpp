#include <iostream>
#include <vector>

int main() {
    std::vector<int> numbers;

    numbers.push_back(10);
    numbers.push_back(20);
    numbers.push_back(30);
    numbers.push_back(40);

    std::cout << "Second element: " << numbers[1] << std::endl;

    numbers[2] = 35;

    std::cout << "Elements in vector: ";
    for (const auto &num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    std::cout << "Size of vector: " << numbers.size() << std::endl;

    numbers.pop_back();

    std::cout << "Vector after pop_back: ";
    for (const auto &num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}

/*
Second element: 20
Elements in vector: 10 20 35 40
Size of vector: 4
Vector after pop_back: 10 20 35
*/
