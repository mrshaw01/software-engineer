/**
 * Assignment: Use STL Algorithms and Lambdas for Data Transformation
 *
 * Given a std::vector<std::string>, perform the following operations:
 * 1. Remove all strings shorter than 3 characters.
 * 2. Convert all remaining strings to uppercase.
 *
 * Use STL algorithms such as std::remove_if, std::transform, and lambda expressions.
 * Do not use manual loops.
 *
 * Bonus:
 * - Discuss performance and complexity implications (O(n) twice).
 * - Parameterize the filter length and transformation logic.
 * - Generalize for different container or value types.
 */

#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>
#include <vector>

void transformWords(std::vector<std::string> &words, size_t minLength = 3) {
    // Remove words shorter than minLength
    words.erase(std::remove_if(words.begin(), words.end(),
                               [minLength](const std::string &word) { return word.length() < minLength; }),
                words.end());

    // Convert each word to uppercase
    std::transform(words.begin(), words.end(), words.begin(), [](std::string word) {
        std::transform(word.begin(), word.end(), word.begin(), ::toupper);
        return word;
    });
}

void printWords(const std::vector<std::string> &words) {
    for (const auto &word : words) {
        std::cout << word << " ";
    }
    std::cout << "\n";
}

int main() {
    std::vector<std::string> words = {"a", "an", "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"};

    std::cout << "Original words:\n";
    printWords(words);

    transformWords(words);

    std::cout << "\nTransformed words:\n";
    printWords(words);

    return 0;
}

/*
Original words:
a an the quick brown fox jumps over lazy dog

Transformed words:
THE QUICK BROWN FOX JUMPS OVER LAZY DOG
*/
