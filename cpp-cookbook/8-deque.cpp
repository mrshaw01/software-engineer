#include <deque>
#include <iostream>
#include <string>

int main() {
    std::deque<std::string> history;

    history.push_back("Page1");
    history.push_back("Page2");

    history.push_front("Login");

    std::cout << "Front: " << history.front() << std::endl;
    std::cout << "Back: " << history.back() << std::endl;

    history.pop_front();
    history.pop_back();

    std::cout << "After popping front and back:" << std::endl;
    for (const auto &page : history) {
        std::cout << page << std::endl;
    }

    history.push_back("Page3");
    history.insert(history.begin() + 1, "Home");

    std::cout << "Final history:" << std::endl;
    for (const auto &page : history) {
        std::cout << page << std::endl;
    }

    return 0;
}

/*
Front: Login
Back: Page2
After popping front and back:
Page1
Final history:
Page1
Home
Page3
*/
