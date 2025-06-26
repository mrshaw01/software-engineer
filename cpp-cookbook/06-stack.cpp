#include <iostream>
#include <stack>
#include <string>

int main() {
    std::stack<std::string> names;

    names.push("Alice");
    names.push("Bob");
    names.push("Charlie");

    std::cout << "Top element: " << names.top() << std::endl;

    names.pop();

    std::cout << "After pop, top element: " << names.top() << std::endl;

    std::cout << "Popping all elements:" << std::endl;
    while (!names.empty()) {
        std::cout << names.top() << std::endl;
        names.pop();
    }

    return 0;
}

/*
Top element: Charlie
After pop, top element: Bob
Popping all elements:
Bob
Alice
*/
