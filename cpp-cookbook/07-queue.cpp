#include <iostream>
#include <queue>
#include <string>

int main() {
    std::queue<std::string> tasks;

    tasks.push("Write report");
    tasks.push("Review code");
    tasks.push("Attend meeting");

    std::cout << "Front of the queue: " << tasks.front() << std::endl;
    std::cout << "Back of the queue: " << tasks.back() << std::endl;

    tasks.pop();

    std::cout << "After pop, front: " << tasks.front() << std::endl;

    std::cout << "Processing remaining tasks:" << std::endl;
    while (!tasks.empty()) {
        std::cout << tasks.front() << std::endl;
        tasks.pop();
    }

    return 0;
}

/*
Front of the queue: Write report
Back of the queue: Attend meeting
After pop, front: Review code
Processing remaining tasks:
Review code
Attend meeting
*/
