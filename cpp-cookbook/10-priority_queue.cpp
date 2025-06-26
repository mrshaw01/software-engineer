#include <iostream>
#include <queue>
#include <vector>

int main() {

    std::priority_queue<int> max_heap;

    max_heap.push(10);
    max_heap.push(5);
    max_heap.push(20);
    max_heap.push(15);

    std::cout << "Max-heap top: " << max_heap.top() << std::endl;

    std::cout << "Elements popped in descending order:" << std::endl;
    while (!max_heap.empty()) {
        std::cout << max_heap.top() << " ";
        max_heap.pop();
    }
    std::cout << std::endl;

    std::priority_queue<int, std::vector<int>, std::greater<int>> min_heap;

    min_heap.push(10);
    min_heap.push(5);
    min_heap.push(20);
    min_heap.push(15);

    std::cout << "Min-heap top: " << min_heap.top() << std::endl;

    std::cout << "Elements popped in ascending order:" << std::endl;
    while (!min_heap.empty()) {
        std::cout << min_heap.top() << " ";
        min_heap.pop();
    }
    std::cout << std::endl;

    return 0;
}

/*
Max-heap top: 20
Elements popped in descending order:
20 15 10 5
Min-heap top: 5
Elements popped in ascending order:
5 10 15 20
*/
