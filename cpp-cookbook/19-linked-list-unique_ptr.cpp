#include <iostream>
#include <memory>

struct ListNode {
    int value;
    std::unique_ptr<ListNode> next;

    ListNode(int x) : value(x), next(nullptr) {}
};

std::unique_ptr<ListNode> reverseList(std::unique_ptr<ListNode> head) {
    std::unique_ptr<ListNode> prev = nullptr;
    while (head) {
        auto next = std::move(head->next);
        head->next = std::move(prev);
        prev = std::move(head);
        head = std::move(next);
    }
    return prev;
}

void printList(const std::unique_ptr<ListNode> &head) {
    const ListNode *curr = head.get();
    while (curr) {
        std::cout << curr->value << " -> ";
        curr = curr->next.get();
    }
    std::cout << "nullptr" << std::endl;
}

int main() {
    auto head = std::make_unique<ListNode>(1);
    head->next = std::make_unique<ListNode>(2);
    head->next->next = std::make_unique<ListNode>(3);
    head->next->next->next = std::make_unique<ListNode>(4);

    std::cout << "Original list: ";
    printList(head);

    head = reverseList(std::move(head));

    std::cout << "Reversed list: ";
    printList(head);

    return 0;
}
