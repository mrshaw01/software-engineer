#include <iostream>

struct ListNode {
    int value;
    ListNode *next;

    ListNode(int x) : value(x), next(nullptr) {}
};

ListNode *reverseList(ListNode *head) {
    ListNode *prev = nullptr;
    ListNode *curr = head;

    while (curr) {
        ListNode *next = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next;
    }

    return prev;
}

void printList(ListNode *head) {
    ListNode *curr = head;
    while (curr) {
        std::cout << curr->value << " -> ";
        curr = curr->next;
    }
    std::cout << "nullptr" << std::endl;
}

void deleteList(ListNode *head) {
    while (head) {
        ListNode *next = head->next;
        delete head;
        head = next;
    }
}

int main() {
    ListNode *head = new ListNode(1);
    head->next = new ListNode(2);
    head->next->next = new ListNode(3);
    head->next->next->next = new ListNode(4);

    std::cout << "Original list: ";
    printList(head);

    head = reverseList(head);

    std::cout << "Reversed list: ";
    printList(head);

    deleteList(head);

    return 0;
}
