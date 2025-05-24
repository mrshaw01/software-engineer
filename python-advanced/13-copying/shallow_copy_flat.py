"""
Shallow copy of a flat list.
"""

import copy

list_a = [1, 2, 3, 4, 5]
list_b = copy.copy(list_a)

list_b[0] = -10

print("list_a:", list_a)
print("list_b:", list_b)
