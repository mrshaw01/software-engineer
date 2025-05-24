"""
Shallow copy with nested lists — inner objects are shared.
"""

import copy

list_a = [[1, 2], [3, 4]]
list_b = copy.copy(list_a)

list_a[0][0] = -100

print("list_a:", list_a)
print("list_b:", list_b)
