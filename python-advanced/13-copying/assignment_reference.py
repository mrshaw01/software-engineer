"""
Assignment reference — creates alias to the same object.
"""

list_a = [1, 2, 3, 4, 5]
list_b = list_a  # same reference

list_a[0] = -10

print("list_a:", list_a)
print("list_b:", list_b)
