"""Unpack containers using * to split sequences"""

data = (1, 2, 3, 4, 5, 6, 7)

*start, last = data
print(start, last)

first, *end = data
print(first, end)

first, *middle, last = data
print(first, middle, last)
