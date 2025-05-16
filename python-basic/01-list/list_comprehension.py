a = [1, 2, 3, 4, 5, 6, 7, 8]
b = [x * x for x in a]
print("Squares:", b)

even = [x for x in a if x % 2 == 0]
print("Evens:", even)
