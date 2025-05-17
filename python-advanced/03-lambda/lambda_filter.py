a = [1, 2, 3, 4, 5, 6, 7, 8]
b = list(filter(lambda x: x % 2 == 0, a))
c = [x for x in a if x % 2 == 0]

print("Filter result:", b)
print("List comp result:", c)
