a = [1, 2, 3, 4, 5, 6]
b = list(map(lambda x: x * 2, a))
c = [x * 2 for x in a]

print("Map result:", b)
print("List comp result:", c)
