a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print("Slice [1:3]:", a[1:3])
print("Slice [2:]:", a[2:])
print("Slice [:3]:", a[:3])
a[0:3] = [0]
print("Replace first 3:", a)
print("Every second element:", a[::2])
print("Reversed list:", a[::-1])
