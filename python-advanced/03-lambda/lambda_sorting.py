points2D = [(1, 9), (4, 1), (5, -3), (10, 2)]
sorted_by_y = sorted(points2D, key=lambda x: x[1])
print("Sorted by Y:", sorted_by_y)

mylist = [-1, -4, -2, -3, 1, 2, 3, 4]
sorted_by_abs = sorted(mylist, key=lambda x: abs(x))
print("Sorted by abs:", sorted_by_abs)
