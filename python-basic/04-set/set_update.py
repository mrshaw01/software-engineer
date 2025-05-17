setA = {1, 2, 3, 4, 5, 6, 7, 8, 9}
setB = {1, 2, 3, 10, 11, 12}

setA.update(setB)
print("update():", setA)

setA = {1, 2, 3, 4, 5, 6, 7, 8, 9}
setA.intersection_update(setB)
print("intersection_update():", setA)

setA = {1, 2, 3, 4, 5, 6, 7, 8, 9}
setA.difference_update(setB)
print("difference_update():", setA)

setA = {1, 2, 3, 4, 5, 6, 7, 8, 9}
setA.symmetric_difference_update(setB)
print("symmetric_difference_update():", setA)
