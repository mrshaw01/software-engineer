from functools import reduce

a = [1, 2, 3, 4]

product_a = reduce(lambda x, y: x * y, a)
sum_a = reduce(lambda x, y: x + y, a)

print("Product:", product_a)
print("Sum:", sum_a)
