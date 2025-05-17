"""Demonstrates basic generator behavior using `yield` and `next()`."""


def countdown(num):
    print("Starting")
    while num > 0:
        yield num
        num -= 1


gen = countdown(3)

print(next(gen))  # Starts and yields 3
print(next(gen))  # Yields 2
print(next(gen))  # Yields 1
# print(next(gen))  # Uncomment to see StopIteration

print("\nUsing for loop:")
for value in countdown(3):
    print(value)

print("Sum:", sum(countdown(3)))
print("Sorted:", sorted(countdown(3)))
