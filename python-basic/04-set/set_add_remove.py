my_set = set()

my_set.add(42)
my_set.add(True)
my_set.add("Hello")
print(my_set)

# Add duplicate
my_set.add(42)
print(my_set)

# Remove
my_set = {"apple", "banana", "cherry"}
my_set.remove("apple")
print(my_set)

# my_set.remove("orange")  # KeyError

# Discard
my_set.discard("cherry")
my_set.discard("blueberry")
print(my_set)

# Clear
my_set.clear()
print(my_set)

# Pop
a = {True, 2, False, "hi", "hello"}
print(a.pop())  # Random element
print(a.pop())  # Random element
print(a.pop())  # Random element
print(a.pop())  # Random element
print(a)
