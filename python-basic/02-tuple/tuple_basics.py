tuple_1 = ("Max", 28, "New York")
tuple_2 = "Linda", 25, "Miami"
tuple_3 = (25,)

print(tuple_1)
print(tuple_2)
print(tuple_3)

tuple_4 = tuple([1, 2, 3])
print(tuple_4)

# Access elements
print(tuple_1[0])
print(tuple_1[-1])

# Attempting to modify raises TypeError
try:
    tuple_1[2] = "Boston"
except TypeError as e:
    print("Error:", e)

# Delete tuple
del tuple_2

# Iterate
for item in tuple_1:
    print(item)

# Membership test
if "New York" in tuple_1:
    print("yes")
else:
    print("no")
