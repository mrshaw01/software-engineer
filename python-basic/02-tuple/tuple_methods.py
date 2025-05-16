my_tuple = ("a", "p", "p", "l", "e")

print(len(my_tuple))
print(my_tuple.count("p"))
print(my_tuple.index("l"))

# Repetition
repeated = ("a", "b") * 5
print(repeated)

# Concatenation
combined = (1, 2, 3) + (4, 5, 6)
print(combined)

# Conversion
my_list = ["a", "b", "c"]
list_to_tuple = tuple(my_list)
print(list_to_tuple)

tuple_to_list = list(list_to_tuple)
print(tuple_to_list)

string_to_tuple = tuple("Hello")
print(string_to_tuple)
