my_dict = {"name": "Max", "age": 28, "city": "New York"}

# Add or update
my_dict["email"] = "max@xyz.com"
print(my_dict)

my_dict["email"] = "coolmax@xyz.com"
print(my_dict)

# Delete items
del my_dict["email"]
print(my_dict)

popped_value = my_dict.pop("age")
print("Popped:", popped_value)

popped_item = my_dict.popitem()
print("Popped item:", popped_item)

print(my_dict)
