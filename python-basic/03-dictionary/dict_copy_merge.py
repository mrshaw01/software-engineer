dict_org = {"name": "Max", "age": 28, "city": "New York"}

# Shallow copy (shared reference)
dict_copy = dict_org
dict_copy["name"] = "Lisa"
print("Shared copy:", dict_copy)
print("Original also changed:", dict_org)

# True copy
dict_org = {"name": "Max", "age": 28, "city": "New York"}
dict_copy = dict_org.copy()
dict_copy["name"] = "Lisa"
print("Copied dict:", dict_copy)
print("Original unchanged:", dict_org)

# Merge
dict_a = {"name": "Max", "email": "max@xyz.com"}
dict_b = {"name": "Lisa", "age": 27, "city": "Boston"}
dict_a.update(dict_b)
print("Merged:", dict_a)
