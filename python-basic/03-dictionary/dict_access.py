my_dict = {"name": "Max", "age": 28, "city": "New York"}

# Access by key
print(my_dict["name"])

# Handle missing key safely
try:
    print(my_dict["firstname"])
except KeyError:
    print("No key found")
