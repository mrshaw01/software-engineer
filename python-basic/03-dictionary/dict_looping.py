my_dict = {"name": "Max", "age": 28, "city": "New York"}

for key in my_dict:
    print(key, my_dict[key])

for key in my_dict.keys():
    print("Key:", key)

for value in my_dict.values():
    print("Value:", value)

for key, value in my_dict.items():
    print(f"{key}: {value}")
