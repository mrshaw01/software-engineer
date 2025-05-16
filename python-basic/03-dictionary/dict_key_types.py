# Numeric keys
my_dict = {3: 9, 6: 36, 9: 81}
print(my_dict[3], my_dict[6], my_dict[9])

# Tuple as key
my_tuple = (8, 7)
my_dict = {my_tuple: 15}
print(my_dict[my_tuple])

# Invalid: list as key (will raise error)
# my_list = [8, 7]
# my_dict = {my_list: 15}
