my_set = {"apple", "banana", "cherry"}
print(my_set)

my_set_2 = set(["one", "two", "three"])
print(my_set_2)

my_set_3 = set("aaabbbcccdddeeeeeffff")
print(my_set_3)

# Empty set must be created with set(), not {}
a = {}
print(type(a))  # dict
a = set()
print(type(a))  # set
