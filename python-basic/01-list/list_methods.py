my_list = ["banana", "cherry", "apple"]
my_list.append("orange")
my_list.insert(1, "blueberry")
print("After append & insert:", my_list)

item = my_list.pop()
print("Popped:", item)

my_list.remove("cherry")
print("After removal:", my_list)

my_list.reverse()
print("Reversed:", my_list)

my_list.sort()
print("Sorted:", my_list)
