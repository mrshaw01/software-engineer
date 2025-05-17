from collections import Counter

a = "aaaaabbbbcccdde"
my_counter = Counter(a)
print(my_counter)
print(my_counter.items())
print(my_counter.keys())
print(my_counter.values())

my_list = [0, 1, 0, 1, 2, 1, 1, 3, 2, 3, 2, 4]
my_counter = Counter(my_list)
print(my_counter)

print("Most common:", my_counter.most_common(1))
print("All elements:", list(my_counter.elements()))
