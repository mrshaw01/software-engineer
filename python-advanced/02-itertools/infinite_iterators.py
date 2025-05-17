from itertools import count, cycle, repeat

# count
for i in count(10):
    print(i)
    if i >= 13:
        break

print("")

# cycle
sum = 0
for i in cycle([1, 2, 3]):
    print(i)
    sum += i
    if sum >= 12:
        break

print("")

# repeat
for i in repeat("A", 3):
    print(i)
