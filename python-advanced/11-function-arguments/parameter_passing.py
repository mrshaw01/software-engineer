"""Parameter passing: value vs. reference semantics."""


def foo_immutable(x):
    x = 5


def foo_mutable(a_list):
    a_list.append(4)


def foo_mutable_reassign(a_list):
    a_list = [50, 60, 70]
    a_list.append(50)


def foo_aug_assign(a_list):
    a_list += [4, 5]


def bar_rebind(a_list):
    a_list = a_list + [4, 5]


var = 10
print("var before foo_immutable():", var)
foo_immutable(var)
print("var after foo_immutable():", var)

my_list = [1, 2, 3]
print("my_list before foo_mutable():", my_list)
foo_mutable(my_list)
print("my_list after foo_mutable():", my_list)

my_list = [1, 2, "Max"]
print("my_list before reassign:", my_list)
foo_mutable_reassign(my_list)
print("my_list after reassign:", my_list)

my_list = [1, 2, 3]
print("my_list before += :", my_list)
foo_aug_assign(my_list)
print("my_list after += :", my_list)

my_list = [1, 2, 3]
print("my_list before + :", my_list)
bar_rebind(my_list)
print("my_list after + :", my_list)
