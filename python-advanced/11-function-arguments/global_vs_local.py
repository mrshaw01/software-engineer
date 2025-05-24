"""Demonstrating global vs. local variables."""


def foo1():
    x = number
    print("number in function:", x)


def foo2():
    global number
    number = 3


def foo3():
    number = 3  # local


number = 0
foo1()
print("number before foo2():", number)
foo2()
print("number after foo2():", number)

print("number before foo3():", number)
foo3()
print("number after foo3():", number)
