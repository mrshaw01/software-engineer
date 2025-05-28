class A:
    pass


a = A()
a.x = 66
a.y = "dynamically created attribute"

print("Attributes in a:", a.__dict__)
