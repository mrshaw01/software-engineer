class MyClass:
    pass


print("Type of class:", type(MyClass))  # <class 'type'>
print("Type of instance:", type(MyClass()))  # <class '__main__.MyClass'>
print("Is MyClass an instance of type?", isinstance(MyClass, type))  # True
