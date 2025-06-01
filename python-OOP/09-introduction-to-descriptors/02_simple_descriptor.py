class SimpleDescriptor:
    def __init__(self, value=None):
        self.__set__(self, value)

    def __get__(self, instance, owner):
        print("Getting value:", self.value)
        return self.value

    def __set__(self, instance, value):
        print("Setting value:", value)
        self.value = value


class MyClass:
    x = SimpleDescriptor("green")


m = MyClass()
print(m.x)
m.x = "yellow"
print(m.x)
