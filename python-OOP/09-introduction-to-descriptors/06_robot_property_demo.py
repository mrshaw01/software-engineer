class Property:
    def __init__(self, fget=None, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, objtype=None):
        return self.fget(obj)

    def __set__(self, obj, value):
        self.fset(obj, value)

    def setter(self, fset):
        return Property(self.fget, fset)


class Robot:
    def __init__(self, name):
        self.name = name

    @Property
    def name(self):
        return self.__name

    @name.setter
    def name(self, value):
        self.__name = "hi" if value == "hello" else value


x = Robot("Marvin")
print(x.name)
x.name = "Eddie"
print(x.name)
