"""Verbose reimplementation of property using chatty_property for debugging."""


class chatty_property:

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        print("\n__init__ called with:")
        print(f"fget={fget}, fset={fset}, fdel={fdel}, doc={doc}")
        if doc is None and fget is not None:
            print(f"doc set to docstring of {fget.__name__} method")
            doc = fget.__doc__
        self.__doc__ = doc

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget(obj)

    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.fset(obj, value)

    def __delete__(self, obj):
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        self.fdel(obj)

    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel, self.__doc__)

    def setter(self, fset):
        print(type(self))
        return type(self)(self.fget, fset, self.fdel, self.__doc__)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.__doc__)


class Robot:

    def __init__(self, city):
        self.city = city

    @chatty_property
    def city(self):
        """city attribute of Robot"""
        print("The Property 'city' will be returned now:")
        return self.__city

    @city.setter
    def city(self, city):
        print("'city' will be set")
        self.__city = city


robo = Robot("Berlin")
print("Current city:", robo.city)
robo.city = "Munich"
print("New city:", robo.city)
