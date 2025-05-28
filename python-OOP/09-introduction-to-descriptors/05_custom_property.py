class Property:
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        self.__doc__ = doc or (fget.__doc__ if fget else None)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.fget(obj)

    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError("Can't set attribute")
        self.fset(obj, value)

    def setter(self, fset):
        return Property(self.fget, fset, self.fdel, self.__doc__)


class A:
    def __init__(self, prop):
        self.prop = prop

    @Property
    def prop(self):
        return self.__prop

    @prop.setter
    def prop(self, value):
        self.__prop = value


x = A("Python")
print(x.prop)
x.prop = "Descriptors"
print(x.prop)
