class DynPropertyClass:
    def add_property(self, attr):
        def getter(self):
            return getattr(self, f"_{type(self).__name__}__{attr}")

        def setter(self, value):
            setattr(self, f"_{type(self).__name__}__{attr}", value)

        setattr(type(self), attr, property(getter, setter))


x = DynPropertyClass()
x.add_property("name")
x.add_property("city")
x.name = "Henry"
x.city = "Hamburg"
print(x.name, x.city)
print(x.__dict__)
