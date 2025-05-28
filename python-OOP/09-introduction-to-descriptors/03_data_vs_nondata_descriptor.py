class NonDataDescriptor:
    def __get__(self, instance, owner):
        return "Non-data descriptor"


class DataDescriptor:
    def __get__(self, instance, owner):
        return "Data descriptor"

    def __set__(self, instance, value):
        print("Setting data descriptor")


class MyClass:
    nd = NonDataDescriptor()
    dd = DataDescriptor()


obj = MyClass()
obj.nd = "Overwritten"
print("nd:", obj.nd)  # instance var overrides non-data
obj.dd = "Try set"
print("dd:", obj.dd)  # data descriptor overrides instance
