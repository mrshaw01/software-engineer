# This is NOT an abstract base class. It doesn't enforce anything.


class AbstractClass:

    def do_something(self):
        pass


class B(AbstractClass):
    pass


a = AbstractClass()  # Works
b = B()  # Also works
print("Instance created without enforcing method implementation.")
