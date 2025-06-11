def add_class_variable(variable_name, value):

    def decorator(cls):
        setattr(cls, variable_name, value)
        return cls

    return decorator


@add_class_variable("city", "Erlangen")
class MyClass:

    def __init__(self, value):
        self.value = value


obj = MyClass(10)
print(obj.city)  # Output: Erlangen
