class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class SingletonClass(metaclass=Singleton):
    pass


class RegularClass:
    pass


x = SingletonClass()
y = SingletonClass()
print(x == y)  # True

x = RegularClass()
y = RegularClass()
print(x == y)  # False

print("All Singleton Instances:", Singleton._instances)
print("SingletonClass Instances:", SingletonClass._instances)
