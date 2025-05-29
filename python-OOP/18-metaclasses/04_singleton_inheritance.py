class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance


class SingletonClass(Singleton):
    pass


class RegularClass:
    pass


x = SingletonClass()
y = SingletonClass()
print(x == y)  # True

x = RegularClass()
y = RegularClass()
print(x == y)  # False
