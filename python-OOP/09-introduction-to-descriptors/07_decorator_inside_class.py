class A:
    def a(func):
        def wrapper(self, x):
            return 4 * func(self, x)

        return wrapper

    @a
    def b(self, x):
        return x + 1


a = A()
print(a.b(4))
