"""Define property without decorators."""


class P:

    def __init__(self, x):
        self.__set_x(x)

    def __get_x(self):
        return self.__x

    def __set_x(self, x):
        if x < 0:
            self.__x = 0
        elif x > 1000:
            self.__x = 1000
        else:
            self.__x = x

    x = property(__get_x, __set_x)


p1 = P(500)
print(p1.x)

p1 = P(5000)
print(p1.x)

p1 = P(-5000)
print(p1.x)
