class Robot:
    def __init__(self, name, build_year):
        self.__name = name
        self._build_year = build_year
        self.type = "Android"

    def get_name(self):
        return self.__name

    def get_build_year(self):
        return self._build_year


x = Robot("Marvin", 1979)
print(x.type)
print(x._build_year)
print(x.get_name())
