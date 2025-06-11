class Robot:

    def __init__(self, name=None, build_year=None):
        self.name = name
        self.build_year = build_year

    def say_hi(self):
        print(f"Hi, I am {self.name} and I was built in {self.build_year}")

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_build_year(self, year):
        self.build_year = year

    def get_build_year(self):
        return self.build_year


x = Robot("Henry", 2008)
y = Robot()
y.set_name(x.get_name())
x.say_hi()
y.say_hi()
