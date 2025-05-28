class Robot:
    counter = 0

    def __init__(self, name):
        self.name = name

    def sayHello(self):
        return "Hi, I am " + self.name


def Rob_init(self, name):
    self.name = name


Robot2 = type("Robot2", (), {"counter": 0, "__init__": Rob_init, "sayHello": lambda self: "Hi, I am " + self.name})

x = Robot2("Marvin")
y = Robot("Marvin")

print("Robot2:", x.name, "|", x.sayHello())
print("Robot :", y.name, "|", y.sayHello())
print("x.__dict__:", x.__dict__)
print("y.__dict__:", y.__dict__)
