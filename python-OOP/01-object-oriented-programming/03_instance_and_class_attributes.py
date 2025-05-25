class Robot:
    pass


x = Robot()
y = Robot()

x.name = "Marvin"
x.build_year = 1979
x.brand = "Boston Dynamics"

y.name = "Caliban"
y.build_year = 1993

Robot.brand = "Kuka"

print(x.__dict__)
print(y.__dict__)
print(Robot.__dict__)
print(x.brand)
print(y.brand)
