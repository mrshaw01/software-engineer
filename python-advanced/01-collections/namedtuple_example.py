from collections import namedtuple

Point = namedtuple("Point", "x, y")
pt = Point(1, -4)
print(pt)
print(pt._fields)
print(type(pt))
print(pt.x, pt.y)

Person = namedtuple("Person", "name, age")
friend = Person(name="Tom", age=25)
print(friend.name, friend.age)
