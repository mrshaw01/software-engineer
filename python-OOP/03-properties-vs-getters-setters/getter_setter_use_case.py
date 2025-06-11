"""Use case: getter/setter with extra logic/arguments."""


class Person:

    def __init__(self, name, height):
        self.name = name
        self._height = height

    def get_height(self):
        return self._height

    def set_height(self, value, validate=True):
        if validate and not (150 <= value <= 200):
            raise ValueError("Height must be between 150 and 200 cm.")
        self._height = value


person = Person("Alice", 170)
person.set_height(175)
print(person.get_height())

try:
    person.set_height(210)
except ValueError as e:
    print(e)

person.set_height(210, validate=False)
print(person.get_height())
