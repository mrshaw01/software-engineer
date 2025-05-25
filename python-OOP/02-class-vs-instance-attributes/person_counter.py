"""Real-world classmethod example to count people."""


class Person:
    total_people = 0

    def __init__(self, name):
        self.name = name
        Person.total_people += 1

    @classmethod
    def display_total_people(cls):
        print("Total number of people:", cls.total_people)


person1 = Person("Alice")
person2 = Person("Bob")
person3 = Person("Charlie")

Person.display_total_people()
