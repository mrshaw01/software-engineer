from weakref import WeakKeyDictionary


class Voter:
    required_age = 18

    def __init__(self):
        self.age = WeakKeyDictionary()

    def __get__(self, instance, objtype):
        return self.age.get(instance)

    def __set__(self, instance, new_age):
        if new_age < Voter.required_age:
            raise Exception(f"{instance.name} is not old enough to vote in Germany")
        self.age[instance] = new_age
        print(f"{instance.name} can vote in Germany")

    def __delete__(self, instance):
        del self.age[instance]


class Person:
    voter_age = Voter()

    def __init__(self, name, age):
        self.name = name
        self.voter_age = age


p1 = Person("Ben", 23)
p2 = Person("Emilia", 22)
print(p1.voter_age)
print(p2.voter_age)
