"""
Copying custom objects with shallow vs deep copy.
"""

import copy


class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age


class Company:
    def __init__(self, boss, employee):
        self.boss = boss
        self.employee = employee


print("--- Shallow copy ---")
boss = Person("Jane", 55)
employee = Person("Joe", 28)
company = Company(boss, employee)

clone = copy.copy(company)
clone.boss.age = 56

print("Original boss age:", company.boss.age)
print("Cloned boss age:", clone.boss.age)

print("\n--- Deep copy ---")
boss = Person("Jane", 55)
employee = Person("Joe", 28)
company = Company(boss, employee)

clone = copy.deepcopy(company)
clone.boss.age = 60

print("Original boss age:", company.boss.age)
print("Cloned boss age:", clone.boss.age)
