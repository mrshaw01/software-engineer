from abc import ABC, abstractmethod


class AbstractClassExample(ABC):
    @abstractmethod
    def do_something(self):
        print("Some implementation!")


class AnotherSubclass(AbstractClassExample):
    def do_something(self):
        super().do_something()
        print("The enrichment from AnotherSubclass")


x = AnotherSubclass()
x.do_something()

# Output:
# Some implementation!
# The enrichment from AnotherSubclass
