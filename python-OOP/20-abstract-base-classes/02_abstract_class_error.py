from abc import ABC
from abc import abstractmethod


class AbstractClassExample(ABC):

    def __init__(self, value):
        self.value = value

    @abstractmethod
    def do_something(self):
        pass


class DoAdd42(AbstractClassExample):
    pass


# This will raise TypeError: Can't instantiate abstract class
x = DoAdd42(4)
