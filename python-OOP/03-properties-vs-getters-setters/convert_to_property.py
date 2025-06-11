"""Migrate public attribute to property without interface break."""


class OurClass:

    def __init__(self, a):
        self.OurAtt = a

    @property
    def OurAtt(self):
        return self.__OurAtt

    @OurAtt.setter
    def OurAtt(self, val):
        if val < 0:
            self.__OurAtt = 0
        elif val > 1000:
            self.__OurAtt = 1000
        else:
            self.__OurAtt = val


x = OurClass(10)
print(x.OurAtt)

x.OurAtt = 2000  # This will set OurAtt to 1000 due to the setter logic
print(x.OurAtt)  # Output will be 1000

x.OurAtt = -5  # This will set OurAtt to 0 due to the setter logic
print(x.OurAtt)  # Output will be 0

print(x.__dict__)  # This will show the internal state of the object
