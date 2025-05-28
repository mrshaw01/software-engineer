class S:
    __slots__ = ["val"]

    def __init__(self, v):
        self.val = v


x = S(42)
print("x.val =", x.val)

try:
    x.new = "not allowed"
except AttributeError as e:
    print("Error when assigning new attribute to S:", e)
