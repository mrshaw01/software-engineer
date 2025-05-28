class A:
    ca_A = "class attribute of A"

    def __init__(self):
        self.ia_A = "instance attribute of A"


class B(A):
    ca_B = "class attribute of B"

    def __init__(self):
        super().__init__()
        self.ia_B = "instance attribute of B"


x = B()
print(x.ia_B)
print(x.ca_B)
print(x.ia_A)
print(x.ca_A)
try:
    print(x.non_existing)
except AttributeError as e:
    print("AttributeError:", e)
