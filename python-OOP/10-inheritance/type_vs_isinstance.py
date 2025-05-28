class A:
    pass


class B(A):
    pass


class C(B):
    pass


x = C()

print(isinstance(x, A))  # True
print(type(x) == A)  # False
