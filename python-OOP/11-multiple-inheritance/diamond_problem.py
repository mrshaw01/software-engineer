class A:
    def m(self):
        print("m of A")


class B(A):
    def m(self):
        print("m of B")
        super().m()


class C(A):
    def m(self):
        print("m of C")
        super().m()


class D(B, C):
    def m(self):
        print("m of D")
        super().m()


x = D()
x.m()

print("MRO of D:", [cls.__name__ for cls in D.mro()])
print(D.__mro__)
