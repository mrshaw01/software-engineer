from metaclass_counter import FuncCallCounter


class A(metaclass=FuncCallCounter):

    def foo(self):
        pass

    def bar(self):
        pass


if __name__ == "__main__":
    a = A()
    print(a.foo.calls, a.bar.calls)  # 0 0
    a.foo()
    print(a.foo.calls, a.bar.calls)  # 1 0
    a.foo()
    a.bar()
    print(a.foo.calls, a.bar.calls)  # 2 1
