"""Counts instances using a class attribute."""


class C:
    counter = 0

    def __init__(self):
        type(self).counter += 1

    def __del__(self):
        type(self).counter -= 1


if __name__ == "__main__":
    x = C()
    print("Instances:", C.counter)
    y = C()
    print("Instances:", C.counter)
    del x
    print("Instances:", C.counter)
    del y
    print("Instances:", C.counter)
