reply = input("Do you need the answer? (y/n): ")
required = reply.lower() in ("y", "yes")


def the_answer(self, *args):
    return 42


def augment_answer(cls):
    if required:
        cls.the_answer = the_answer
    return cls


@augment_answer
class Philosopher1:
    pass


@augment_answer
class Philosopher2:
    pass


plato = Philosopher1()
aristotle = Philosopher2()

if required:
    print(plato.the_answer())
    print(aristotle.the_answer())
else:
    print("The silence of the philosophers")
