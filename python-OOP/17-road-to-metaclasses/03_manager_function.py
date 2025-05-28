reply = input("Do you need the answer? (y/n): ")
required = reply.lower() in ("y", "yes")


def the_answer(self, *args):
    return 42


def augment_answer(cls):
    if required:
        cls.the_answer = the_answer
    return cls


class Philosopher1:
    pass


augment_answer(Philosopher1)


class Philosopher2:
    pass


augment_answer(Philosopher2)

plato = Philosopher1()
aristotle = Philosopher2()

if required:
    print(plato.the_answer())
    print(aristotle.the_answer())
else:
    print("The silence of the philosophers")
