reply = input("Do you need the answer? (y/n): ")
required = reply.lower() in ("y", "yes")


def the_answer(self, *args):
    return 42


class Philosopher1:
    pass


class Philosopher2:
    pass


class Philosopher3:
    pass


if required:
    Philosopher1.the_answer = the_answer
    Philosopher2.the_answer = the_answer
    Philosopher3.the_answer = the_answer

plato = Philosopher1()
aristotle = Philosopher2()

if required:
    print(plato.the_answer())
    print(aristotle.the_answer())
else:
    print("The silence of the philosophers")
