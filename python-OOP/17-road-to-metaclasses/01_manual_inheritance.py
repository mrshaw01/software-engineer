class Answers:

    def the_answer(self, *args):
        return 42


class Philosopher1(Answers):
    pass


class Philosopher2(Answers):
    pass


plato = Philosopher1()
aristotle = Philosopher2()

print(plato.the_answer())  # 42
print(aristotle.the_answer())  # 42
