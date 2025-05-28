class RunningAverage:
    def __init__(self):
        self.numbers = []

    def add_number(self, number):
        self.numbers.append(number)

    def __call__(self):
        return sum(self.numbers) / len(self.numbers) if self.numbers else 0

    def reset(self):
        self.numbers = []


average = RunningAverage()
print("Init average:", average())
for x in [3, 5, 12, 9, 1]:
    average.add_number(x)
    print("Average:", average())
average.reset()
for x in [3.1, 19.8, 3]:
    average.add_number(x)
    print("Average:", average())
