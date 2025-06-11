class FoodSupply:

    def __init__(self, *ingredients):
        self.ingredients = ingredients

    def __call__(self):
        return " ".join(self.ingredients) + " plus delicious spam!"


f = FoodSupply("fish", "rice")
print(f())

g = FoodSupply("vegetables")
print(g())
