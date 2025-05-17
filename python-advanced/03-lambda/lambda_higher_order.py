def myfunc(n):
    return lambda x: x * n


doubler = myfunc(2)
tripler = myfunc(3)

print(doubler(6))
print(tripler(6))
