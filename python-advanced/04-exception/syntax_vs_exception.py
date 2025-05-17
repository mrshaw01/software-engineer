# SyntaxError example (won't run at all if uncommented)
# a = 5 print(a)

# TypeError (runtime exception)
try:
    a = 5 + "10"
except TypeError as e:
    print("Caught TypeError:", e)
