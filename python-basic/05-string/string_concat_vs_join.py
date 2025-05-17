from timeit import default_timer as timer

my_list = ["a"] * 1000000

# Bad: using +
start = timer()
a = ""
for i in my_list:
    a += i
end = timer()
print("Concatenate with + : %.5f" % (end - start))

# Good: using join
start = timer()
a = "".join(my_list)
end = timer()
print("Concatenate with join(): %.5f" % (end - start))
