a = "Hello {0} and {1}".format("Bob", "Tom")
print(a)

a = "Hello {} and {}".format("Bob", "Tom")
print(a)

a = "The integer value is {}".format(2)
print(a)

a = "The float value is {0:.3f}".format(2.1234)
print(a)

a = "The float value is {0:e}".format(2.1234)
print(a)

a = "The binary value is {0:b}".format(2)
print(a)

# Old-style
print("Hello %s and %s" % ("Bob", "Tom"))
val = 10.12345
print("The decimal value is %d" % val)
print("The float value is %f" % val)
print("The float value is %.2f" % val)
