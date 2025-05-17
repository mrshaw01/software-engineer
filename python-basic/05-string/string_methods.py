my_string = "     Hello World "
my_string = my_string.strip()
print(my_string)

print(len(my_string))
print(my_string.upper())
print(my_string.lower())
print("hello".startswith("he"))
print("hello".endswith("llo"))
print("Hello".find("o"))
print("Hello".count("e"))

message = "Hello World"
new_message = message.replace("World", "Universe")
print(new_message)

# Splitting
print("how are you doing".split())
print("one,two,three".split(","))

# Joining
my_list = ["How", "are", "you", "doing"]
print(" ".join(my_list))
