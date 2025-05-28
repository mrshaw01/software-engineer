def the_answer(question):
    return 42


print("the_answer: ", callable(the_answer))
print("int is callable:", callable(int))
print("list is callable:", callable(list))
print("dict is callable:", callable(dict))
