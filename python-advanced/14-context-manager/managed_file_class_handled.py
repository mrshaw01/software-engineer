"""Custom context manager that suppresses exceptions."""


class ManagedFile:
    def __init__(self, filename):
        print("init", filename)
        self.filename = filename

    def __enter__(self):
        print("enter")
        self.file = open(self.filename, "w")
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        if exc_type:
            print("Exception has been handled")
        print("exit")
        return True


with ManagedFile("notes2.txt") as f:
    print("doing stuff...")
    f.write("some todo...")
    f.do_something()

print("continuing...")
