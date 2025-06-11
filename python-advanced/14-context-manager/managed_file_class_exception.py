"""Custom context manager that logs exceptions but re-raises them."""


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
        print("exc:", exc_type, exc_val)
        print("exit")


# Normal case
with ManagedFile("notes.txt") as f:
    print("doing stuff...")
    f.write("some todo...")

# With an error
with ManagedFile("notes2.txt") as f:
    print("doing stuff...")
    f.write("some todo...")
    f.do_something()
