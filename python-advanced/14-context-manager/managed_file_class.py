"""Custom context manager using a class."""


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
        print("exit")


with ManagedFile("notes.txt") as f:
    print("doing stuff...")
    f.write("some todo...")
