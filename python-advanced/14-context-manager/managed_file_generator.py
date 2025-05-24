"""Context manager implemented with @contextmanager and yield."""

from contextlib import contextmanager


@contextmanager
def open_managed_file(filename):
    f = open(filename, "w")
    try:
        yield f
    finally:
        f.close()


with open_managed_file("notes.txt") as f:
    f.write("some todo...")
