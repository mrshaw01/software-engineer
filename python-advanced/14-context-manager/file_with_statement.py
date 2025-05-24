"""Open and write to a file using a context manager."""

with open("notes.txt", "w") as f:
    f.write("some todo...")
