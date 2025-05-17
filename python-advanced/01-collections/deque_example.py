from collections import deque

d = deque()
d.append("a")
d.append("b")
print(d)

d.appendleft("c")
print(d)

print("Popped:", d.pop())
print("Popleft:", d.popleft())
print("After pops:", d)

d.clear()
print("Cleared:", d)

d = deque(["a", "b", "c", "d"])
d.extend(["e", "f", "g"])
d.extendleft(["h", "i", "j"])
print("Extended both ends:", d)

print("Count 'h':", d.count("h"))

d.rotate(1)
print("Rotate right:", d)

d.rotate(-2)
print("Rotate left:", d)
