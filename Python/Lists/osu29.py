from collections import deque
x = deque([])
x.append(1)
x.append(2)
x.append(3)
x.popleft()
print(x)
if not x:
    print("empty")
