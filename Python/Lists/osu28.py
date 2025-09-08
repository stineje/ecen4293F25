x = []
x.append(1)
x.append(2)
x.append(3)
print(x)
x.pop()
print(x)
# be careful with stack (Falsy values for [])
if not x:
    print("Your stack is empty")
