def sumdemo():
    s = 0
    for i in range(10000):
        s = s + 1e-4
    return s


print(sumdemo())
