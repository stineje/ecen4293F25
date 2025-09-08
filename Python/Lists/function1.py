def make_function(a, b):
    # returns f(x) = a*x + b
    return lambda x: a * x + b


# Example usage
f = make_function(2, 3)   # f(x) = 2x + 3
print(f(5))               # prints 13
print(f(10))              # prints 23
