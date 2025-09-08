from timeit import timeit

code1 = """
def calculate_gopokes(age):
    if age <= 0:
        raise ValueError("Age cannot be 0 or less.")
    return 10 / age

try:
    calculate_gopokes(-1)
except ValueError as error:
    pass
"""

code2 = """
def calculate_gopokes(age):
    if age <= 0:
        return None
    return 10 / age

pokes = calculate_gopokes(-1)
if pokes == None:
    pass
    
"""

print("Execution time = ", timeit(code1, number=10000))
print("Execution time = ", timeit(code2, number=10000))
