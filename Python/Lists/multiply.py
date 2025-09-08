def multiply(*numbers):
    value = 1
    for number in numbers:
        value *= number
    return value


# Use a collection of arguments
print("Start")
# generated via random.org
answer = multiply(20, 37, 33, 73, 40, 65, 93, 41)

# Using str.format() method
# Adding comma between numbers
res = ('{:,}'.format(answer))

# answer = 17,672,934,708,000
print(res)
