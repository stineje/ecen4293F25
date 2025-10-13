def generate_fibonacci(n):
    # Initialize the first two Fibonacci numbers
    fibonacci_sequence = [0, 1]

    # Generate the Fibonacci sequence up to n elements
    for i in range(2, n):
        next_number = fibonacci_sequence[-1] + fibonacci_sequence[-2]
        fibonacci_sequence.append(next_number)

    return fibonacci_sequence


# Example: Generate a Fibonacci sequence with 10 elements
n = 10
fibonacci_list = generate_fibonacci(n)
print(fibonacci_list)
