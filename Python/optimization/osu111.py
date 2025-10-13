def golden_ratio_recursive(n, prev_approx=1.0, iteration=1, tolerance=1e-10):
    """
    Compute the golden ratio using a recursive approach and print each iteration.

    Parameters:
    - n: The total number of iterations.
    - prev_approx: The approximation from the previous iteration.
    - iteration: The current iteration number.
    - tolerance: The precision of the result.

    Returns:
    - The final approximation of the golden ratio.
    """
    # Compute the current approximation
    current_approx = 1 + 1 / prev_approx

    # Print the current iteration and approximation
    print(f"Iteration {iteration}: {current_approx:.16f}")

    # Check if the result is within the desired tolerance or max iterations reached
    if iteration >= n:
        return current_approx
    else:
        # Recursive call with updated values
        return golden_ratio_recursive(n, current_approx, iteration + 1, tolerance)


# Number of iterations
iterations = 20

# Compute the golden ratio and print each iteration
phi = golden_ratio_recursive(iterations)
print(f"\nApproximated Golden Ratio after {iterations} iterations: {phi:.16f}")
