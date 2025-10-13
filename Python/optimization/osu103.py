from bisect import bisect
import matplotlib.pyplot as plt
import numpy as np


def calculate_salmon_population(years):
    """
    Calculate the number of salmon returning to the creek for each year up to 'years'.

    :param years: The number of years to calculate.
    :return: A list with the number of salmon returning for each year.
    """
    # Initialize a list to store the salmon population for each year
    population = [0]  # S(0) = 0

    # Calculate the salmon population for each year using the recurrence relation
    for n in range(1, years + 1):
        S_n = 1000 + 0.3 * population[n - 1]
        population.append(S_n)

    return population


def plot_population(population):
    """
    Plot the salmon population over the years using matplotlib.

    :param population: A list of salmon populations for each year.
    """
    years = list(range(len(population)))

    plt.figure(figsize=(10, 6))
    plt.plot(years, population, marker='o', linestyle='-',
             color='b', label='Salmon Population')
    plt.title('Salmon Population Over Years')
    plt.xlabel('Year')
    plt.ylabel('Salmon Population')
    plt.grid(True)
    plt.legend()
    plt.show()


# Number of years to calculate
years = 10

# Calculate the salmon population for each year
population = calculate_salmon_population(years)

# Print the results
print("Year\tSalmon Population")
for year, pop in enumerate(population):
    print(f"{year}\t{pop:.2f}")

# Plot the salmon population
plot_population(population)
