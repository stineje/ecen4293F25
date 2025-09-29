import csv
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def linear_model(x, m, c):
    return m * x + c

def read_csv_data(file_path):
    data = []
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8-sig') as file:
            csv_reader = csv.reader(file)
            # Skip header row if present
            try:
                next(csv_reader)
            except StopIteration:
                print("CSV file is empty.")
                return []

            for i, row in enumerate(csv_reader, start=2):
                try:
                    yearID = int(row[0])
                    teamID = row[1]
                    lgID = row[2]
                    playerID = row[3]
                    salary = float(row[4])
                    data.append({
                        'Year': yearID,
                        'Team ID': teamID,
                        'League': lgID,
                        'Player ID': playerID,
                        'Salary [$]': salary
                    })
                except (ValueError, IndexError):
                    print(f"Skipping bad row {i}: {row}")
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred while reading: {e}")

    return data

# Usage
data = read_csv_data('Salaries.csv')

def S(a):
    abar = np.mean(a)
    adev = a - abar
    return np.sum(adev**2)

if data:
    salary = [entry['Salary [$]'] for entry in data]
    yearID = [entry['Year'] for entry in data]

    mlbbar = np.mean(salary)
    print(f"mean estimate = ${mlbbar:,.2f}")
    print(f"sample median = ${np.median(salary):,.2f}")
    mlbmode = stats.mode(salary, axis=None)
    print(f"sample mode = ${mlbmode.mode} (count: {mlbmode.count})")
    print(f"sample variance = ${np.var(salary, ddof=1):,.2f}")
    mlbstd = np.std(salary, ddof=1)
    print(f"sample standard deviation = ${mlbstd:,.2f}")
    print(f"coefficient of variation = {mlbstd/mlbbar*100:5.3f} %")
    print(f"total corrected sum of squares = ${S(salary):,.2f}")
    MADmlb = stats.median_abs_deviation(salary)
    print(f"MAD (robust sigma estimate) = ${MADmlb/0.6745:,.2f}")

    # curve_fit may fail -> wrap in try
    try:
        p_opt, p_cov = curve_fit(linear_model, yearID, salary)
        m_fit, c_fit = p_opt
        y_fit = linear_model(np.array(yearID), m_fit, c_fit)
        residuals = salary - y_fit

        slope_error, intercept_error = np.sqrt(np.diag(p_cov))
        ss_res = np.sum((salary - y_fit) ** 2)
        ss_tot = np.sum((salary - np.mean(salary)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f"Fitted slope: ${m_fit:,.2f} ± ${slope_error:,.2f}")
        print(f"Fitted intercept: ${c_fit:,.4f} ± ${intercept_error:,.4f}")
        print(f"Residual std dev: ${np.std(residuals):,.2f}")
        print(f"R-squared: {r_squared:.6f}")

        # Plot
        plt.scatter(yearID, salary, label='Data points')
        plt.plot(yearID, y_fit, 'r', label=f'Fit: $y = {m_fit:,.1f}x + {c_fit:,.1f}$')
        plt.xlabel('Year')
        plt.ylabel('Salary ($)')
        plt.title('Least Squares Fit')
        plt.legend()
        plt.savefig('read-salaries.png')
        plt.show()

    except Exception as e:
        print(f"Curve fitting failed: {e}")
        
