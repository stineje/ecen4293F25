import numpy as np
import matplotlib.pyplot as plt

# Parameters
V_s = 5.0    # Source voltage (V)
R = 1000.0   # Resistor (Ohms)
I_s = 1e-12  # Saturation current (A)
V_T = 0.025  # Thermal voltage (V)


def f(V):
    """ Diode equation function """
    return V_s - R * I_s * (np.exp(V / V_T) - 1) - V


def f_prime(V):
    """ Derivative of the diode equation function """
    return -R * I_s / V_T * np.exp(V / V_T) - 1


# TODO: implement Newton-Raphson to find V
V = 0.5   # Initial guess (V)
tol = 1e-6
max_iter = 20
for _ in range(max_iter):
    f_val = f(V)
    f_prime_val = f_prime(V)
    if abs(f_prime_val) < 1e-10:  # Prevent division by zero
        print("Derivative too small, stopping Newton-Raphson")
        break
    V_new = V - f_val / f_prime_val
    if abs(V_new - V) < tol:
        print(
            f"Newton-Raphson converged to V = {V_new:.8f} V after {_ + 1} iterations")
        break
    V = V_new
else:
    print("Newton-Raphson did not converge")

# Plot f(V) and f'(V)
V_range = np.linspace(0, 1, 100)
f_values = f(V_range)
f_prime_values = f_prime(V_range)

plt.figure(figsize=(8, 5))
plt.plot(V_range, f_values, 'b-',
         label='f(V) = V_s - R I_s (e^(V/V_T) - 1) - V')
plt.plot(V_range, f_prime_values, 'r--', label="f'(V)")
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.axvline(V, color='g', linestyle='--', label=f'Solution V = {V:.3f} V')
plt.xlabel('Voltage V (V)')
plt.ylabel('f(V) and f\'(V)')
plt.title('Nonlinear Diode Equation and Derivative')
plt.legend()
plt.grid(True)
plt.savefig('diode_plot.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
