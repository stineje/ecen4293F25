import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# --- import newtraph from ../roots/newtraph.py ---
HERE = Path(__file__).resolve().parent
ROOTS_DIR = (HERE / ".." / "roots").resolve()
if str(ROOTS_DIR) not in sys.path:
    sys.path.insert(0, str(ROOTS_DIR))

from newtraph import newtraph  # uses: newtraph(f, fp, x0, Ea=1e-7, maxit=30)

V_s = 5.0    # Source voltage (V)
R   = 1000.0 # Resistor (Ohms)
I_s = 1e-12  # Saturation current (A)
V_T = 0.025  # Thermal voltage (V)

def f(V):
    """Diode KCL equation f(V) = 0 at the node across the diode."""
    return V_s - R * I_s * (np.exp(V / V_T) - 1.0) - V

def f_prime(V):
    """df/dV."""
    return -R * I_s / V_T * np.exp(V / V_T) - 1.0

# Solve with our Newton–Raphson wrapper
x0 = 0.5
V_sol, f_at_sol, ea, niter = newtraph(f, f_prime, x0, Ea=1e-6, maxit=20)

print(f"Newton-Raphson (newtraph) converged in {niter} iterations")
print(f"V* = {V_sol:.6f} V")
print(f"f(V*) = {f_at_sol:.3e}")
print(f"relative error ea = {ea:.3e}")

# Plot f(V) and f'(V)
V_range = np.linspace(0, 1.0, 400)
f_values = f(V_range)
fp_values = f_prime(V_range)

plt.figure(figsize=(8, 5))
plt.plot(V_range, f_values, label="f(V) = V_s - R I_s (e^{V/V_T}-1) - V")
plt.plot(V_range, fp_values, linestyle="--", label="f'(V)")
plt.axhline(0.0, linewidth=1, linestyle=":", color="k")
plt.axvline(V_sol, linewidth=1, linestyle="--", label=f"V* = {V_sol:.6f} V")
plt.xlabel("Voltage V (V)")
plt.ylabel("f(V), f'(V)")
plt.title("Diode nonlinear equation and derivative")
plt.grid(True)
plt.legend()
plt.savefig("diode_plot.png", dpi=300, bbox_inches="tight")
plt.show()
