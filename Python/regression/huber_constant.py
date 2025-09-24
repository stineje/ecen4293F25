import math

# --- Standard normal PDF and CDF ---
def phi(x):
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def Phi(x):
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


# --- Asymptotic relative efficiency (ARE) of Huber estimator ---
def ARE(k):
    """
    Efficiency under standard normal distribution.
    Formula:
      ARE(k) = (E[psi'(U)])^2 / E[psi(U)^2]
    with closed forms for the Huber psi.
    """
    # Probability inside cutoff
    P_in = 2 * Phi(k) - 1.0
    # E[U^2 1{|U| <= k}]
    Eu2_in = (2 * Phi(k) - 1.0) - 2.0 * k * phi(k)
    # Probability outside cutoff
    P_out = 2.0 * (1.0 - Phi(k))

    numerator = P_in ** 2
    denominator = Eu2_in + (k ** 2) * P_out
    return numerator / denominator


# --- Solve for k with bisection ---
def solve_k_for_eff(target=0.95, lo=0.1, hi=5.0, tol=1e-10, maxit=200):
    """
    Find k such that ARE(k) = target.
    Bisection method: assumes ARE(k) is increasing in k.
    """
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        f_mid = ARE(mid) - target
        if abs(f_mid) < tol:
            return mid
        if f_mid > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    targets = [0.85, 0.90, 0.95, 0.975, 0.99]
    print("Huber tuning constants (k) for given target efficiencies:")
    for t in targets:
        k_star = solve_k_for_eff(t)
        print(f"  {t*100:5.1f}% efficiency -> k ≈ {k_star:.9f}")
