import numpy as np
import matplotlib.pyplot as plt


def r2_score(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1.0 - ss_res/ss_tot if ss_tot != 0 else 0.0


def fit_linear(x, y):
    # y = a + b x
    b, a = np.polyfit(x, y, 1)  # returns [b, a]
    yhat = a + b*x
    return {"name": "Linear", "params": (a, b), "yhat": yhat, "r2": r2_score(y, yhat),
            "eq": f"y = {a:.4g} + {b:.4g} x"}


def fit_exponential(x, y):
    # y = A * exp(B x)  -> ln(y) = ln A + B x
    mask = y > 0
    x2, y2 = x[mask], y[mask]
    if len(x2) < 2:
        return {"name": "Exponential", "params": None, "yhat": np.full_like(y, np.nan),
                "r2": -np.inf, "eq": "(insufficient positive y)"}
    B, lnA = np.polyfit(x2, np.log(y2), 1)
    A = np.exp(lnA)
    yhat = A * np.exp(B*x)
    return {"name": "Exponential", "params": (A, B), "yhat": yhat, "r2": r2_score(y[mask], yhat[mask]),
            "eq": f"y = {A:.4g} e^({B:.4g} x)"}


def fit_power(x, y):
    # y = A * x^B  -> ln(y) = ln A + B ln x
    mask = (x > 0) & (y > 0)
    x2, y2 = x[mask], y[mask]
    if len(x2) < 2:
        return {"name": "Power", "params": None, "yhat": np.full_like(y, np.nan),
                "r2": -np.inf, "eq": "(x>0 and y>0 required)"}
    B, lnA = np.polyfit(np.log(x2), np.log(y2), 1)
    A = np.exp(lnA)
    yhat = A * x**B
    return {"name": "Power", "params": (A, B), "yhat": yhat, "r2": r2_score(y[mask], yhat[mask]),
            "eq": f"y = {A:.4g} x^{B:.4g}"}


def fit_logarithmic(x, y):
    # y = a + b ln x
    mask = x > 0
    x2, y2 = x[mask], y[mask]
    if len(x2) < 2:
        return {"name": "Logarithmic", "params": None, "yhat": np.full_like(y, np.nan),
                "r2": -np.inf, "eq": "(x>0 required)"}
    b, a = np.polyfit(np.log(x2), y2, 1)
    yhat = a + b*np.log(x)
    return {"name": "Logarithmic", "params": (a, b), "yhat": yhat, "r2": r2_score(y[mask], yhat[mask]),
            "eq": f"y = {a:.4g} + {b:.4g} ln x"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = np.linspace(1, 10, 25)
    # Try swapping the generator to test different truths
    y = 5*np.exp(0.3*x) + rng.normal(0, 10, len(x))     # exponential-ish
    # y = 3*x**1.7 + rng.normal(0, 5, len(x))           # power-ish
    # y = 2 + 4*x + rng.normal(0, 5, len(x))            # linear-ish
    # y = 10 + 2*np.log(x) + rng.normal(0, 0.8, len(x)) # log-ish

    fits = [
        fit_linear(x, y),
        fit_exponential(x, y),
        fit_power(x, y),
        fit_logarithmic(x, y),
    ]

    # Pick best by R^2
    best = max(fits, key=lambda d: d["r2"])

    # Print summary
    for f in fits:
        print(f'{f["name"]:<12} R^2 = {f["r2"]:.4f}   {f["eq"]}')

    print(f"\nBest model by R^2: {best['name']}  ->  {best['eq']}")

    # Plot + save
    plt.figure()
    plt.scatter(x, y, label="Data")
    for f in fits:
        if np.isfinite(f["r2"]):
            plt.plot(x, f["yhat"], label=f'{f["name"]} (R²={f["r2"]:.3f})')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Least-Squares Model Comparison (NumPy only)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ls_model_comparison.png", dpi=300)
    plt.show()
