import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Helpers (no sklearn)
# -----------------------


def r2_score(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1.0 - ss_res/ss_tot if ss_tot != 0 else 0.0, ss_res


def aic(n, rss, k):
    # AIC for Gaussian errors with LS fit
    return n * np.log(rss / n) + 2 * k if rss > 0 else -np.inf


def aicc(n, rss, k):
    # Small-sample corrected AIC
    if n - k - 1 <= 0 or rss <= 0:
        return np.inf
    return aic(n, rss, k) + (2 * k * (k + 1)) / (n - k - 1)


def fit_linear(x, y):
    # y = a + b x
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    r2, rss = r2_score(y, yhat)
    return {"name": "Linear", "params": (a, b), "yhat": yhat, "r2": r2, "rss": rss,
            "k": 2, "eq": f"y = {a:.6g} + {b:.6g} x"}


def fit_exponential(x, y):
    # y = A * exp(B x) -> ln y = ln A + B x (require y>0)
    mask = y > 0
    yhat = np.full_like(y, np.nan, dtype=float)
    if np.count_nonzero(mask) < 2:
        return {"name": "Exponential", "params": None, "yhat": yhat, "r2": -np.inf, "rss": np.inf, "k": 2, "eq": "(y>0 required)"}
    B, lnA = np.polyfit(x[mask], np.log(y[mask]), 1)
    A = np.exp(lnA)
    yhat = A * np.exp(B * x)
    r2, rss = r2_score(y[mask], yhat[mask])
    return {"name": "Exponential", "params": (A, B), "yhat": yhat, "r2": r2, "rss": rss,
            "k": 2, "eq": f"y = {A:.6g} e^({B:.6g} x)"}


def fit_power(x, y):
    # y = A * x^B -> ln y = ln A + B ln x (require x>0, y>0)
    mask = (x > 0) & (y > 0)
    yhat = np.full_like(y, np.nan, dtype=float)
    if np.count_nonzero(mask) < 2:
        return {"name": "Power", "params": None, "yhat": yhat, "r2": -np.inf, "rss": np.inf, "k": 2, "eq": "(x>0,y>0 required)"}
    B, lnA = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    A = np.exp(lnA)
    yhat = A * x**B
    r2, rss = r2_score(y[mask], yhat[mask])
    return {"name": "Power", "params": (A, B), "yhat": yhat, "r2": r2, "rss": rss,
            "k": 2, "eq": f"y = {A:.6g} x^{B:.6g}"}


def fit_logarithmic(x, y):
    # y = a + b ln x (require x>0)
    mask = x > 0
    yhat = np.full_like(y, np.nan, dtype=float)
    if np.count_nonzero(mask) < 2:
        return {"name": "Logarithmic", "params": None, "yhat": yhat, "r2": -np.inf, "rss": np.inf, "k": 2, "eq": "(x>0 required)"}
    b, a = np.polyfit(np.log(x[mask]), y[mask], 1)
    yhat = a + b * np.log(x)
    r2, rss = r2_score(y[mask], yhat[mask])
    return {"name": "Logarithmic", "params": (a, b), "yhat": yhat, "r2": r2, "rss": rss,
            "k": 2, "eq": f"y = {a:.6g} + {b:.6g} ln x"}


def summarize_models(x, y, fits):
    n = len(y)
    for f in fits:
        f["AIC"] = aic(n=np.count_nonzero(
            ~np.isnan(f["yhat"])), rss=f["rss"], k=f["k"])
        f["AICc"] = aicc(n=np.count_nonzero(
            ~np.isnan(f["yhat"])), rss=f["rss"], k=f["k"])
    return fits


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = np.linspace(1, 10, 25)

    # Choose one to test:
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
    fits = summarize_models(x, y, fits)

    # Rank by AICc (lower is better); fall back to R^2 if equal
    fits_sorted = sorted(fits, key=lambda d: (d["AICc"], -d["r2"]))
    best = fits_sorted[0]

    # ---- Print summary
    for f in fits_sorted:
        print(
            f'{f["name"]:<12} R^2={f["r2"]:.4f}  AIC={f["AIC"]:.3f}  AICc={f["AICc"]:.3f}   {f["eq"]}')
    print(f"\nBest model: {best['name']}  ->  {best['eq']}")

    # ---- Plot 1: Model overlay
    plt.figure()
    plt.scatter(x, y, label="Data")
    for f in fits:
        yhat = f["yhat"]
        if np.all(np.isnan(yhat)):
            continue
        plt.plot(x, yhat, label=f'{f["name"]} (R²={f["r2"]:.3f})')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Least-Squares Model Comparison (NumPy-only)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ls_model_comparison.png", dpi=300)

    # ---- Plot 2: Residuals grid (4 panels)
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    axes = axes.ravel()
    for ax, f in zip(axes, fits):
        yhat = f["yhat"]
        if np.all(np.isnan(yhat)):
            ax.text(0.5, 0.5, f"{f['name']}\n(no valid fit)",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        # Residuals only where yhat is valid (masks already applied in fits)
        mask = ~np.isnan(yhat)
        res = y[mask] - yhat[mask]
        ax.scatter(x[mask], res, s=15)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title(f"{f['name']} residuals")
        ax.set_ylabel("y - ŷ")
    for ax in axes[-2:]:
        ax.set_xlabel("x")
    fig.tight_layout()
    fig.savefig("residuals_grid.png", dpi=300)

    # Optional: also save a simple residual histogram for the best model
    mask_best = ~np.isnan(best["yhat"])
    res_best = y[mask_best] - best["yhat"][mask_best]
    plt.figure()
    plt.hist(res_best, bins=10)
    plt.title(f"Residuals (best: {best['name']})")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("residuals_hist_best.png", dpi=300)
    plt.show()  # optional
