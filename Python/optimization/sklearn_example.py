import os, sys
this_dir = os.path.dirname(os.path.abspath(__file__))
if this_dir in sys.path:
    sys.path.remove(this_dir)   # avoid local module shadowing
sys.path.append(this_dir)       # (keep if you have other local imports that you need)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Reproducible data
rng = np.random.default_rng(42)
X = rng.random((100, 1))                   # shape (n_samples, n_features)
y = 3.0 * X.squeeze() + rng.normal(0, 0.2, size=100)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2  = r2_score(y_test, preds)

print(f"MSE: {mse:.4f}")
print(f"R^2: {r2:.4f}")
print(f"coef_: {model.coef_[0]:.4f}, intercept_: {model.intercept_:.4f}")

# Plot + save (for slides)
plt.figure(figsize=(6,4))
plt.scatter(X_train, y_train, s=20, alpha=0.6, label="train")
plt.scatter(X_test, y_test, s=20, alpha=0.8, label="test")
xs = np.linspace(0, 1, 100).reshape(-1, 1)
plt.plot(xs, model.predict(xs), linewidth=2.0, label="fit", color="#F97306")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression: fit line and data")
plt.legend()
plt.tight_layout()
plt.savefig("linear_regression_fit.png", dpi=300, bbox_inches="tight")
print("Saved plot: linear_regression_fit.png")
# plt.show()
