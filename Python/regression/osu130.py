from strlinregr import strlinregr
import numpy as np
import matplotlib.pyplot as plt

# Sample data: x and y values
x = np.array([10, 20, 30, 40, 50, 60, 70, 80])
y = np.array([25, 70, 380, 550, 610, 1220, 830, 1450])

# Least-squares linear fit with NumPy (degree = 1)
m, c = np.polyfit(x, y, 1)
y_fit = m * x + c  # not used below, but kept for completeness

# Straight-line regression (your function)
a0, a1, Rsq, SE = strlinregr(x, y)
print('Intercept = {0:7.2f}'.format(a0))
print('Slope = {0:7.3f}'.format(a1))
print('R-squared = {0:5.3f}'.format(Rsq))
print('Standard error = {0:7.2f}'.format(SE))

# Lines and residuals
xline = np.linspace(0, 90, 10)
yline = a0 + a1 * xline
yhat  = a0 + a1 * x
e     = y - yhat

# Scatter + best-fit line
plt.figure()
plt.scatter(x, y, c='k', marker='s')
plt.plot(xline, yline, c='k')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data and Best-Fit Line')
plt.savefig('fit_plot.png', dpi=300, bbox_inches='tight')

# Histogram of residuals
plt.figure()
plt.hist(e, bins=3, color='w', edgecolor='k', linewidth=2.)
plt.grid(True)
plt.xlabel('Residual')
plt.title('Histogram of Residuals')
plt.savefig('residual_hist.png', dpi=300, bbox_inches='tight')

# Residuals vs. fits
plt.figure()
plt.plot(yhat, e, c='k', marker='o')
plt.grid(True)
plt.xlabel('Predicted y')
plt.ylabel('Residual')
plt.title('Residuals vs. Fits')
plt.savefig('residual_vs_fit.png', dpi=300, bbox_inches='tight')

plt.show()
