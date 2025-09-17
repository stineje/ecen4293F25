import pylab
from book import strlinregr
import numpy as np
import matplotlib.pyplot as plt

# Sample data: x and y values
x = np.array([10, 20, 30, 40, 50, 60, 70, 80])
y = np.array([25, 70, 380, 550, 610, 1220, 830, 1450])

# Perform a least-squares fit: fit a line y = m*x + c
# numpy.polyfit finds the best fit for a polynomial of degree 1 (linear)
m, c = np.polyfit(x, y, 1)

# Calculate the fitted y-values
y_fit = m * x + c

a0, a1, Rsq, SE = strlinregr(x, y)
print('Intercept = {0:7.2f}'.format(a0))
print('Slope = {0:7.3f}'.format(a1))
print('R-squared = {0:5.3f}'.format(Rsq))
print('Standard error = {0:7.2f}'.format(SE))


xline = np.linspace(0, 90, 10)
yline = a0 + a1*xline
yhat = a0 + a1*x
e = y - yhat
pylab.scatter(x, y, c='k', marker='s')
pylab.plot(xline, yline, c='k')
pylab.grid()
pylab.xlabel('x')
pylab.ylabel('y')
pylab.figure()
pylab.hist(e, bins=3, color='w', edgecolor='k', linewidth=2.)
pylab.grid()
pylab.xlabel('Residual')
pylab.figure()
pylab.plot(yhat, e, c='k', marker='o')
pylab.grid()
pylab.xlabel('Predicted y')
pylab.ylabel('Residual')
pylab.title('Residuals vs. Fits')
pylab.show()
