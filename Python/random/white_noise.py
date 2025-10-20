import numpy as np, matplotlib.pyplot as plt

rng = np.random.default_rng(0)
white = rng.normal(0,1,500)
corr  = np.convolve(white,np.ones(10)/10,mode='same')
plt.plot(white,alpha=.5,label='White')
plt.plot(corr,color='orange',label='Correlated')
plt.legend(); plt.title("Smoothing introduces correlation")
