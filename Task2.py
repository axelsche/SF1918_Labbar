"""
Fördelningen ser ut att passa bra för relighfölrdelnignen, då det mest frekventa värdet i simuleringen är runt 4 vilket
passar fördelningens mdel och att det är en relativt jämn fördelning. Staplarna från den diskreta mätningen.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

## Problem 2: Maximum Likelihood and Method of Moments Estimation

M = 100000  # Number of Rayleigh-distributed random variables
b = 4       # True scale parameter of the Rayleigh distribution

# Simulate M outcomes with parameter b
x = stats.rayleigh.rvs(scale=b, size=M)

# Compute Maximum Likelihood Estimate (MLE) of the scale parameter
est_mle = np.sqrt((1 / (2 * M)) * np.sum(x**2))  # MLE estimation

# Compute Method of Moments Estimate (MME) of the scale parameter
est_mme = (np.sum(x) / M) * np.sqrt(2 / np.pi)   # Method of Moments estimation

# Create figure
plt.figure()

# Show histogram of the simulated data
plt.hist(x, bins=40, density=True, alpha=0.6, color='g')

# Plot the MLE and MME estimates
plt.plot(est_mle, 0.2, 'r*', markersize=10, label='MLE Estimate')
plt.plot(est_mme, 0.2, 'g*', markersize=10, label='Method of Moments Estimate')
plt.plot(b, 0.2, 'bo', label='True Value')

# Plot the density functions for the estimated and true parameters
x_grid = np.linspace(0, np.max(x), 1000)
pdf_mle = stats.rayleigh.pdf(x_grid, scale=est_mle)
pdf_true = stats.rayleigh.pdf(x_grid, scale=b)

plt.plot(x_grid, pdf_mle, 'r', label='Estimated PDF (MLE)')
plt.plot(x_grid, pdf_true, 'b', label='True PDF')

plt.legend()
plt.title('Rayleigh Distribution Estimation')
plt.xlabel('x')
plt.ylabel('Density')
plt.show()
