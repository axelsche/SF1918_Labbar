"""
This script demonstrates parameter estimation for a Rayleigh distribution using both
the Maximum Likelihood Estimation (MLE) and the Method of Moments Estimation (MME).
A set of Rayleigh-distributed random variables is simulated with a known true scale
parameter (b). The script then computes the MLE and MME of the scale parameter based
on the simulated data. The empirical distribution of the data, along with the
estimated and true probability density functions (PDFs), are visualized using a histogram
and overlaid PDFs. The script concludes by printing the true parameter value, the MLE,
and the MME results.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

## Problem 2: Maximum Likelihood and Method of Moments Estimation

M = 10000  # Number of Rayleigh-distributed random variables
b = 4       # True scale parameter of the Rayleigh distribution

# Simulate M outcomes with parameter b
x = stats.rayleigh.rvs(scale=b, size=M)

# Compute Maximum Likelihood Estimate (MLE) of the scale parameter
est_mle = np.sqrt((1 / (2 * M)) * np.sum(x**2))  # MLE estimation

# Compute Method of Moments Estimate (MME) of the scale parameter
est_mme = (np.mean(x)) * np.sqrt(2 / np.pi)   # Method of Moments estimation

# Create a figure
plt.figure(figsize=(10,6))

# Show histogram of the simulated data
plt.hist(x, bins=40, density=True, alpha=0.6, color='g', label='Empirical Data')

# Plot the MLE and MME estimates as points on the same axis
# Note: The 'y' value (0.2) is arbitrary for visual marker placement.
plt.plot(est_mle, 0.2, 'r*', markersize=10, label='MLE Estimate')
plt.plot(est_mme, 0.2, 'm*', markersize=10, label='MME Estimate')
plt.plot(b, 0.2, 'bo', label='True Value')

# Define a grid for plotting PDFs
x_grid = np.linspace(0, np.max(x), 1000)

# Compute PDFs for the estimated and true parameters
pdf_mle = stats.rayleigh.pdf(x_grid, scale=est_mle)
pdf_mme = stats.rayleigh.pdf(x_grid, scale=est_mme)
pdf_true = stats.rayleigh.pdf(x_grid, scale=b)

# Plot the density functions for the estimated and true parameters
plt.plot(x_grid, pdf_mle, 'r', label='Estimated PDF (MLE)')
plt.plot(x_grid, pdf_mme, 'm', label='Estimated PDF (MME)')
plt.plot(x_grid, pdf_true, 'b', label='True PDF')

plt.legend()
plt.title('Rayleigh Distribution Estimation (MLE & MME)')
plt.xlabel('x')
plt.ylabel('Density')
plt.show()

print("True parameter (b):", b)
print("MLE Estimate:", est_mle)
print("MME Estimate:", est_mme)
