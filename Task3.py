import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

## Problem 3: Confidence Intervals for the Rayleigh Distribution

# Load data
y = np.loadtxt('wave_data.dat')

# Create a figure with multiple subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot the first 100 data points of the signal
axs[0, 0].plot(y[:100])
axs[0, 0].set_title('First 100 Data Points of the Signal')
axs[0, 0].set_xlabel('Sample Index')
axs[0, 0].set_ylabel('Amplitude')

# Plot the histogram of the data
axs[0, 1].hist(y, density=True, bins=30, alpha=0.6, color='g')
axs[0, 1].set_title('Histogram of the Data')
axs[0, 1].set_xlabel('Data')
axs[0, 1].set_ylabel('Density')

## Problem 3: Confidence Intervals (continued)

n = len(y)          # Number of data points
pi = np.pi          # Define pi

# Method of Moments estimation of the scale parameter
est = (np.sqrt(2 / pi)) * np.mean(y)

# Confidence interval
alpha = 0.05        # For a 95% confidence level
z = stats.norm.ppf(1 - alpha / 2)  # z-score for the confidence level

# Calculate the standard error
d = est * np.sqrt((4 - pi) / (pi * n))

# Compute the lower and upper bounds of the confidence interval
lower_bound = est - z * d
upper_bound = est + z * d

# Plot the histogram and the estimated PDF
axs[1, 0].hist(y, density=True, bins=30, alpha=0.6, color='g', label='Data Histogram')

# Plot the estimated PDF
x_grid = np.linspace(0, np.max(y), 1000)
pdf = stats.rayleigh.pdf(x_grid, scale=est)
axs[1, 0].plot(x_grid, pdf, 'r', label='Estimated PDF')

# Plot the confidence interval bounds
axs[1, 0].plot([lower_bound, upper_bound], [0.6, 0.6], 'g*', markersize=10, label='Confidence Interval Bounds')

axs[1, 0].set_title('Rayleigh Distribution with Confidence Interval')
axs[1, 0].set_xlabel('Data')
axs[1, 0].set_ylabel('Density')
axs[1, 0].legend()

# Hide the unused subplot (bottom right)
axs[1, 1].axis('off')

# Adjust layout
plt.tight_layout()

# Show all plots
plt.show()
