"""
This script loads wave amplitude data from a text file, visualizes the data,
and then estimates parameters of the Rayleigh distribution using the method of
moments. It also computes a 95% confidence interval for the estimated scale
parameter and visualizes the results.

Specifically, it:
- Plots the first 100 samples of the given signal data.
- Plots a histogram of the entire dataset.
- Estimates the scale parameter of a Rayleigh distribution fit to the data.
- Computes the confidence interval for this scale parameter estimate.
- Overlays the estimated PDF and confidence interval bounds on the histogram.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load data
# The dataset "wave_data.dat" should be a text file containing numerical values.
# Each line typically represents one data point.
y = np.loadtxt('wave_data.dat')

# Create a figure with multiple subplots for visualization
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot the first 100 data points of the signal to give a quick look at the waveform
axs[0, 0].plot(y[:100], 'b')  # If desired, remove '[:100]' to plot all data points
axs[0, 0].set_title('First 100 Data Points of the Signal')
axs[0, 0].set_xlabel('Sample Index')
axs[0, 0].set_ylabel('Amplitude')

# Plot a histogram of the entire dataset to see its distribution
# Setting density=True normalizes the histogram to a probability density
axs[0, 1].hist(y, density=True, bins=30, alpha=0.6, color='g')
axs[0, 1].set_title('Histogram of the Data')
axs[0, 1].set_xlabel('Data')
axs[0, 1].set_ylabel('Density')

# Number of data points
n = len(y)
# Define pi for convenience
pi = np.pi

# Method of Moments (MoM) estimation of the Rayleigh scale parameter
# For a Rayleigh distribution, the scale parameter 'sigma' can be estimated using:
# sigma_hat = sqrt( (2 / pi) ) * mean(y)
est = (np.sqrt(2 / pi)) * np.mean(y)

# Compute a 95% confidence interval for the estimated scale parameter
alpha = 0.05
z = stats.norm.ppf(1 - alpha / 2)  # z-score for the 95% confidence level

# Calculate the standard error for the estimated parameter
# Derived from the variance of the estimator; the formula here is problem-specific
d = est * np.sqrt((4 - pi) / (pi * n))

# Compute the lower and upper bounds of the confidence interval
lower_bound = est - z * d
upper_bound = est + z * d

# Plot the histogram of the data again (in a different subplot) for overlaying the PDF
axs[1, 0].hist(y, density=True, bins=30, alpha=0.6, color='g', label='Data Histogram')

# Plot the Rayleigh PDF using the estimated scale parameter
x_grid = np.linspace(0, np.max(y), 1000)
pdf = stats.rayleigh.pdf(x_grid, scale=est)
axs[1, 0].plot(x_grid, pdf, 'r', label='Estimated PDF')

# Plot markers to indicate the confidence interval bounds on the subplot.
# Note: The placement of the markers on the y-axis is arbitrary (0.6 here) for visualization.
axs[1, 0].plot([lower_bound, upper_bound], [0.6, 0.6], 'g*', markersize=10, label='Confidence Interval Bounds')

axs[1, 0].set_title('Rayleigh Distribution with Confidence Interval')
axs[1, 0].set_xlabel('Data')
axs[1, 0].set_ylabel('Density')
axs[1, 0].legend()

# The bottom-right subplot is unused, so we turn it off for a cleaner layout
axs[1, 1].axis('off')

# Adjust the layout to prevent overlap
plt.tight_layout()

# Display all plots
plt.show()
