import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

## Problem 1: Simulation of Confidence Intervals

# Parameters

# Number of measurements
n = 10000
# Mean
mu = 2
# Standard deviation
sigma = 1
# One minus the confidence level (alpha)
alpha = 0.05
# Number of intervals
m = 100

# Simulate n observations for each interval
x = stats.norm.rvs(loc=mu, scale=sigma, size=(m, n))

# Estimate mu with the sample mean
xbar = np.mean(x, axis=-1)

# Calculate quantiles and standard error of the mean
z_alpha_over_2 = stats.norm.ppf(1 - alpha / 2)
standard_error = sigma / np.sqrt(n)

# Calculate lower and upper bounds
lower_bound = xbar - z_alpha_over_2 * standard_error
upper_bound = xbar + z_alpha_over_2 * standard_error

## Problem 1: Simulation of Confidence Intervals (continued)

# Create a figure with size 4 × 8 inches
plt.figure(figsize=(4, 8))

# Plot all intervals
for k in range(m):
    # Color red if the interval does not contain mu
    if upper_bound[k] < mu or lower_bound[k] > mu:
        color = 'r'
    else:
        color = 'b'
    plt.plot([lower_bound[k], upper_bound[k]], [k, k], color)

# Adjust axis limits for better visualization
b_min = np.min(lower_bound)
b_max = np.max(upper_bound)
plt.axis([b_min, b_max, -1, m])

# Plot the true value
plt.plot([mu, mu], [-1, m], 'g')

# Show the plot
plt.show()
