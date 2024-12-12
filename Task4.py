"""
This script analyzes birth-related data from a dataset 'birth.dat'.
It visualizes distributions of various variables (e.g., child's birth weight,
mother's age, mother's height, mother's weight), and compares subsets of the data
(based on whether the mother smokes or not, and whether the mother is older or
younger than 24) using histograms, box plots, and kernel density estimates.

Variables studied:
- Child's birth weight (column 3 in the data, index 2 in NumPy)
- Mother's age (column 4, index 3)
- Mother's height (column 16, index 15)
- Mother's weight (column 15, index 14)

The script:
1. Loads the data and extracts specific columns.
2. Creates histograms to visualize distributions of these variables.
3. Divides the dataset into two groups based on smoking status and compares birth weights.
4. Plots box plots and kernel density estimates to compare the distributions between smokers and non-smokers.
5. Divides the dataset based on maternal age (>24 or ≤24) and similarly compares birth weights.
6. Visualizes distributions for these two maternal age groups using box plots and kernel density estimates.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load the data file 'birth.dat' (assumes a text file with numeric data, whitespace-delimited).
# The 'birth' array will have rows corresponding to births and columns to different variables.
birth = np.loadtxt('birth.dat')

# Extract columns from the birth data:
# According to the problem statement and indexing:
# Column 3 (0-based index 2): Child's birth weight
# Column 4 (0-based index 3): Mother's age
# Column 16 (0-based index 15): Mother's height
# Column 15 (0-based index 14): Mother's weight
x1 = birth[:, 2]   # Child's birth weight
x2 = birth[:, 3]   # Mother's age
x3 = birth[:, 15]  # Mother's height
x4 = birth[:, 14]  # Mother's weight

# Create a figure with 4 subplots to visualize histograms of these variables
plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.hist(x1)  # Histogram of child's birth weight
plt.title("Barnets födelsevikt")

plt.subplot(2, 2, 2)
plt.hist(x2)  # Histogram of mother's age
plt.title("Moderns ålder")

plt.subplot(2, 2, 3)
plt.hist(x3)  # Histogram of mother's height
plt.title("Moderns längd")

plt.subplot(2, 2, 4)
plt.hist(x4)  # Histogram of mother's weight
plt.title("Moderns vikt")

plt.show()


# ----------------------------------------------------
# Comparing birth weights of children whose mothers smoke vs. those who do not.
# According to the problem statement:
# Column 20 (index 19) indicates smoking status:
# - Values < 3 (1 or 2) indicate non-smokers
# - Value = 3 indicates smokers

non_smokers = (birth[:, 19] < 3)
smokers = (birth[:, 19] == 3)

# Extract birth weight (column 3, index 2) for non-smokers and smokers
x = birth[non_smokers, 2]
y = birth[smokers, 2]

# Count how many mothers are non-smokers and how many are smokers
vecsizenonsmoker = len(x)
vecsizesmoker = len(y)
print(f"Icke-rökare = {vecsizenonsmoker}")  # Number of non-smokers
print(f"Rökare = {vecsizesmoker}")          # Number of smokers


# Create a new figure to visualize distributions of birth weights for smokers vs non-smokers
plt.figure(figsize=(8, 8))

# Box plot for non-smokers
plt.subplot(2, 2, 1)
plt.boxplot(x)
plt.axis([0, 2, 500, 5000])  # Set axis limits for better visualization
plt.title("Icke-rökare(BLÅ)")

# Box plot for smokers
plt.subplot(2, 2, 2)
plt.boxplot(y)
plt.axis([0, 2, 500, 5000])
plt.title("Rökare(RÖD)")

# Compute kernel density estimates (KDE) for both distributions
kde_x = stats.gaussian_kde(x)
kde_y = stats.gaussian_kde(y)

# Create a grid of points over the range of birth weights to evaluate the KDEs
min_val = np.min(birth[:, 2])
max_val = np.max(birth[:, 2])
grid = np.linspace(min_val, max_val, 60)

# Plot the KDEs for non-smokers (blue) and smokers (red)
plt.subplot(2, 2, (3, 4))
plt.plot(grid, kde_x(grid), 'b')
plt.plot(grid, kde_y(grid), 'r')
plt.title("Barnets födelsevikt")
plt.show()


# ----------------------------------------------------
# Comparing birth weights of children based on mother's category.
# According to the problem statement:
# - Column 8 (index 7) indicates if the mother is a first-time mother, and possibly other info.
#   Here it's used as a filter: >1 means over 24, =1 means under 24 (as per given logic).

birth1 = np.loadtxt('birth.dat')

over_24 = (birth1[:, 7] > 1)
under_24 = (birth1[:, 7] == 1)

# Extract birth weights for these two categories
b1 = birth1[over_24, 2]
b2 = birth1[under_24, 2]

# Remove NaN values if any, to ensure clean data for plotting/KDE
vect1 = b1[~np.isnan(b1)]
vect2 = b2[~np.isnan(b2)]

# Count how many first-time mothers are older than 24 and how many are younger than 24
vecsize_over_24 = len(vect1)
vecsize_under_24 = len(vect2)
print(f"Över 24 = {vecsize_over_24}")
print(f"Under 24 = {vecsize_under_24}")

# Create a new figure to visualize distributions of birth weights for mothers over 24 vs under 24
plt.figure(figsize=(8, 8))

# Box plot for over 24
plt.subplot(2, 2, 1)
plt.boxplot(vect1)
plt.axis([0, 2, 500, 5000])
plt.title("Över 24(BLÅ)")

# Box plot for under 24
plt.subplot(2, 2, 2)
plt.boxplot(vect2)
plt.axis([0, 2, 500, 5000])
plt.title("Under 24(RÖD)")

# Compute KDEs for both distributions
kde_vect1 = stats.gaussian_kde(vect1)
kde_vect2 = stats.gaussian_kde(vect2)

# Use the same grid as before or recreate one if needed
min_val = np.min(birth1[:, 2])
max_val = np.max(birth1[:, 2])
grid = np.linspace(min_val, max_val, 60)

# Plot the KDEs for mothers over 24 (blue) and under 24 (red)
plt.subplot(2, 2, (3, 4))
plt.plot(grid, kde_vect1(grid), 'b')
plt.plot(grid, kde_vect2(grid), 'r')
plt.title("Barnets ålder")
plt.show()
