"""
This script demonstrates how to visually and statistically test the normality of data.
We will:
1. Load data from 'birth.dat'.
2. Extract variables: child's birth weight, mother's age, mother's height, and mother's weight.
3. Remove NaN values to avoid plotting issues with `stats.probplot`.
4. Create Q-Q plots for each variable to visually assess their normality.
5. Perform Jarque-Bera tests for formal normality testing at a 5% significance level.

A P value under 0.05 indicates that data isn't normally distributed.

Variables:
- Child's Birth Weight  (column 3 in data, index 2)
- Mother's Age          (column 4 in data, index 3)
- Mother's Height       (column 16 in data, index 15)
- Mother's Weight       (column 15 in data, index 14)

If the p-value of the Jarque-Bera test is less than 0.05, we reject the null hypothesis that
the data are normally distributed.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load the data file 'birth.dat'.
# The file is expected to be whitespace-separated numeric data.
birth = np.loadtxt('birth.dat')

# Extract relevant columns based on given indices.
x1 = birth[:, 2]   # Child's birth weight
x2 = birth[:, 3]   # Mother's age in years
x3 = birth[:, 15]  # Mother's height in cm
x4 = birth[:, 14]  # Mother's weight in kg

# Remove any NaN values from the arrays to prevent issues with stats.probplot.
x1 = x1[~np.isnan(x1)]
x2 = x2[~np.isnan(x2)]
x3 = x3[~np.isnan(x3)]
x4 = x4[~np.isnan(x4)]

# Create Q-Q plots for each variable.
# Q-Q plots compare the distribution of the sample data with a normal distribution.
# If the data are close to normal, points should lie approximately on the reference line.
plt.figure(figsize=(8, 8))

# Q-Q plot for Child's Birth Weight
plt.subplot(2, 2, 1)
_ = stats.probplot(x1, plot=plt)
plt.title("Child's Birth Weight")

# Q-Q plot for Mother's Age
plt.subplot(2, 2, 2)
_ = stats.probplot(x2, plot=plt)
plt.title("Mother's Age")

# Q-Q plot for Mother's Height
plt.subplot(2, 2, 3)
_ = stats.probplot(x3, plot=plt)
plt.title("Mother's Height")

# Q-Q plot for Mother's Weight
plt.subplot(2, 2, 4)
_ = stats.probplot(x4, plot=plt)
plt.title("Mother's Weight")

plt.tight_layout()
plt.show()

# Perform Jarque-Bera tests for normality on each variable.
# The test evaluates whether the sample data have the skewness and kurtosis of a normal distribution.
jb_x1 = stats.jarque_bera(x1)
jb_x2 = stats.jarque_bera(x2)
jb_x3 = stats.jarque_bera(x3)
jb_x4 = stats.jarque_bera(x4)

# Print out Jarque-Bera test statistics and p-values.
# If p-value < 0.05, we reject the null hypothesis of normality at the 5% level.
print("Jarque-Bera Test Results:")
print(f"Child's Birth Weight: Statistic={jb_x1.statistic:.4f}, p-value={jb_x1.pvalue:.60f}")
print(f"Mother's Age:         Statistic={jb_x2.statistic:.4f}, p-value={jb_x2.pvalue:.4f}")
print(f"Mother's Height:      Statistic={jb_x3.statistic:.4f}, p-value={jb_x3.pvalue:.4f}")
print(f"Mother's Weight:      Statistic={jb_x4.statistic:.4f}, p-value={jb_x4.pvalue:.4f}")
