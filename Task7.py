"""
This script demonstrates simple and multiple linear regressions on birth-related data
using the 'birth.dat' dataset.

1. Simple Linear Regression:
   - Response variable (y): Baby's birth weight (column 3, index 2)
   - Predictor variable (x): Mother's height (column 16, index 15)
   The code fits a simple linear regression model and plots the fitted line
   against the original data.

2. Multiple Linear Regression:
   - Response variable (y): Baby's birth weight (column 3, index 2)
   - Predictor variables:
       x1: Mother's weight (column 15, index 14)
       w:  Mother's smoking habits (column 20, index 19)
           (Converted to a binary variable: w_binary = 1 if w == 3, else 0)
       z:  Mother's age group (column 8, index 7)
           (Converted to a binary variable: z_binary = 1 if z == 1, else 0)
   The code fits a multiple linear regression model and then assesses
   residuals using Q-Q plots and histograms to check normality and distribution.

After fitting the models, the script prints out the estimated coefficients
and their confidence intervals.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

# -----------------------------------------------------------
# Simple Linear Regression
# -----------------------------------------------------------

# Load the data from 'birth.dat'
birth = np.loadtxt('birth.dat')

# Extract variables:
# - Mother's height: column 16 (index 15)
# - Baby's birth weight: column 3 (index 2)
x = birth[:, 15]
y = birth[:, 2]

# Remove any NaN values that might interfere with regression
mask = ~np.isnan(x) & ~np.isnan(y)
x = x[mask]
y = y[mask]

# Add intercept to X (statsmodels expects a design matrix with a constant)
X = sm.add_constant(x)

# Perform the simple linear regression
model = sm.OLS(y, X)
results = model.fit()

# Obtain estimated coefficients and confidence intervals
beta_hat = results.params
beta_int = results.conf_int()

# Compute the fitted values (y_hat) from the model
y_hat = results.predict(X)

# Plot the original data and the fitted regression line
plt.figure()
plt.scatter(x, y, label='Data', color='b')
plt.plot(x, y_hat, label='Fitted Model', color='r')
plt.xlabel("Mother's Height")
plt.ylabel("Birth Weight")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()

# -----------------------------------------------------------
# Multiple Linear Regression
# -----------------------------------------------------------

# Extract additional variables:
# - Mother's weight: column 15 (index 14)
# - Mother's smoking habits: column 20 (index 19)
# - Mother's age group: column 8 (index 7)
# - Baby's birth weight: column 3 (index 2) (re-extracted in case original indexing changed)
x1 = birth[:, 14]   # Mother's weight
w = birth[:, 19]    # Mother's smoking habits
z = birth[:, 7]     # Mother's age group
y = birth[:, 2]     # Baby's birth weight

# Remove NaN values for multiple regression
mask = ~np.isnan(x1) & ~np.isnan(w) & ~np.isnan(z) & ~np.isnan(y)
x1 = x1[mask]
w = w[mask]
z = z[mask]
y = y[mask]

# Convert w (smoking habit) to a binary variable:
# w_binary = 1 if mother is a heavy smoker (w == 3), else 0
w_binary = np.where(w == 3, 1, 0)

# Convert z (age group) to a binary variable:
# z_binary = 1 if z == 1, else 0
z_binary = np.where(z == 1, 1, 0)

# Create the design matrix for multiple regression: [constant, x1, w_binary, z_binary]
X1 = sm.add_constant(np.column_stack((x1, w_binary, z_binary)))

# Perform the multiple linear regression
model = sm.OLS(y, X1)
results = model.fit()

# Extract coefficients and confidence intervals
beta_hat = results.params
beta_int = results.conf_int()

# Compute predicted values
y_hat = results.predict(X1)

# Compute residuals
res = y - y_hat

# Visualize residuals
plt.figure(figsize=(6, 8))

# Q-Q plot of residuals to check normality
plt.subplot(2, 1, 1)
stats.probplot(res, plot=plt)
plt.title("Q-Q Plot of Residuals")

# Histogram of residuals to see their distribution
plt.subplot(2, 1, 2)
plt.hist(res, bins=30, edgecolor='black')
plt.title("Histogram of Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Print coefficients and confidence intervals for the multiple regression
print("Coefficients:")
for i, coef in enumerate(beta_hat):
    print(f"beta_{i}: {coef:.4f}")

print("\nConfidence Intervals:")
for i, ci in enumerate(beta_int):
    print(f"beta_{i}: [{ci[0]:.4f}, {ci[1]:.4f}]")
