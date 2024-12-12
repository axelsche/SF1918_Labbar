"""
This script performs a linear regression on a dataset representing Moore's law.
The data ("moore.dat") contains two columns:
- Column 1: Year
- Column 2: Number of transistors on a chip

We:
1. Load the data and take the natural logarithm of the transistor counts.
2. Perform linear regression using a custom regression function from `tools.regress`.
3. Plot the original log-transformed data and the fitted regression line.
4. Compute residuals and analyze their distribution using:
   - Q-Q plots to visually assess normality of residuals.
   - Histograms to examine the distribution shape.
   - Jarque-Bera test to statistically test normality of residuals.
5. Use the fitted model to predict the number of transistors in the year 2025.

If the residuals are approximately normally distributed, it supports the linear regression model assumption.
"""

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import tools

# Load the data from "moore.dat"
# Data format:
#   Column 1: Year (x)
#   Column 2: Number of transistors (y)
data = np.loadtxt("moore.dat")

x = data[:, 0]  # Extract year data
y = data[:, 1]  # Extract transistor count data

# Compute the natural logarithm of the transistor counts
# Linearizing the exponential growth assumption of Moore's law
w_k = np.log(y)

# Form the design matrix X with a column of ones and the year data
# This allows for a linear regression model of the form: w_k = beta0 + beta1 * x
X = np.column_stack((np.ones_like(x), x))

# Perform the regression using a custom tool function "tools.regress"
# It returns the estimated regression coefficients (beta_hat) and confidence intervals (beta_int)
beta_hat, beta_int = tools.regress(X, w_k)

# Compute the fitted values by multiplying the design matrix by the estimated coefficients
y_hat = np.dot(X, beta_hat)

# Plot the original log-transformed data and the fitted regression line
plt.scatter(x, w_k, label='Mätdata (Log-trans)', color='b')
plt.plot(x, y_hat, label='Skattad modell', color='r')
plt.xlabel('Årsdata')
plt.ylabel('Log(Transistordata)')
plt.legend()
plt.show()

# Compute the residuals (difference between fitted values and observed values)
res = X @ beta_hat - w_k

# Create a figure to visualize residual analysis
plt.figure(figsize=(6, 8))

# Subplot 1: Q-Q plot of residuals to visually assess normality
plt.subplot(2, 1, 1)
_ = stats.probplot(res, plot=plt)
plt.title("Q-Q Plot of Residuals")

# Subplot 2: Histogram of residuals to see their distribution
plt.subplot(2, 1, 2)
plt.hist(res, density=True)
plt.title("Histogram of Residuals")

plt.tight_layout()
plt.show()

# Perform the Jarque-Bera test to statistically assess the normality of residuals
jb_res = stats.jarque_bera(res)
print(f"Jarque-Bera för residualerna = Statistic={jb_res.statistic:.4f}, p-value={jb_res.pvalue:.4f}")

# Use the fitted model to predict the number of transistors in the year 2025
numtrans = np.exp(beta_hat[0] + beta_hat[1] * 2025)
print(f"Predicerat antal transistorer år 2025: {numtrans}")
