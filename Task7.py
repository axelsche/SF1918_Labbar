import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Simple Linear Regression

# Load the data file
birth = np.loadtxt('birth.dat')

# Extract variables
x = birth[:, 15]  # Mother's height
y = birth[:, 2]   # Baby's birth weight

# Remove NaN values
mask = ~np.isnan(x) & ~np.isnan(y)
x = x[mask]
y = y[mask]

# Add intercept term
X = sm.add_constant(x)

# Perform regression
model = sm.OLS(y, X)
results = model.fit()

# Obtain estimated coefficients and confidence intervals
beta_hat = results.params
beta_int = results.conf_int()

# Predicted values
y_hat = results.predict(X)

# Plot the data and the fitted model
plt.scatter(x, y, label='Data', color='b')
plt.plot(x, y_hat, label='Fitted Model', color='r')
plt.xlabel("Mother's Height")
plt.ylabel("Birth Weight")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()

# Multiple Linear Regression

# Extract variables
x1 = birth[:, 14]  # Mother's weight
w = birth[:, 19]   # Mother's smoking habits
z = birth[:, 7]    # Mother's age group
y = birth[:, 2]    # Baby's birth weight

# Remove NaN values
mask = ~np.isnan(x1) & ~np.isnan(w) & ~np.isnan(z) & ~np.isnan(y)
x1 = x1[mask]
w = w[mask]
z = z[mask]
y = y[mask]

# Process 'w' to create a binary variable
# w_binary = 1 if mother is a heavy smoker (w == 3), else 0
w_binary = np.where(w == 3, 1, 0)

# Process 'z' to create a binary variable
# z_binary = 1 if mother's age group is 1, else 0
z_binary = np.where(z == 1, 1, 0)

# Create design matrix
X1 = sm.add_constant(np.column_stack((x1, w_binary, z_binary)))

# Perform regression
model = sm.OLS(y, X1)
results = model.fit()

# Obtain estimated coefficients and confidence intervals
beta_hat = results.params
beta_int = results.conf_int()

# Predicted values
y_hat = results.predict(X1)

# Compute residuals
res = y - y_hat

# Create figure
plt.figure(figsize=(6, 8))

# Plot Q-Q plot of residuals
plt.subplot(2, 1, 1)
stats.probplot(res, plot=plt)
plt.title("Q-Q Plot of Residuals")

# Plot histogram of residuals
plt.subplot(2, 1, 2)
plt.hist(res, bins=30, edgecolor='black')
plt.title("Histogram of Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Print coefficients and confidence intervals
print("Coefficients:")
for i, coef in enumerate(beta_hat):
    print(f"beta_{i}: {coef:.4f}")

print("\nConfidence Intervals:")
for i, ci in enumerate(beta_int):
    print(f"beta_{i}: [{ci[0]:.4f}, {ci[1]:.4f}]")
