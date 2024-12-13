


import tools
import numpy as np

# Example data (replace with your actual data)
x = np.array([1, 2, 3, 4, 5])  # Independent variable (x_k)
y = np.array([2.7, 7.4, 20.1, 55.5, 148.4])  # Dependent variable (y_k)

# Transform the dependent variable y_k to w_k = log(y_k)
w = np.log(y)

# Perform regression using tools.regress
# Unpack the result tuple into beta_0 and beta_1
beta_0, beta_1 = tools.regress(x, w)

# Ensure beta_1 is a scalar (extract the correct value if it's nested)
if isinstance(beta_1, (list, np.ndarray)) and len(beta_1) > 1:
    beta_1 = beta_1[0]  # Extract the first slope if it's a nested array/list

# Print the estimated parameters
print(f"Estimated parameters:")
print(f"Intercept (β0): {beta_0}")
print(f"Slope (β1): {beta_1}")

# Predict w_k using the estimated regression model
w_predicted = beta_0 + beta_1 * x

# Convert back to the original scale of y_k (if needed)
y_predicted = np.exp(w_predicted)

# Print the predicted values
print(f"Predicted log-transformed values (w_k): {w_predicted}")
print(f"Predicted original values (y_k): {y_predicted}")
