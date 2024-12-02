from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Load the data file
birth = np.loadtxt('birth.dat')

# Import data
x1 = birth[:, 2]   # Child's weight
x2 = birth[:, 3]   # Mother's age in years
x3 = birth[:, 15]  # Mother's height in cm
x4 = birth[:, 14]  # Mother's weight in kg

# Eliminate NaN values
x1 = x1[~np.isnan(x1)]
x2 = x2[~np.isnan(x2)]
x3 = x3[~np.isnan(x3)]
x4 = x4[~np.isnan(x4)]

# Create Q-Q plots
plt.subplot(2, 2, 1)
stats.probplot(x1, plot=plt)
plt.title("Child's Weight")

plt.subplot(2, 2, 2)
stats.probplot(x2, plot=plt)
plt.title("Mother's Age")

plt.subplot(2, 2, 3)
stats.probplot(x3, plot=plt)
plt.title("Mother's Height")

plt.subplot(2, 2, 4)
stats.probplot(x4, plot=plt)
plt.title("Mother's Weight")

plt.tight_layout()
plt.show()

# Perform Jarque-Bera tests
jb_x1 = stats.jarque_bera(x1)
jb_x2 = stats.jarque_bera(x2)
jb_x3 = stats.jarque_bera(x3)
jb_x4 = stats.jarque_bera(x4)

print(f"Jarque-Bera test for Child's Weight: Statistic={jb_x1.statistic}, p-value={jb_x1.pvalue}")
print(f"Jarque-Bera test for Mother's Age: Statistic={jb_x2.statistic}, p-value={jb_x2.pvalue}")
print(f"Jarque-Bera test for Mother's Height: Statistic={jb_x3.statistic}, p-value={jb_x3.pvalue}")
print(f"Jarque-Bera test for Mother's Weight: Statistic={jb_x4.statistic}, p-value={jb_x4.pvalue}")
