from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import tools

# Problem 6: Regression
data = np.loadtxt("moore.dat")

x = data[:, 0]  # Årsdata
y = data[:, 1]  # Transistordata

w_k = np.log(y)  # Logaritmen av y vektorn

X = np.column_stack((np.ones_like(x), x))  # Skapar X-vektorn med [1:or i kolonn 1] och [x-värden i kolonn 2]
beta_hat, beta_int = tools.regress(X, w_k)  # Utför regressionen, returnerar värdet på koefficien
y_hat = np.dot(X, beta_hat)  # Utför matrismultiplikationen mellan X och beta_hat,

# Plotta originaldatan och den skattade modellen
plt.scatter(x, w_k, label='Mätdata', color='b')  # Plotta log(y) mot x i blått
plt.plot(x, y_hat, label='Skattad modell', color='r')  # Plotta skattad modell mot x i rött
plt.xlabel('Årsdata')
plt.ylabel('Transistordata')
plt.legend()
plt.show()

# Problem 6: Regression (forts.)

# Bilda residualerna.
res = X @ beta_hat - w_k

# Skapa figur.
plt.figure(figsize=(6, 8))

# Plotta kvantil-kvantil-plot för residualerna.
plt.subplot(2, 1, 1)
_ = stats.probplot(res, plot=plt)

# Plotta histogram för residualerna.
plt.subplot(2, 1, 2)
plt.hist(res, density=True)
plt.show()

jb_res = stats.jarque_bera(res)
print(f"Jarque-Bera för residualerna = {jb_res}")

numtrans = np.exp(beta_hat[0] + beta_hat[1] * 2025)
print(numtrans)
