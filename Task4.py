import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# Problem 4: Fördelningar av givna data
# Ladda datafilen.
birth = np.loadtxt('birth.dat')

x1 = birth[:, 2]  # Barnets födelsevikt
x2 = birth[:, 3]  # Moderns ålder
x3 = birth[:, 12]  # Moderns längd
x4 = birth[:, 14]  # Moderns vikt

plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.hist(x1)
plt.title("Barnets födelsevikt")

plt.subplot(2, 2, 2)
plt.hist(x2)
plt.title("Moderns ålder")

plt.subplot(2, 2, 3)
plt.hist(x3)
plt.title("Moderns längd")

plt.subplot(2, 2, 4)
plt.hist(x4)
plt.title("Moderns vikt")
plt.show()


# Definiera filter beroende på om modern röker (kolonn 20 är 3) eller inte (kolonn 20 är 1 eller 2). Notera att
# eftersom indexering i Python börjar med noll så betecknas kolonn 20 med indexet 19.
non_smokers = (birth[:, 19] < 3)
smokers = (birth[:, 19] == 3)

# Extrahera födelsevikten (kolonn 3) för de två kategorierna.
x = birth[non_smokers, 2]
y = birth[smokers, 2]

# Avgör hur många förstföderskor som röker eller inte under graviditeten
vecsizenonsmoker = len(x)
vecsizesmoker = len(y)
print(f"Icke-rökare = {vecsizenonsmoker}")  # Antal icke-rökare
print(f"Rökare = {vecsizesmoker}")  # Antal rökare

# Problem 4: Fördelningar av givna data (forts.)

# Skapa en stor figur.
plt.figure(figsize=(8, 8))

# Plotta ett låddiagram över x.
plt.subplot(2, 2, 1)
plt.boxplot(x)
plt.axis([0, 2, 500, 5000])
plt.title("Icke-rökare(BLÅ)")

# Plotta ett låddiagram över y.
plt.subplot(2, 2, 2)
plt.boxplot(y)
plt.axis([0, 2, 500, 5000])
plt.title("Rökare(RÖD)")

# Beräkna kärnestimator för x och y. Funktionen
# gaussian_kde returnerar ett funktionsobjekt som sedan
# kan evalueras i godtyckliga punkter.
kde_x = stats.gaussian_kde(x)
kde_y = stats.gaussian_kde(y)

# Skapa ett rutnät för vikterna som vi kan använda för att
# beräkna kärnestimatorernas värden.
min_val = np.min(birth[:, 2])
max_val = np.max(birth[:, 2])
grid = np.linspace(min_val, max_val, 60)

# Plotta kärnestimatorerna.
plt.subplot(2, 2, (3, 4))
plt.plot(grid, kde_x(grid), 'b')
plt.plot(grid, kde_y(grid), 'r')
plt.title("Barnets ålder")
plt.show()




# Förstföderskans ålder

# Definiera filter beroende på om modern röker (kolonn 20 är 3) eller inte (kolonn 20 är 1 eller 2). Notera att
# eftersom indexering i Python börjar med noll så betecknas kolonn 20 med indexet 19.
birth1 = np.loadtxt('birth.dat')

over_24 = (birth1[:, 7] > 1)
under_24 = (birth1[:, 7] == 1)

# Extrahera födelsevikten (kolonn 3) för de två kategorierna.
b1 = birth1[over_24, 2]
b2 = birth1[under_24, 2]

vect1 = b1[~np.isnan(b1)]
vect2 = b2[~np.isnan(b2)]

# Avgör hur många förstföderskor som är äldre än 24 år eller yngre än 24 år under graviditeten.
vecsize_over_24 = len(vect1)
vecsize_under_24 = len(vect2)
print(f"Över 24 = {vecsize_over_24}")
print(f"Under 24 = {vecsize_under_24}")

# Förstföderskans ålder (forts.)

# Skapa en stor figur.
plt.figure(figsize=(8, 8))

# Plotta ett låddiagram över x.
plt.subplot(2, 2, 1)
plt.boxplot(vect1)
plt.axis([0, 2, 500, 5000])
plt.title("Över 24(BLÅ)")

# Plotta ett låddiagram över y.
plt.subplot(2, 2, 2)
plt.boxplot(vect2)
plt.axis([0, 2, 500, 5000])
plt.title("Under 24(RÖD)")

# Beräkna kärnestimator för x och y. Funktionen
# gaussian_kde returnerar ett funktionsobjekt som sedan
# kan evalueras i godtyckliga punkter.
kde_vect1 = stats.gaussian_kde(vect1)
kde_vect2 = stats.gaussian_kde(vect2)

# Skapa ett rutnät för vikterna som vi kan använda för att
# beräkna kärnestimatorernas värden.
min_val = np.min(birth1[:, 2])
max_val = np.max(birth1[:, 2])
grid = np.linspace(min_val, max_val, 60)

# Plotta kärnestimatorerna.
plt.subplot(2, 2, (3, 4))
plt.plot(grid, kde_vect1(grid), 'b')
plt.plot(grid, kde_vect2(grid), 'r')
plt.title("Barnets ålder")
plt.show()
