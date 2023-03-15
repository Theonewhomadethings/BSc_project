'''
plot of the fermi dirac distribution

f(E) = 1 / e^(E-u/k_B*T) + 1 

'''

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Fermi-Dirac Distribution
def fermi(E: float, E_f: float, T: float) -> float:
    k_b = 8.617 * (10**-5) # eV/K
    return 1/(np.exp((E - E_f)/(k_b * T)) + 1)


# Create figure and add axes
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)

# Temperature values
T = np.linspace(100, 1000, 10)
# Get colors from coolwarm colormap
colors = plt.get_cmap('coolwarm', 10)
# Plot F-D data
for i in range(len(T)):
    x = np.linspace(0, 1, 100)
    y = fermi(x, 0.5, T[i])
    ax.plot(x, y, color=colors(i), linewidth=2.5)

# Add legend
labels = ['100 K', '200 K', '300 K', '400 K', '500 K', '600 K', 
          '700 K', '800 K', '900 K', '1000 K']
ax.legend(labels, bbox_to_anchor=(1.05, -0.1), loc='lower left', 
          frameon=False, labelspacing=0.2)