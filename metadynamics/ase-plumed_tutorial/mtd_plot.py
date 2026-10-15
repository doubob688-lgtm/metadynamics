import numpy as np
import matplotlib.pyplot as plt

# Load PLUMED FES file
data = np.loadtxt("fes.dat", comments="#")

scm = data[:, 0]
tcm = data[:, 1]
fes = data[:, 2]

# Reconstruct the 2D grid
x = np.unique(scm)
y = np.unique(tcm)

X, Y = np.meshgrid(x, y)

# Sort grid values into their correct positions
ix = np.searchsorted(x, scm)
iy = np.searchsorted(y, tcm)

Z = np.full((len(y), len(x)), np.nan)
Z[iy, ix] = fes

# Normalise free energy
Z = Z - np.nanmin(Z)

# Plot
plt.figure(figsize=(8, 6))

contour = plt.contourf(
    X, Y, Z,
    levels=30,
    cmap="viridis"
)

plt.colorbar(contour, label="Free energy (eV)")

plt.xlabel("SCM (Second Coordination Moment)")
plt.ylabel("TCM (Third Coordination Moment)")
plt.title("Free Energy Surface")

plt.tight_layout()
plt.savefig("FES.png", dpi=300)
plt.show()
