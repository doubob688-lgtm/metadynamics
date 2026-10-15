import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load PLUMED COLVAR output

#Directory containing this py
folder = Path(__file__).resolve().parent
#Build Path to the CVs
file = folder / "COLVAR_postpro"

data = np.loadtxt(file, comments="#")

time = data[:, 0]
scm = data[:, 1]
tcm = data[:, 2]

plt.figure(figsize=(7, 5))
plt.scatter(scm, tcm, s=5, alpha=0.5, color="royalblue")

plt.xlabel("SCM (Second Coordination Moment)")
plt.ylabel("TCM (Third Coordination Moment)")
plt.title("MD Sampling in CV Space")

plt.tight_layout()
plt.savefig("SCM_TCM.png", dpi=300)
plt.show()
