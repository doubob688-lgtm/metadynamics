#!/home/bob/miniconda3/envs/metad312/bin/python
"""Evaluate MACE OMol energies for the selected diazirine structures."""

from __future__ import annotations

import csv
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from molecule import Molecule  # noqa: E402


STRUCTURES = (
    ("SM_Clus", "Starting_Cluster.xyz"),
    ("TS01", "TS01.xyz"),
    ("Hidden_IM1", "Hidden_Intermediate_IM1.xyz"),
    ("IM2", "IM2.xyz"),
)


def main() -> None:
    """Calculate single-point energies and save them as a CSV file."""
    output_dir = PROJECT_ROOT / "result" / "get_selected_energies"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "energies.csv"

    rows: list[tuple[str, float]] = []
    for name, filename in STRUCTURES:
        molecule = Molecule.from_xyz(PROJECT_ROOT / "data" / filename)
        # MACE OMol is the default backend. CUDA is selected when available,
        # with automatic fallback to CPU if CUDA cannot be used.
        energy_ev = molecule.get_energy()
        rows.append((name, energy_ev))
        print(f"{name}: {energy_ev:.12f} eV")

    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("structure", "energy_eV"))
        writer.writerows(rows)

    print(f"Saved energies to {output_path}")


if __name__ == "__main__":
    main()
