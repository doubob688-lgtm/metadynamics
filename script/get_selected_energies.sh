#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/bob/miniconda3/envs/metad312/bin/python"
OUTPUT_DIR="${PROJECT_ROOT}/result/get_selected_energies"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" - "${PROJECT_ROOT}" "${OUTPUT_DIR}/energies.csv" <<'PY'
from __future__ import annotations

import csv
from pathlib import Path
import sys

from molecule import Molecule


project_root = Path(sys.argv[1])
output_path = Path(sys.argv[2])
structures = (
    ("SM_Clus", "Starting_Cluster.xyz"),
    ("TS01", "TS01.xyz"),
    ("Hidden_IM1", "Hidden_Intermediate_IM1.xyz"),
    ("IM2", "IM2.xyz"),
)

rows: list[tuple[str, float]] = []
for name, filename in structures:
    molecule = Molecule.from_xyz(project_root / "data" / filename)
    energy_ev = molecule.get_energy()
    rows.append((name, energy_ev))
    print(f"{name}: {energy_ev:.12f} eV")

with output_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(("structure", "energy_eV"))
    writer.writerows(rows)

print(f"Saved energies to {output_path}")
PY
