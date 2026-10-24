#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/bob/miniconda3/envs/metad312/bin/python"
OUTPUT_DIR="${PROJECT_ROOT}/result/get_selected_geooptimizations"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" - "${PROJECT_ROOT}" "${OUTPUT_DIR}" <<'PY'
from __future__ import annotations

import csv
from pathlib import Path
import sys

from ase.io import write

from molecule import Molecule


project_root = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
structure_names = ("IM1_prime", "TS01", "IM2_prime")

rows: list[tuple[str, bool, int | str, float | str, float | str, str]] = []
for name in structure_names:
    molecule = Molecule.from_xyz(project_root / "data" / f"{name}.xyz")
    optimized = molecule.get_GeoOptimization(
        # The default backend is the local MACE OMol model. It selects CUDA
        # when available and automatically falls back to CPU.
        fmax=0.01,
        steps=500,
        logfile=output_dir / f"{name}.log",
        trajectory=output_dir / f"{name}.traj",
        restart=output_dir / f"{name}.restart.json",
    )

    # Detach the calculator before writing so stored metadata and calculator
    # results do not create duplicate fields in the extended XYZ output.
    optimized.calc = None
    output_path = output_dir / f"{name}_optimized.xyz"
    write(output_path, optimized, format="extxyz")

    converged = bool(optimized.info["optimization_converged"])
    steps = optimized.info.get("optimization_steps", "")
    energy_ev = optimized.info.get("energy", "")
    max_force = optimized.info.get("max_force_ev_per_angstrom", "")
    backend = str(optimized.info["optimization_backend"])
    rows.append((name, converged, steps, energy_ev, max_force, backend))
    print(
        f"{name}: converged={converged}, steps={steps}, "
        f"energy={energy_ev} eV, output={output_path}"
    )

summary_path = output_dir / "optimization_summary.csv"
with summary_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(
        (
            "structure",
            "converged",
            "steps",
            "energy_eV",
            "max_force_eV_per_A",
            "backend",
        )
    )
    writer.writerows(rows)

print(f"Saved optimization summary to {summary_path}")
PY
