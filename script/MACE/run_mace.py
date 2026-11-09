#!/usr/bin/env python3
"""Evaluate energies and forces for the MACE XYZ input set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Literal

from ase.io import read, write
from mace.calculators import mace_omol
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_calculator(
    model_path: Path,
    requested_device: Literal["auto", "cuda", "cpu"],
    default_dtype: Literal["float32", "float64"],
):
    """Construct MACE OMol on CUDA, falling back to CPU when necessary."""
    device = (
        "cuda"
        if requested_device == "auto" and torch.cuda.is_available()
        else "cpu"
        if requested_device == "auto"
        else requested_device
    )
    try:
        calculator = mace_omol(
            model=model_path,
            device=device,
            default_dtype=default_dtype,
            compute_stress=False,
        )
    except Exception:
        if device == "cpu":
            raise
        calculator = mace_omol(
            model=model_path,
            device="cpu",
            default_dtype=default_dtype,
            compute_stress=False,
        )
        device = "cpu"
    return calculator, device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data/MACE/xyz_inputs",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "model/MACE-omol-0-extra-large-1024.model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "run/MACE",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    calculator, device = build_calculator(args.model, args.device, args.dtype)
    rows: list[tuple[str, float, float, str]] = []

    for xyz_path in sorted(args.input_dir.glob("*.xyz")):
        atoms = read(xyz_path, index=0)
        atoms.calc = calculator
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()

        task_dir = args.output_dir / f"run_{xyz_path.stem}"
        task_dir.mkdir(parents=True, exist_ok=True)
        atoms.info["mace_energy_eV"] = energy
        atoms.arrays["mace_forces_eV_per_A"] = forces
        atoms.calc = None
        write(task_dir / f"{xyz_path.stem}_mace.xyz", atoms, format="extxyz")

        maximum_force = float((forces**2).sum(axis=1).max() ** 0.5)
        rows.append((xyz_path.name, energy, maximum_force, device))

    with (args.output_dir / "mace_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("structure", "energy_eV", "max_force_eV_per_A", "device"))
        writer.writerows(rows)


if __name__ == "__main__":
    main()
