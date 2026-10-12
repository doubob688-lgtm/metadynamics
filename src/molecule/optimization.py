"""Geometry-optimization helpers for ASE and the ORCA Python Interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import BaseCalculator
from ase.optimize import BFGS, FIRE, LBFGS
from ase.optimize.optimize import Optimizer
from ase.units import Hartree


@dataclass(frozen=True)
class GeometryOptimizationResult:
    """Backend-independent result of a geometry optimization."""

    atoms: Atoms
    converged: bool
    steps: int | None
    energy_ev: float | None
    energy_hartree: float | None = None
    max_force_ev_per_angstrom: float | None = None


def optimize_with_ase(
    atoms: Atoms,
    calculator: BaseCalculator,
    *,
    optimizer: str | type[Optimizer] = "LBFGS",
    fmax: float = 0.05,
    steps: int = 500,
    logfile: str | Path | None = None,
    trajectory: str | Path | None = None,
    restart: str | Path | None = None,
    optimizer_kwargs: dict[str, Any] | None = None,
) -> GeometryOptimizationResult:
    """Optimize an ASE structure and return its final state."""
    if fmax <= 0:
        raise ValueError(f"fmax must be positive, got {fmax}")
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")

    optimizer_class = _resolve_optimizer(optimizer)
    options = dict(optimizer_kwargs or {})
    reserved = {"restart", "logfile", "trajectory"}.intersection(options)
    if reserved:
        names = ", ".join(sorted(reserved))
        raise ValueError(
            f"Pass {names} as get_GeoOptimization arguments, not optimizer_kwargs"
        )

    atoms.calc = calculator
    dynamics = optimizer_class(
        atoms,
        restart=restart,
        logfile=logfile,
        trajectory=trajectory,
        **options,
    )
    converged = bool(dynamics.run(fmax=fmax, steps=steps))
    energy_ev = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    max_force = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0
    return GeometryOptimizationResult(
        atoms=atoms,
        converged=converged,
        steps=dynamics.get_number_of_steps(),
        energy_ev=energy_ev,
        max_force_ev_per_angstrom=max_force,
    )


def optimize_with_opi(
    atoms: Atoms,
    calculator: Any,
    *,
    charge: int,
    multiplicity: int,
    timeout: int = -1,
) -> GeometryOptimizationResult:
    """Run a native ORCA geometry optimization through ``orca-pi``."""
    try:
        from opi.input.simple_keywords.task import Task
        from opi.input.structures.structure import Structure
    except ImportError as exc:
        raise ImportError(
            "ORCA geometry optimization requires the 'orca-pi' package"
        ) from exc

    opi_input = getattr(calculator, "input", None)
    if opi_input is not None:
        opi_input.remove_simple_keywords(Task.SP)
        opi_input.add_simple_keywords(Task.OPT)

    calculator.structure = Structure.from_ase(
        atoms, charge=charge, multiplicity=multiplicity
    )
    terminated_normally = bool(calculator.write_and_run(timeout=timeout))
    output = calculator.get_output()
    final_structure = output.get_structure(index=-1, fallback=True)
    if final_structure is None:
        status = "normally" if terminated_normally else "abnormally"
        raise RuntimeError(
            f"ORCA terminated {status} but no optimized geometry could be recovered"
        )

    final_atoms = final_structure.to_ase()
    if final_atoms.get_atomic_numbers().tolist() != atoms.get_atomic_numbers().tolist():
        raise ValueError("ORCA optimized geometry changed the atom identities or order")

    convergence_check = getattr(output, "geometry_optimization_converged", None)
    converged = (
        bool(convergence_check())
        if callable(convergence_check)
        else terminated_normally
    )
    energy_hartree = output.get_final_energy()
    if energy_hartree is None:
        energy_ev = None
    else:
        energy_hartree = float(energy_hartree)
        energy_ev = energy_hartree * Hartree

    return GeometryOptimizationResult(
        atoms=final_atoms,
        converged=converged,
        steps=None,
        energy_ev=energy_ev,
        energy_hartree=energy_hartree,
    )


def _resolve_optimizer(
    optimizer: str | type[Optimizer],
) -> type[Optimizer]:
    if isinstance(optimizer, str):
        optimizers: dict[str, type[Optimizer]] = {
            "bfgs": BFGS,
            "fire": FIRE,
            "lbfgs": LBFGS,
        }
        try:
            return optimizers[optimizer.lower()]
        except KeyError as exc:
            supported = ", ".join(sorted(optimizers))
            raise ValueError(
                f"Unknown ASE optimizer {optimizer!r}; choose {supported}"
            ) from exc
    if isinstance(optimizer, type) and issubclass(optimizer, Optimizer):
        return optimizer
    raise TypeError("optimizer must be an ASE Optimizer class or its supported name")
