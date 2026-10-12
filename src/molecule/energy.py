"""Energy-calculator construction and backend adapters."""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.calculator import BaseCalculator
from ase.units import Hartree


DEFAULT_OMOL_MODEL = (
    Path(__file__).resolve().parents[2]
    / "model"
    / "MACE-omol-0-extra-large-1024.model"
)


def resolve_mace_device(device: str | None = None) -> str:
    """Select CUDA when available and otherwise report a CPU fallback."""
    import torch

    if device is not None and device != "":
        if device.startswith("cuda") and not torch.cuda.is_available():
            warnings.warn(
                f"Requested MACE device {device!r} is unavailable; falling back to CPU",
                RuntimeWarning,
                stacklevel=2,
            )
            return "cpu"
        return device

    if torch.cuda.is_available():
        return "cuda"
    warnings.warn(
        "CUDA is unavailable; using the MACE OMol model on CPU",
        RuntimeWarning,
        stacklevel=2,
    )
    return "cpu"


def get_mace_omol_calculator(
    *,
    model_path: str | Path | None = None,
    device: str | None = None,
    default_dtype: str = "float64",
    cache: bool = True,
    **calculator_kwargs: Any,
) -> BaseCalculator:
    """Build the project's default MACE OMol ASE calculator.

    Calculator instances are cached when no custom MACE keyword arguments are
    supplied, avoiding repeated loading of the large local model.
    """
    path = Path(model_path) if model_path is not None else DEFAULT_OMOL_MODEL
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MACE OMol model does not exist: {path}")

    selected_device = resolve_mace_device(device)
    if cache and not calculator_kwargs:
        return _cached_mace_omol_calculator(
            str(path), selected_device, default_dtype
        )
    return _build_mace_omol_calculator(
        path=path,
        device=selected_device,
        default_dtype=default_dtype,
        **calculator_kwargs,
    )


@lru_cache(maxsize=4)
def _cached_mace_omol_calculator(
    model_path: str, device: str, default_dtype: str
) -> BaseCalculator:
    return _build_mace_omol_calculator(
        path=Path(model_path), device=device, default_dtype=default_dtype
    )


def _build_mace_omol_calculator(
    *, path: Path, device: str, default_dtype: str, **calculator_kwargs: Any
) -> BaseCalculator:
    from mace.calculators import mace_omol

    try:
        return mace_omol(
            model=path,
            device=device,
            default_dtype=default_dtype,
            **calculator_kwargs,
        )
    except RuntimeError as exc:
        if device.startswith("cuda") and is_cuda_failure(exc):
            warnings.warn(
                "MACE CUDA initialization failed; falling back to CPU",
                RuntimeWarning,
                stacklevel=2,
            )
            return mace_omol(
                model=path,
                device="cpu",
                default_dtype=default_dtype,
                **calculator_kwargs,
            )
        raise


def is_cuda_failure(error: BaseException) -> bool:
    """Return whether an exception appears to be a CUDA runtime failure."""
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "cuda",
            "cublas",
            "cudnn",
            "gpu",
            "nvidia",
            "out of memory",
        )
    )


def is_opi_calculator(calculator: Any) -> bool:
    """Recognize an ORCA Python Interface calculator without a hard dependency."""
    return all(
        callable(getattr(calculator, attribute, None))
        for attribute in ("write_and_run", "get_output")
    ) and hasattr(calculator, "structure")


def evaluate_with_ase(atoms: Atoms, calculator: BaseCalculator) -> float:
    """Evaluate and return an ASE calculator energy in eV."""
    atoms.calc = calculator
    return float(atoms.get_potential_energy())


def evaluate_with_opi(
    atoms: Atoms,
    calculator: Any,
    *,
    charge: int,
    multiplicity: int,
    timeout: int = -1,
) -> tuple[float, float]:
    """Run an OPI calculation and return energy as ``(eV, Hartree)``."""
    try:
        from opi.input.structures.structure import Structure
    except ImportError as exc:
        raise ImportError(
            "ORCA backend support requires the 'orca-pi' package"
        ) from exc

    calculator.structure = Structure.from_ase(
        atoms, charge=charge, multiplicity=multiplicity
    )
    terminated_normally = calculator.write_and_run(timeout=timeout)
    if not terminated_normally:
        raise RuntimeError("ORCA calculation did not terminate normally")

    energy_hartree = calculator.get_output().get_final_energy()
    if energy_hartree is None:
        raise RuntimeError("ORCA output does not contain a final energy")
    energy_hartree = float(energy_hartree)
    return energy_hartree * Hartree, energy_hartree
