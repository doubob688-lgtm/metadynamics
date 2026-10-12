"""ASE-backed molecular structures and XYZ trajectory loading."""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, overload

from ase import Atoms
from ase.calculators.calculator import BaseCalculator

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class Molecule(Atoms):
    """A molecular structure represented by ASE's established ``Atoms`` type.

    The subclass only adds project-specific loading helpers; coordinates,
    chemical symbols, cell data, constraints, and calculator integration all
    remain provided by ASE.
    """

    @classmethod
    def from_atoms(cls, atoms: Atoms) -> Molecule:
        """Create a molecule by copying an ASE ``Atoms`` object."""
        if not isinstance(atoms, Atoms):
            raise TypeError("atoms must be an ase.Atoms instance")
        return cls(atoms)

    @classmethod
    def from_xyz(cls, path: str | Path, *, index: int = 0) -> Molecule:
        """Load one frame from an ``.xyz`` file.

        Args:
            path: Path to a standard or extended XYZ file.
            index: Zero-based frame index. Negative indices count from the end.
        """
        xyz_path = _validate_xyz_path(path, allowed_suffixes={".xyz"})
        molecules = _read_xyz_frames(xyz_path)
        try:
            return molecules[index]
        except IndexError as exc:
            raise IndexError(
                f"XYZ frame index {index} is out of range for "
                f"{xyz_path} ({len(molecules)} frame(s))"
            ) from exc

    @classmethod
    def load(cls, path: str | Path, *, index: int = 0) -> Molecule:
        """Load one molecule from a supported structure file."""
        return cls.from_xyz(path, index=index)

    def get_energy(
        self,
        calculator: Any | None = None,
        *,
        backend: str = "mace",
        model_path: str | Path | None = None,
        device: str | None = None,
        charge: int | None = None,
        multiplicity: int | None = None,
        default_dtype: str = "float64",
        timeout: int = -1,
        **calculator_kwargs: Any,
    ) -> float:
        """Calculate and return potential energy in eV.

        MACE OMol with the local model is the default. A configured calculator
        from the ORCA Python Interface (``orca-pi``/``opi``) can be supplied
        with ``backend="orca"``. Any ASE calculator, including ASE's ORCA
        calculator, can also be supplied directly.

        Molecular charge and multiplicity default to ``info['charge']`` and
        ``info['spin']``, or to neutral singlet values when absent.
        """
        from .energy import (
            evaluate_with_ase,
            evaluate_with_opi,
            get_mace_omol_calculator,
            is_cuda_failure,
            is_opi_calculator,
        )

        normalized_backend = backend.lower().replace("_", "-")
        mace_backend = normalized_backend in {"mace", "mace-omol", "omol"}
        orca_backend = normalized_backend in {"orca", "orca-pi", "opi"}
        if not mace_backend and not orca_backend:
            raise ValueError(
                f"Unsupported energy backend {backend!r}; expected 'mace' or 'orca'"
            )

        molecular_charge = _resolve_integer_metadata(
            self, explicit=charge, key="charge", default=0
        )
        molecular_multiplicity = _resolve_integer_metadata(
            self, explicit=multiplicity, key="spin", default=1, minimum=1
        )
        self.info["charge"] = molecular_charge
        self.info["spin"] = molecular_multiplicity

        if calculator is not None and calculator_kwargs:
            raise TypeError(
                "calculator_kwargs cannot be used with a preconfigured calculator"
            )

        if calculator is not None and is_opi_calculator(calculator):
            energy_ev, energy_hartree = evaluate_with_opi(
                self,
                calculator,
                charge=molecular_charge,
                multiplicity=molecular_multiplicity,
                timeout=timeout,
            )
            self.info["energy"] = energy_ev
            self.info["orca_energy_hartree"] = energy_hartree
            self.info["energy_backend"] = "orca-pi"
            return energy_ev

        if orca_backend and calculator is None:
            raise ValueError(
                "The ORCA backend requires a configured orca-pi Calculator "
                "or ASE ORCA calculator"
            )

        internally_created = calculator is None
        if calculator is None:
            calculator = get_mace_omol_calculator(
                model_path=model_path,
                device=device,
                default_dtype=default_dtype,
                **calculator_kwargs,
            )
        if not isinstance(calculator, BaseCalculator):
            raise TypeError(
                "calculator must be an ASE calculator or an orca-pi Calculator"
            )

        try:
            energy_ev = evaluate_with_ase(self, calculator)
        except RuntimeError as exc:
            calculator_device = str(getattr(calculator, "device", ""))
            if not (
                internally_created
                and calculator_device.startswith("cuda")
                and is_cuda_failure(exc)
            ):
                raise
            warnings.warn(
                "MACE CUDA evaluation failed; retrying on CPU",
                RuntimeWarning,
                stacklevel=2,
            )
            calculator = get_mace_omol_calculator(
                model_path=model_path,
                device="cpu",
                default_dtype=default_dtype,
                **calculator_kwargs,
            )
            energy_ev = evaluate_with_ase(self, calculator)

        self.info["energy"] = energy_ev
        self.info["energy_backend"] = "mace-omol" if mace_backend else "ase-orca"
        return energy_ev


class MoleculeGroup(Sequence[Molecule]):
    """An ordered collection of molecular structures, such as a trajectory."""

    def __init__(self, molecules: Iterable[Atoms]) -> None:
        self._molecules = tuple(Molecule.from_atoms(item) for item in molecules)

    @classmethod
    def from_xyz(cls, path: str | Path) -> MoleculeGroup:
        """Load every frame from an ``.xyz`` trajectory."""
        xyz_path = _validate_xyz_path(path, allowed_suffixes={".xyz"})
        return cls(_read_xyz_frames(xyz_path))

    @classmethod
    def from_orca_allxyz(cls, path: str | Path) -> MoleculeGroup:
        """Load every geometry from an ORCA ``.allxyz`` file."""
        xyz_path = _validate_xyz_path(path, allowed_suffixes={".allxyz"})
        return cls(_read_xyz_frames(xyz_path))

    @classmethod
    def from_allxyz(cls, path: str | Path) -> MoleculeGroup:
        """Alias for :meth:`from_orca_allxyz`."""
        return cls.from_orca_allxyz(path)

    @classmethod
    def load(cls, path: str | Path) -> MoleculeGroup:
        """Load an XYZ trajectory, selecting the reader by file extension."""
        suffix = Path(path).suffix.lower()
        if suffix == ".allxyz":
            return cls.from_orca_allxyz(path)
        if suffix == ".xyz":
            return cls.from_xyz(path)
        raise ValueError(
            f"Unsupported molecule-group file extension {suffix!r}; "
            "expected '.xyz' or '.allxyz'"
        )

    def __len__(self) -> int:
        return len(self._molecules)

    def __iter__(self) -> Iterator[Molecule]:
        return iter(self._molecules)

    @overload
    def __getitem__(self, index: int) -> Molecule: ...

    @overload
    def __getitem__(self, index: slice) -> MoleculeGroup: ...

    def __getitem__(self, index: int | slice) -> Molecule | MoleculeGroup:
        if isinstance(index, slice):
            return type(self)(self._molecules[index])
        return self._molecules[index]

    def get_energy(
        self,
        calculator: Any | None = None,
        *,
        backend: str = "mace",
        calculator_factory: Callable[[int, Molecule], Any] | None = None,
        model_path: str | Path | None = None,
        device: str | None = None,
        charge: int | None = None,
        multiplicity: int | None = None,
        default_dtype: str = "float64",
        timeout: int = -1,
        **calculator_kwargs: Any,
    ) -> list[float]:
        """Calculate every frame's energy in eV.

        A single MACE or ASE calculator is reused across frames. For ORCA jobs
        that need separate basenames or working directories, pass a callable as
        ``calculator_factory``; it receives ``(frame_index, molecule)`` and must
        return a configured ASE or orca-pi calculator.
        """
        if calculator is not None and calculator_factory is not None:
            raise ValueError("Pass calculator or calculator_factory, not both")
        if (calculator is not None or calculator_factory is not None) and calculator_kwargs:
            raise TypeError(
                "calculator_kwargs cannot be used with a preconfigured calculator "
                "or calculator_factory"
            )
        if not self:
            return []

        normalized_backend = backend.lower().replace("_", "-")
        default_mace_calculator = False
        if (
            calculator is None
            and calculator_factory is None
            and normalized_backend in {"mace", "mace-omol", "omol"}
        ):
            from .energy import get_mace_omol_calculator

            calculator = get_mace_omol_calculator(
                model_path=model_path,
                device=device,
                default_dtype=default_dtype,
                **calculator_kwargs,
            )
            default_mace_calculator = True

        energies: list[float] = []
        for frame_index, molecule in enumerate(self):
            frame_calculator = (
                calculator_factory(frame_index, molecule)
                if calculator_factory is not None
                else calculator
            )
            try:
                energy = molecule.get_energy(
                    calculator=frame_calculator,
                    backend=backend,
                    charge=charge,
                    multiplicity=multiplicity,
                    timeout=timeout,
                )
            except RuntimeError as exc:
                if not (
                    default_mace_calculator
                    and str(getattr(calculator, "device", "")).startswith("cuda")
                ):
                    raise
                from .energy import is_cuda_failure

                if not is_cuda_failure(exc):
                    raise
                warnings.warn(
                    "MACE CUDA evaluation failed; retrying group energies on CPU",
                    RuntimeWarning,
                    stacklevel=2,
                )
                from .energy import get_mace_omol_calculator

                calculator = get_mace_omol_calculator(
                    model_path=model_path,
                    device="cpu",
                    default_dtype=default_dtype,
                    **calculator_kwargs,
                )
                energy = molecule.get_energy(
                    calculator=calculator,
                    backend=backend,
                    charge=charge,
                    multiplicity=multiplicity,
                    timeout=timeout,
                )
            energies.append(energy)
        return energies

    def plot_energy(
        self,
        *,
        energy_key: str | None = None,
        relative: bool = False,
        ax: Axes | None = None,
        output: str | Path | None = None,
        **plot_kwargs: Any,
    ) -> Axes:
        """Plot trajectory energy against frame index.

        ORCA energies parsed from XYZ comments are selected automatically and
        labelled in Hartree. Otherwise, ``energy`` in each molecule's ``info``
        mapping is used. Pass ``energy_key`` to select another numeric metadata
        field explicitly.

        Args:
            energy_key: Key in every molecule's ``info`` mapping. If omitted,
                ORCA energy and then generic ``energy`` metadata are tried.
            relative: Subtract the minimum energy before plotting.
            ax: Existing Matplotlib axes. A new figure and axes are created if
                omitted.
            output: Optional image path passed to ``Figure.savefig``.
            **plot_kwargs: Additional keyword arguments for ``Axes.plot``.

        Returns:
            The Matplotlib axes containing the energy curve.
        """
        if not self:
            raise ValueError("Cannot plot energy for an empty MoleculeGroup")

        selected_key = energy_key
        if selected_key is None:
            for candidate in ("orca_energy_hartree", "energy"):
                if all(candidate in molecule.info for molecule in self):
                    selected_key = candidate
                    break
        if selected_key is None:
            raise ValueError(
                "No common energy metadata found; set molecule.info['energy'] "
                "or pass energy_key"
            )

        energies: list[float] = []
        for frame_index, molecule in enumerate(self):
            if selected_key not in molecule.info:
                raise KeyError(
                    f"Energy key {selected_key!r} is missing from frame {frame_index}"
                )
            try:
                energies.append(float(molecule.info[selected_key]))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Energy value for frame {frame_index} under "
                    f"{selected_key!r} is not numeric"
                ) from exc

        if relative:
            minimum = min(energies)
            energies = [energy - minimum for energy in energies]

        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        style: dict[str, Any] = {"marker": "o"}
        style.update(plot_kwargs)
        ax.plot(range(len(energies)), energies, **style)
        ax.set_xlabel("Frame")
        unit = " (Eh)" if selected_key == "orca_energy_hartree" else ""
        prefix = "Relative " if relative else ""
        ax.set_ylabel(f"{prefix}Energy{unit}")

        if output is not None:
            ax.figure.savefig(Path(output))
        return ax


def _validate_xyz_path(
    path: str | Path, *, allowed_suffixes: set[str]
) -> Path:
    xyz_path = Path(path)
    if xyz_path.suffix.lower() not in allowed_suffixes:
        expected = " or ".join(sorted(allowed_suffixes))
        raise ValueError(
            f"Unsupported file extension {xyz_path.suffix!r}; expected {expected}"
        )
    if not xyz_path.is_file():
        raise FileNotFoundError(f"Molecule file does not exist: {xyz_path}")
    return xyz_path


def _resolve_integer_metadata(
    molecule: Molecule,
    *,
    explicit: int | None,
    key: str,
    default: int,
    minimum: int | None = None,
) -> int:
    value = explicit if explicit is not None else molecule.info.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Molecular {key} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"Molecular {key} must be at least {minimum}, got {value}")
    return value


def _read_xyz_frames(path: Path) -> list[Molecule]:
    """Parse concatenated XYZ blocks while retaining their comment lines."""
    with path.open(encoding="utf-8") as stream:
        lines = stream.readlines()

    frames: list[Molecule] = []
    cursor = 0

    while cursor < len(lines):
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor == len(lines):
            break

        count_line_number = cursor + 1
        try:
            atom_count = int(lines[cursor].strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid XYZ atom count at {path}:{count_line_number}"
            ) from exc
        if atom_count < 0:
            raise ValueError(
                f"Negative XYZ atom count at {path}:{count_line_number}"
            )
        cursor += 1

        if cursor >= len(lines):
            raise ValueError(f"Missing XYZ comment line after {path}:{count_line_number}")
        comment = lines[cursor].rstrip("\r\n")
        cursor += 1

        symbols: list[str] = []
        positions: list[tuple[float, float, float]] = []
        for atom_index in range(atom_count):
            if cursor >= len(lines):
                raise ValueError(
                    f"Incomplete XYZ frame at {path}:{count_line_number}; "
                    f"expected {atom_count} atoms, found {atom_index}"
                )
            fields = lines[cursor].split()
            line_number = cursor + 1
            cursor += 1
            if len(fields) < 4:
                raise ValueError(
                    f"Invalid XYZ atom record at {path}:{line_number}; "
                    "expected an element and three coordinates"
                )
            try:
                coordinates = tuple(
                    float(value.replace("D", "E").replace("d", "e"))
                    for value in fields[1:4]
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid XYZ coordinates at {path}:{line_number}"
                ) from exc
            symbols.append(fields[0])
            positions.append(coordinates)

        molecule = Molecule(symbols=symbols, positions=positions)
        molecule.info["xyz_comment"] = comment
        molecule.info["source_path"] = str(path)
        molecule.info["frame_index"] = len(frames)
        orca_energy = _parse_orca_energy(comment)
        if orca_energy is not None:
            molecule.info["orca_energy_hartree"] = orca_energy
        frames.append(molecule)

    if not frames:
        raise ValueError(f"XYZ file contains no molecular frames: {path}")
    return frames


_ORCA_ENERGY_PATTERN = re.compile(
    r"(?:^|\s)(?:E|Energy)\s*(?:=|:)?\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?)",
    flags=re.IGNORECASE,
)


def _parse_orca_energy(comment: str) -> float | None:
    """Extract the Hartree energy used in ORCA XYZ trajectory comments."""
    match = _ORCA_ENERGY_PATTERN.search(comment)
    if match is None:
        return None
    return float(match.group(1).replace("D", "E").replace("d", "e"))
