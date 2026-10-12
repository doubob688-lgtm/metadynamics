from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Hartree

from molecule import Molecule, MoleculeGroup


XYZ_TRAJECTORY = """2
first geometry
H 0.0 0.0 0.0
H 0.0 0.0 0.7
2
second geometry
H 0.0 0.0 0.0
H 0.0 0.0 0.8
"""


class AtomCountCalculator(Calculator):
    implemented_properties = ["energy"]

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def calculate(
        self, atoms=None, properties=("energy",), system_changes=all_changes
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        self.results["energy"] = float(len(atoms))


class HarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(
        self, atoms=None, properties=("energy",), system_changes=all_changes
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = 0.5 * float((atoms.positions**2).sum())
        self.results["forces"] = -atoms.positions.copy()


class FakeOrcaOutput:
    def get_final_energy(self) -> float:
        return -1.25

    def get_structure(self, *, index=-1, fallback=True):
        from opi.input.structures.structure import Structure

        return Structure.from_lists(["H"], [(0.0, 0.0, 0.25)])

    def geometry_optimization_converged(self) -> bool:
        return True


class FakeOpiCalculator:
    def __init__(self) -> None:
        self.structure = None
        self.timeout = None

    def write_and_run(self, *, timeout: int = -1) -> bool:
        self.timeout = timeout
        return True

    def get_output(self) -> FakeOrcaOutput:
        return FakeOrcaOutput()


class MoleculeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.tmp_path = Path(self.temporary_directory.name)

    def test_molecule_loads_selected_xyz_frame(self) -> None:
        path = self.tmp_path / "trajectory.xyz"
        path.write_text(XYZ_TRAJECTORY, encoding="utf-8")

        molecule = Molecule.from_xyz(path, index=1)

        self.assertIsInstance(molecule, Atoms)
        self.assertEqual(molecule.get_chemical_formula(), "H2")
        self.assertAlmostEqual(molecule.positions[1, 2], 0.8)
        self.assertEqual(molecule.info["xyz_comment"], "second geometry")
        self.assertEqual(molecule.info["frame_index"], 1)

    def test_group_loads_xyz_trajectory(self) -> None:
        for suffix in (".xyz", ".allxyz"):
            with self.subTest(suffix=suffix):
                path = self.tmp_path / f"optimization{suffix}"
                path.write_text(XYZ_TRAJECTORY, encoding="utf-8")

                group = MoleculeGroup.load(path)

                self.assertEqual(len(group), 2)
                self.assertTrue(all(isinstance(frame, Molecule) for frame in group))
                self.assertEqual(group[0].info["xyz_comment"], "first geometry")
                self.assertIsInstance(group[:1], MoleculeGroup)

    def test_orca_allxyz_accepts_fortran_exponents(self) -> None:
        path = self.tmp_path / "orca.allxyz"
        path.write_text(
            "1\nCoordinates from ORCA-job test E -40.123D+00\n"
            "C 1.0D+00 2.0D+00 3.0D+00\n",
            encoding="utf-8",
        )

        group = MoleculeGroup.from_orca_allxyz(path)

        self.assertEqual(group[0].positions[0].tolist(), [1.0, 2.0, 3.0])
        self.assertAlmostEqual(group[0].info["orca_energy_hartree"], -40.123)

    def test_plot_energy_uses_orca_metadata(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        path = self.tmp_path / "orca.allxyz"
        path.write_text(
            "1\nCoordinates from ORCA-job test E -40.0\nH 0 0 0\n"
            "1\nCoordinates from ORCA-job test E -39.5\nH 0 0 0.1\n",
            encoding="utf-8",
        )
        output = self.tmp_path / "energy.png"

        axes = MoleculeGroup.load(path).plot_energy(relative=True, output=output)

        self.assertEqual(axes.get_xlabel(), "Frame")
        self.assertEqual(axes.get_ylabel(), "Relative Energy (Eh)")
        self.assertEqual(list(axes.lines[0].get_ydata()), [0.0, 0.5])
        self.assertTrue(output.is_file())

    def test_group_rejects_unsupported_extension(self) -> None:
        path = self.tmp_path / "trajectory.dat"
        path.write_text(XYZ_TRAJECTORY, encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "expected '.xyz' or '.allxyz'"):
            MoleculeGroup.load(path)

    def test_molecule_get_energy_accepts_ase_calculator(self) -> None:
        molecule = Molecule("H2", positions=[[0, 0, 0], [0, 0, 0.7]])

        energy = molecule.get_energy(calculator=AtomCountCalculator())

        self.assertEqual(energy, 2.0)
        self.assertEqual(molecule.info["energy"], 2.0)

    def test_group_get_energy_builds_default_mace_once(self) -> None:
        calculator = AtomCountCalculator()
        group = MoleculeGroup(
            [
                Atoms("H", positions=[[0, 0, 0]]),
                Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]]),
            ]
        )

        with patch(
            "molecule.energy.get_mace_omol_calculator",
            return_value=calculator,
        ) as factory:
            energies = group.get_energy()

        self.assertEqual(energies, [1.0, 2.0])
        factory.assert_called_once()
        self.assertEqual(calculator.calls, 2)

    def test_molecule_get_energy_supports_orca_python_interface(self) -> None:
        molecule = Molecule("H", positions=[[0, 0, 0]])
        calculator = FakeOpiCalculator()

        energy = molecule.get_energy(
            calculator=calculator,
            backend="orca",
            charge=0,
            multiplicity=2,
            timeout=30,
        )

        self.assertAlmostEqual(energy, -1.25 * Hartree)
        self.assertAlmostEqual(molecule.info["orca_energy_hartree"], -1.25)
        self.assertEqual(molecule.info["energy_backend"], "orca-pi")
        self.assertEqual(calculator.structure.multiplicity, 2)
        self.assertEqual(calculator.timeout, 30)

    def test_molecule_geometry_optimization_preserves_input(self) -> None:
        molecule = Molecule("H", positions=[[1.0, 0.0, 0.0]])

        optimized = molecule.get_GeoOptimization(
            calculator=HarmonicCalculator(),
            fmax=1.0e-4,
            steps=100,
            logfile=None,
        )

        self.assertAlmostEqual(molecule.positions[0, 0], 1.0)
        self.assertAlmostEqual(optimized.positions[0, 0], 0.0, places=3)
        self.assertTrue(optimized.info["optimization_converged"])
        self.assertLess(optimized.info["max_force_ev_per_angstrom"], 1.0e-4)

    def test_group_geometry_optimization_builds_default_mace_once(self) -> None:
        calculator = HarmonicCalculator()
        group = MoleculeGroup(
            [
                Atoms("H", positions=[[1.0, 0.0, 0.0]]),
                Atoms("H", positions=[[0.0, 1.0, 0.0]]),
            ]
        )

        with patch(
            "molecule.energy.get_mace_omol_calculator",
            return_value=calculator,
        ) as factory:
            optimized = group.get_GeoOptimization(
                fmax=1.0e-4, steps=100, logfile=None
            )

        self.assertEqual(len(optimized), 2)
        self.assertTrue(all(frame.info["optimization_converged"] for frame in optimized))
        factory.assert_called_once()

    def test_geometry_optimization_supports_orca_python_interface(self) -> None:
        molecule = Molecule("H", positions=[[1.0, 0.0, 0.0]])
        calculator = FakeOpiCalculator()

        optimized = molecule.get_GeoOptimization(
            calculator=calculator,
            backend="orca",
            multiplicity=2,
            timeout=45,
        )

        self.assertAlmostEqual(optimized.positions[0, 2], 0.25)
        self.assertTrue(optimized.info["optimization_converged"])
        self.assertAlmostEqual(optimized.info["energy"], -1.25 * Hartree)
        self.assertEqual(optimized.info["optimization_backend"], "orca-pi")
        self.assertEqual(calculator.structure.multiplicity, 2)
        self.assertEqual(calculator.timeout, 45)


if __name__ == "__main__":
    unittest.main()
