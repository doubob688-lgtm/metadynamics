#!/home/bob/miniconda3/envs/metad312/bin/python
"""Split ORCA docking multi-structure XYZ files into individual XYZ files."""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCKING_ROOT = PROJECT_ROOT / "data" / "multiple_sm_et2o_docking"
FILE_GROUPS = (
    ("*.all.optimized.xyz", "optimized"),
    ("*.all.preoptimized.xyz", "preoptimized"),
)


def read_xyz_frames(path: Path) -> list[tuple[int, str]]:
    """Return ``(structure_id, raw_frame)`` pairs from a multi-frame XYZ."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    frames: list[tuple[int, str]] = []
    cursor = 0

    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue

        try:
            atom_count = int(lines[cursor].strip())
        except ValueError as exc:
            raise ValueError(
                f"{path}: expected atom count at line {cursor + 1}"
            ) from exc
        if atom_count < 1:
            raise ValueError(f"{path}: invalid atom count {atom_count}")

        frame_end = cursor + atom_count + 2
        if frame_end > len(lines):
            raise ValueError(
                f"{path}: incomplete frame beginning at line {cursor + 1}"
            )

        comment = lines[cursor + 1].strip()
        try:
            structure_id = int(comment.split(maxsplit=1)[0])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"{path}: comment line does not begin with a structure ID: "
                f"{comment!r}"
            ) from exc

        raw_frame = "".join(lines[cursor:frame_end])
        if not raw_frame.endswith("\n"):
            raw_frame += "\n"
        frames.append((structure_id, raw_frame))
        cursor = frame_end

    structure_ids = [structure_id for structure_id, _ in frames]
    if len(structure_ids) != len(set(structure_ids)):
        raise ValueError(f"{path}: duplicate structure IDs found")
    return frames


def split_file(source: Path, output_dir: Path) -> int:
    """Split one multi-frame XYZ file, refusing to overwrite existing files."""
    frames = read_xyz_frames(source)
    output_dir.mkdir(parents=True, exist_ok=True)

    for structure_id, raw_frame in frames:
        output_path = output_dir / f"structure_{structure_id:04d}.xyz"
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite {output_path}")
        output_path.write_text(raw_frame, encoding="utf-8")
    return len(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "docking_root",
        nargs="?",
        type=Path,
        default=DEFAULT_DOCKING_ROOT,
        help=f"directory containing run_* folders (default: {DEFAULT_DOCKING_ROOT})",
    )
    args = parser.parse_args()

    run_dirs = sorted(path for path in args.docking_root.glob("run_*") if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found in {args.docking_root}")

    total = 0
    for run_dir in run_dirs:
        for pattern, group_name in FILE_GROUPS:
            sources = sorted(run_dir.glob(pattern))
            if len(sources) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one {pattern!r} file in {run_dir}; "
                    f"found {len(sources)}"
                )
            count = split_file(sources[0], run_dir / group_name)
            total += count
            print(f"{run_dir.name}/{group_name}: wrote {count} XYZ files")

    print(f"Wrote {total} individual XYZ files")


if __name__ == "__main__":
    main()
