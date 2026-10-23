#!/home/bob/miniconda3/envs/metad312/bin/python
"""Split one ORCA docking run's XYZ bundles into three result directories."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    PROJECT_ROOT / "data" / "multiple_sm_et2o_docking_1" / "run_et2o_sm"
)
GROUP_PATTERNS = {
    "optimized": ("*.all.optimized.xyz",),
    "preoptimized": ("*.all.preoptimized.xyz",),
    "other": ("*.all.swarm.xyz", "*.docker.build.xyz", "*.docker.xyz"),
}


def read_xyz_frames(path: Path) -> list[tuple[str, str]]:
    """Read raw XYZ frames, retaining every comment and coordinate exactly."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    frames: list[tuple[str, str]] = []
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
        raw_frame = "".join(lines[cursor:frame_end])
        if not raw_frame.endswith("\n"):
            raw_frame += "\n"
        frames.append((comment, raw_frame))
        cursor = frame_end
    return frames


def source_label(path: Path) -> str:
    """Make a concise, collision-free label from a docking XYZ filename."""
    struc_match = re.search(r"\.docker\.(struc\d+)\.all\.(\w+)\.xyz$", path.name)
    if struc_match:
        structure, stage = struc_match.groups()
        return structure if stage in {"optimized", "preoptimized"} else f"{structure}_{stage}"
    if path.name.endswith(".docker.build.xyz"):
        return "build"
    if path.name.endswith(".docker.xyz"):
        return "docker"
    raise ValueError(f"Unrecognized docking XYZ filename: {path.name}")


def split_source(source: Path, output_dir: Path) -> int:
    """Split a source bundle without overwriting any existing output."""
    label = source_label(source)
    frames = read_xyz_frames(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    for frame_number, (comment, raw_frame) in enumerate(frames, start=1):
        first_token = comment.split(maxsplit=1)[0] if comment else ""
        frame_id = int(first_token) if first_token.isdigit() else frame_number
        output_name = f"{label}_structure_{frame_id:04d}.xyz"
        if output_name in used_names:
            raise ValueError(f"{source}: duplicate frame ID {frame_id}")
        used_names.add(output_name)
        output_path = output_dir / output_name
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite {output_path}")
        output_path.write_text(raw_frame, encoding="utf-8")
    return len(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)

    total = 0
    for group, patterns in GROUP_PATTERNS.items():
        sources = sorted({path for pattern in patterns for path in run_dir.glob(pattern)})
        if not sources:
            raise FileNotFoundError(f"No {group} XYZ bundles found in {run_dir}")
        group_total = 0
        for source in sources:
            count = split_source(source, run_dir / group)
            group_total += count
            print(f"{group}: {source.name} -> {count} files")
        total += group_total
        print(f"{group}: wrote {group_total} files total")
    print(f"Wrote {total} individual XYZ files")


if __name__ == "__main__":
    main()
