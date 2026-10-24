#!/usr/bin/env python3
"""Extract all XYZ coordinate blocks from Computational.docx."""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile


WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WORD_TAG = f"{{{WORD_NAMESPACE}}}"
COORDINATE_PATTERN = re.compile(
    r"^\s*(?P<element>[A-Z][a-z]?)\s+"
    r"(?P<x>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?)\s+"
    r"(?P<y>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?)\s+"
    r"(?P<z>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?)\s*$"
)
SCAN_SECTION_PATTERN = re.compile(r"^TS01 Locating (?P<scan>\d+)$")
SCAN_STEP_PATTERN = re.compile(r"Relaxed Surface Scan Step (?P<step>\d+)")


@dataclass(frozen=True)
class CoordinateBlock:
    start: int
    end: int
    atoms: tuple[tuple[str, str, str, str], ...]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path, help="Source DOCX file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which to create one XYZ file per structure",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing XYZ files"
    )
    return parser.parse_args()


def read_paragraphs(document: Path) -> list[str]:
    with ZipFile(document) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))

    paragraphs: list[str] = []
    for paragraph in root.findall(f".//{WORD_TAG}body//{WORD_TAG}p"):
        pieces: list[str] = []
        for element in paragraph.iter():
            if element.tag == f"{WORD_TAG}t":
                pieces.append(element.text or "")
            elif element.tag == f"{WORD_TAG}tab":
                pieces.append("\t")
            elif element.tag in {f"{WORD_TAG}br", f"{WORD_TAG}cr"}:
                pieces.append("\n")
        paragraphs.append("".join(pieces).strip())
    return paragraphs


def find_coordinate_blocks(paragraphs: list[str]) -> list[CoordinateBlock]:
    blocks: list[CoordinateBlock] = []
    index = 0
    while index < len(paragraphs):
        match = COORDINATE_PATTERN.fullmatch(paragraphs[index])
        if match is None:
            index += 1
            continue

        start = index
        atoms: list[tuple[str, str, str, str]] = []
        while index < len(paragraphs):
            match = COORDINATE_PATTERN.fullmatch(paragraphs[index])
            if match is None:
                break
            atoms.append(
                (
                    match.group("element"),
                    match.group("x").replace("D", "E").replace("d", "e"),
                    match.group("y").replace("D", "E").replace("d", "e"),
                    match.group("z").replace("D", "E").replace("d", "e"),
                )
            )
            index += 1
        blocks.append(CoordinateBlock(start=start, end=index - 1, atoms=tuple(atoms)))
    return blocks


def structure_records(
    paragraphs: list[str], blocks: list[CoordinateBlock]
) -> list[tuple[str, str, CoordinateBlock]]:
    records: list[tuple[str, str, CoordinateBlock]] = []
    previous_end = -1
    active_scan: str | None = None

    for block in blocks:
        context = [
            text
            for text in paragraphs[previous_end + 1 : block.start]
            if text
        ]
        for text in context:
            if SCAN_SECTION_PATTERN.fullmatch(text):
                active_scan = text

        orca_comment = next(
            (text for text in reversed(context) if text.startswith("Coordinates from ORCA-job")),
            None,
        )
        if orca_comment is not None:
            step_match = SCAN_STEP_PATTERN.search(orca_comment)
            if active_scan is None or step_match is None:
                raise ValueError(
                    f"Could not identify scan and step for paragraph {block.start}"
                )
            label = f"{active_scan} Step {int(step_match.group('step')):02d}"
            comment = f"{label} | {orca_comment}"
        else:
            labels = [text for text in context if _is_structure_label(text)]
            if not labels:
                raise ValueError(
                    f"Could not identify structure label before paragraph {block.start}"
                )
            label = labels[-1]
            metadata = [text for text in context if _is_structure_metadata(text)]
            comment = " | ".join([label, *metadata])

        records.append((label, comment, block))
        previous_end = block.end

    return records


def _is_structure_label(text: str) -> bool:
    if text == ">" or text.isdigit():
        return False
    if SCAN_SECTION_PATTERN.fullmatch(text):
        return False
    ignored_prefixes = (
        "Gibbs Free Energy",
        "Number of Imaginary Frequency",
        "Note:",
        "Coordinates from ORCA-job",
        "Geometric Scan along",
    )
    return not text.startswith(ignored_prefixes)


def _is_structure_metadata(text: str) -> bool:
    return text.startswith(
        ("Gibbs Free Energy", "Number of Imaginary Frequency", "Note:")
    )


def filename_for_label(label: str) -> str:
    normalized = label.replace("’", "_prime").replace("'", "_prime")
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        raise ValueError(f"Structure label cannot form a filename: {label!r}")
    return f"{normalized}.xyz"


def relative_path_for_label(label: str) -> Path:
    """Place relaxed-scan frames in their dedicated trajectory directory."""
    filename = filename_for_label(label)
    if SCAN_SECTION_PATTERN.match(label):
        return Path("Trajectory") / filename
    return Path(filename)


def write_xyz_files(
    records: list[tuple[str, str, CoordinateBlock]],
    output_dir: Path,
    *,
    force: bool,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    relative_paths = [relative_path_for_label(label) for label, _, _ in records]
    duplicates = sorted(
        {str(path) for path in relative_paths if relative_paths.count(path) > 1}
    )
    if duplicates:
        raise ValueError(f"Duplicate output filenames: {', '.join(duplicates)}")

    paths = [output_dir / relative_path for relative_path in relative_paths]
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing files: {names}")

    for path, (_, comment, block) in zip(paths, records, strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [str(len(block.atoms)), comment]
        lines.extend(
            f"{element:<2} {x:>20} {y:>20} {z:>20}"
            for element, x, y, z in block.atoms
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths


def main() -> None:
    arguments = parse_arguments()
    paragraphs = read_paragraphs(arguments.document)
    blocks = find_coordinate_blocks(paragraphs)
    records = structure_records(paragraphs, blocks)
    paths = write_xyz_files(records, arguments.output_dir, force=arguments.force)
    print(f"Extracted {len(paths)} XYZ structures to {arguments.output_dir}")
    for path in paths:
        print(path.name)


if __name__ == "__main__":
    main()
