#!/home/bob/miniconda3/envs/metad312/bin/python
"""Plot thermodynamic observables from ORCA molecular-dynamics runs."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from statistics import pstdev

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audit_panel_alignment import require_matplotlib_panel_alignment

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


PROJECT_ROOT = Path(os.environ.get("diazirine", "/mnt/c/users/bobdo/diazirine"))
RUN_ROOT = PROJECT_ROOT / "run/MD/md_et2o"
THERMODYNAMIC_VARIABLES = (
    ("Temp", "Temperature (K)"),
    ("E_Kin", "Kinetic energy (Eh)"),
    ("E_Pot", "Potential energy (Eh)"),
    ("E_Tot", "Total energy (Eh)"),
    ("Cons.Qty", "Conserved quantity (Eh)"),
    ("E.Drift", "Energy drift"),
)


def read_series(
    csv_path: Path, value_column_name: str
) -> tuple[list[float], list[float]]:
    times: list[float] = []
    values: list[float] = []

    with csv_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter=";")
        header = [column.removeprefix("#").strip() for column in next(reader)]
        time_column = header.index("Sim. Time")
        value_column = header.index(value_column_name)

        for row in reader:
            value = row[value_column].strip()
            if value:
                times.append(float(row[time_column]))
                values.append(float(value))
    return times, values


def plot_method_series(
    method_dir: Path,
    *,
    column: str,
    ylabel: str,
    title: str,
    output_name: str,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.5))

    for csv_path in sorted(method_dir.glob("run_*/*-md-ener.csv")):
        times, values = read_series(csv_path, column)
        label = csv_path.parent.name.removeprefix("run_")
        axis.plot(times, values, linewidth=1.2, label=label)

    axis.set_xlabel("Simulation time (fs)")
    axis.set_ylabel(ylabel)
    axis.set_title(f"{title}: {method_dir.name.replace('_', ' ')}")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(method_dir / output_name, dpi=600)
    plt.close(figure)


def plot_temperature_standard_deviation(method_dir: Path) -> None:
    labels: list[str] = []
    standard_deviations: list[float] = []

    for csv_path in sorted(method_dir.glob("run_*/*-md-ener.csv")):
        times, temperatures = read_series(csv_path, "Temp")
        temperatures_after_200_fs = [
            temperature
            for time, temperature in zip(times, temperatures, strict=True)
            if time > 200.0
        ]
        labels.append(csv_path.parent.name.removeprefix("run_"))
        standard_deviations.append(pstdev(temperatures_after_200_fs))

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    bars = axis.bar(labels, standard_deviations)
    axis.bar_label(bars, fmt="%.2f", padding=3)
    axis.set_ylabel("Temperature standard deviation (K)")
    axis.set_title(
        "Temperature standard deviation after 200 fs: "
        f"{method_dir.name.replace('_', ' ')}"
    )
    axis.grid(axis="y", alpha=0.3)
    axis.tick_params(axis="x", rotation=10, rotation_mode="anchor")
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    figure.tight_layout()
    figure.savefig(method_dir / "Temperature_std_after_200fs.png", dpi=600)
    plt.close(figure)


def plot_all_thermodynamic_variables(method_dir: Path) -> None:
    csv_paths = sorted(method_dir.glob("run_*/*-md-ener.csv"))
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(7.2, 8.2),
        sharex=True,
    )
    figure.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.08,
        top=0.86,
        hspace=0.28,
        wspace=0.32,
    )

    legend_handles = []
    legend_labels = []
    for panel_index, (axis, (column, ylabel)) in enumerate(
        zip(axes.flat, THERMODYNAMIC_VARIABLES, strict=True)
    ):
        for csv_path in csv_paths:
            times, values = read_series(csv_path, column)
            label = csv_path.parent.name.removeprefix("run_")
            line = axis.plot(times, values, linewidth=0.9, label=label)[0]
            if panel_index == 0:
                legend_handles.append(line)
                legend_labels.append(label)

        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.text(
            -0.12,
            1.04,
            chr(ord("a") + panel_index),
            transform=axis.transAxes,
            fontweight="bold",
            va="bottom",
        )

    for axis in axes[-1, :]:
        axis.set_xlabel("Simulation time (fs)")

    figure.suptitle(
        f"Thermodynamic variables: {method_dir.name.replace('_', ' ')}",
        y=0.985,
    )
    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncols=3,
        frameon=False,
    )
    figure.canvas.draw()
    output_stem = method_dir / "Thermodynamic_variables_vs_time"
    require_matplotlib_panel_alignment(
        figure,
        json_out=output_stem.with_suffix(".alignment.json"),
        overlay_svg=output_stem.with_suffix(".alignment.svg"),
        require_panel_labels=True,
        strict=True,
    )
    figure.savefig(output_stem.with_suffix(".png"), dpi=600)
    figure.savefig(output_stem.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    for method_dir in sorted(path for path in RUN_ROOT.iterdir() if path.is_dir()):
        plot_method_series(
            method_dir,
            column="E.Drift",
            ylabel="E. drift",
            title="Energy drift",
            output_name="E_drift_vs_time.png",
        )
        plot_method_series(
            method_dir,
            column="Temp",
            ylabel="Temperature (K)",
            title="Temperature",
            output_name="Temperature_vs_time.png",
        )
        plot_method_series(
            method_dir,
            column="E_Pot",
            ylabel="Potential energy (Eh)",
            title="Potential energy",
            output_name="E_pot_vs_time.png",
        )
        plot_temperature_standard_deviation(method_dir)
        plot_all_thermodynamic_variables(method_dir)


if __name__ == "__main__":
    main()
