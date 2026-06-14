from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figure4_pipeline import OBJECTIVES, load_summary


SETTING_LABELS = {
    "wo_cgfm": "w/o CGFM",
    "w_cgfm": "w/ CGFM",
}
SETTING_COLORS = {
    "wo_cgfm": "#d95f02",
    "w_cgfm": "#1f78b4",
}
LOG_OBJECTIVES = {"delta_alpha", "delta_T"}
LOG_EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the Figure 4 reproduction summary.")
    parser.add_argument("--summary", type=Path, required=True, help="Path to figure4_summary.json")
    parser.add_argument("--output", type=Path, required=True, help="Path to output PNG")
    return parser.parse_args()


def plot_values(objective_name: str, values: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if objective_name in LOG_OBJECTIVES:
        return np.maximum(array, LOG_EPSILON)
    return array


def main() -> None:
    args = parse_args()
    summary = load_summary(args.summary)
    aggregated = summary["aggregated"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    flat_axes = axes.flatten()

    for idx, objective in enumerate(OBJECTIVES):
        axis = flat_axes[idx]
        for setting, color in SETTING_COLORS.items():
            key = f"{objective.name}:{setting}"
            if key not in aggregated:
                continue
            stats = aggregated[key]
            mean_curve = np.asarray(stats["mean_curve"], dtype=np.float64)
            std_curve = np.asarray(stats.get("std_curve", np.zeros_like(mean_curve)), dtype=np.float64)
            min_curve = np.asarray(stats.get("min_curve", mean_curve), dtype=np.float64)
            max_curve = np.asarray(stats.get("max_curve", mean_curve), dtype=np.float64)
            x_values = np.arange(1, len(mean_curve) + 1)

            label = SETTING_LABELS.get(setting, setting)
            axis.fill_between(
                x_values,
                plot_values(objective.name, min_curve),
                plot_values(objective.name, max_curve),
                color=color,
                alpha=0.14,
                linewidth=0,
                label=f"{label} min-max",
            )
            axis.fill_between(
                x_values,
                plot_values(objective.name, mean_curve - std_curve),
                plot_values(objective.name, mean_curve + std_curve),
                color=color,
                alpha=0.24,
                linewidth=0,
                label=f"{label} std.",
            )
            axis.plot(
                x_values,
                plot_values(objective.name, mean_curve),
                label=label,
                color=color,
                linewidth=2,
                marker="x" if setting == "w_cgfm" else None,
                markevery=max(1, len(mean_curve) // 12),
                markersize=4,
            )
        axis.set_title(objective.name)
        axis.set_xlabel("Active learning iteration")
        axis.set_ylabel("Best-so-far objective")
        if objective.name in LOG_OBJECTIVES:
            axis.set_yscale("log")
        axis.grid(True, alpha=0.3)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend()

    flat_axes[-1].axis("off")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
