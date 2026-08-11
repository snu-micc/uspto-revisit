"""Plot Gemini 3.6 Flash outcomes sorted by denoising accuracy."""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

MODEL = "gemini-3.6-flash-high"
BLUE = "#36A3D9"
RED = "#D9574F"
PURPLE = "#7657B5"
HEURISTIC_LABELS = {
    "rare_template": "Template anomaly",
    "many_reactants": "Many reactants",
    "many_reagents": "Many reagents",
    "no_reagent": "No reagent",
    "many_products": "Many products",
}


def one_decimal(value: float) -> Decimal:
    return Decimal(str(round(float(value), 8))).quantize(
        Decimal("0.1"), rounding=ROUND_HALF_UP
    )


def load_data(path: Path) -> pd.DataFrame:
    required = {
        "model",
        "heuristic",
        "noise_free_data_saved_rate",
        "noisy_data_filtered_rate",
        "denoising_accuracy",
    }
    data = pd.read_csv(path)
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    data = data.loc[data["model"].eq(MODEL)].copy()
    if data.empty:
        raise ValueError(f"No rows found for {MODEL}")
    return data.sort_values("denoising_accuracy", ascending=False).reset_index(drop=True)


def style_axis(ax: plt.Axes, *, reverse: bool = False) -> None:
    ax.set_xlim((100, 0) if reverse else (0, 100))
    ax.set_xticks([100, 50, 0] if reverse else [0, 50, 100])
    ax.tick_params(axis="x", labelsize=38, length=6)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.9, alpha=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.patch.set_alpha(0)


def label_bars(
    ax: plt.Axes,
    values,
    positions,
    *,
    reverse: bool = False,
    color: str = "white",
    fontsize: float = 34,
) -> None:
    for value, position in zip(values, positions):
        value = float(value)
        value_fontsize = fontsize
        if value < 8:
            value_fontsize = min(fontsize, 24)
        elif value < 16:
            value_fontsize = min(fontsize, 24)
        if value < 16:
            x = value / 2
            ha = "center"
        elif reverse:
            x = value - 2.0
            ha = "left"
        else:
            x = value - 2.0
            ha = "right"
        ax.text(
            x,
            position,
            f"{one_decimal(value)}",
            ha=ha,
            va="center",
            fontsize=value_fontsize,
            fontweight="bold",
            color=color,
            clip_on=True,
            zorder=5,
        )


def plot(data: pd.DataFrame, output: Path, dpi: int) -> tuple[Path, Path]:
    spacing = 1.18
    centers = [index * spacing for index in range(len(data))]
    pair_height = 0.34
    clean_positions = [position - pair_height / 2 for position in centers]
    noisy_positions = [position + pair_height / 2 for position in centers]

    clean = data["noise_free_data_saved_rate"].to_numpy() * 100
    noisy = data["noisy_data_filtered_rate"].to_numpy() * 100
    accuracy = data["denoising_accuracy"].to_numpy() * 100

    fig = plt.figure(figsize=(31, 13.5))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 0.44, 1.0), wspace=0.08)
    ax_outcomes = fig.add_subplot(grid[0, 0])
    ax_labels = fig.add_subplot(grid[0, 1], sharey=ax_outcomes)
    ax_accuracy = fig.add_subplot(grid[0, 2], sharey=ax_outcomes)

    ax_outcomes.barh(
        clean_positions,
        clean,
        pair_height,
        color=BLUE,
        edgecolor="none",
        zorder=3,
    )
    ax_outcomes.barh(
        noisy_positions,
        noisy,
        pair_height,
        color=RED,
        edgecolor="none",
        zorder=3,
    )
    style_axis(ax_outcomes, reverse=True)
    ax_outcomes.set_title("Saved and filtered reactions", fontsize=46, fontweight="bold", pad=18)
    label_bars(ax_outcomes, clean, clean_positions, reverse=True, fontsize=28)
    label_bars(ax_outcomes, noisy, noisy_positions, reverse=True, fontsize=28)

    ax_accuracy.barh(
        centers,
        accuracy,
        0.68,
        color=PURPLE,
        edgecolor="none",
        zorder=3,
    )
    style_axis(ax_accuracy)
    ax_accuracy.set_title("Denoising accuracy", fontsize=46, fontweight="bold", pad=18)
    label_bars(ax_accuracy, accuracy, centers)

    ax_outcomes.set_ylim(-0.72, centers[-1] + 0.72)
    ax_outcomes.invert_yaxis()

    ax_labels.set_xlim(0, 1)
    ax_labels.set_title("Heuristic", fontsize=46, fontweight="bold", pad=18)
    ax_labels.axis("off")
    for center, heuristic in zip(centers, data["heuristic"]):
        ax_labels.text(
            0.5,
            center,
            HEURISTIC_LABELS[heuristic].replace("\n", " "),
            ha="center",
            va="center",
            fontsize=40,
            fontweight="bold",
            color="black",
        )

    outcome_legend = [
        Patch(facecolor=BLUE, edgecolor="none", label="Clean saved"),
        Patch(facecolor=RED, edgecolor="none", label="Noisy filtered"),
    ]
    fig.legend(
        handles=outcome_legend,
        loc="lower left",
        bbox_to_anchor=(0.08, 0.025),
        ncol=2,
        frameon=False,
        fontsize=36,
        handlelength=1.1,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    fig.supxlabel("Percentage (%)", fontsize=40, y=0.105)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.80, bottom=0.20)
    fig.patch.set_alpha(0)

    output.parent.mkdir(parents=True, exist_ok=True)
    png = Path(f"{output}.png")
    svg = Path(f"{output}.svg")
    fig.savefig(png, dpi=dpi, transparent=True, facecolor="none")
    fig.savefig(svg, transparent=True, facecolor="none")
    plt.close(fig)
    return png, svg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    png, svg = plot(load_data(args.metrics_input), args.output, args.dpi)
    print(f"Wrote {png}")
    print(f"Wrote {svg}")


if __name__ == "__main__":
    main()
