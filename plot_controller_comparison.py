"""Create final tables and plots for the four-controller comparison."""

from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONTROLLERS = ["PD", "PD+BRT", "PPO", "PPO+BRT"]
DISPLAY_NAMES = ["PD", "PD + Safety", "PPO", "PPO + Safety"]
COLORS = ["#7f8c8d", "#2e86de", "#9b59b6", "#16a085"]
CAUSE_COLORS = {
    "goal": "#2ca02c",
    "spill": "#d62728",
    "obstacle": "#ff7f0e",
    "joint": "#9467bd",
    "ground": "#8c564b",
    "timeout": "#7f7f7f",
}
SLOSH_LIMIT_MM = 7.388005166533489


def load_results(path):
    with open(path) as handle:
        return json.load(handle)


def write_tables(results, output_dir):
    rows = []
    for key, name in zip(CONTROLLERS, DISPLAY_NAMES):
        summary = results[key]["summary"]
        causes = summary["causes"]
        rows.append(
            {
                "Controller": name,
                "Goals": causes["goal"],
                "Success Rate": 100 * summary["completion_rate"],
                "Spills": causes["spill"],
                "Obstacle Hits": causes["obstacle"],
                "Mean Peak Slosh (mm)": 1000 * summary["mean_peak_slosh"],
                "Mean Min. Goal Distance (m)": summary["mean_minimum_goal_distance"],
                "Mean Reward": summary["mean_reward"],
                "Intervention Rate": 100 * summary["mean_intervention_rate"],
            }
        )

    csv_path = os.path.join(output_dir, "controller_comparison_table.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = os.path.join(output_dir, "controller_comparison_table.md")
    with open(markdown_path, "w") as handle:
        handle.write(
            "| Controller | Goals | Success | Spills | Obstacle Hits | "
            "Mean Peak Slosh | Mean Min. Goal Distance | Mean Reward | "
            "Intervention Rate |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['Controller']} | {row['Goals']}/30 | "
                f"{row['Success Rate']:.1f}% | {row['Spills']} | "
                f"{row['Obstacle Hits']} | "
                f"{row['Mean Peak Slosh (mm)']:.2f} mm | "
                f"{row['Mean Min. Goal Distance (m)']:.3f} m | "
                f"{row['Mean Reward']:.2f} | "
                f"{row['Intervention Rate']:.1f}% |\n"
            )
    return rows


def style_axis(ax):
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def create_summary_plot(results, output_dir):
    x = np.arange(len(CONTROLLERS))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    success = [100 * results[key]["summary"]["completion_rate"] for key in CONTROLLERS]
    axes[0, 0].bar(x, success, color=COLORS, width=0.72)
    axes[0, 0].set_ylabel("Completion rate (%)")
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].set_title("(a) Goal completion")
    for index, value in enumerate(success):
        axes[0, 0].text(index, value + 2, f"{value:.1f}%", ha="center")
    style_axis(axes[0, 0])

    causes = ["goal", "spill", "obstacle", "joint", "ground", "timeout"]
    bottom = np.zeros(len(CONTROLLERS))
    for cause in causes:
        values = [
            100
            * results[key]["summary"]["causes"][cause]
            / results[key]["summary"]["episodes"]
            for key in CONTROLLERS
        ]
        axes[0, 1].bar(
            x,
            values,
            bottom=bottom,
            color=CAUSE_COLORS[cause],
            label=cause.capitalize(),
            width=0.72,
        )
        bottom += np.asarray(values)
    axes[0, 1].set_ylabel("Episodes (%)")
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].set_title("(b) Terminal outcomes")
    axes[0, 1].legend(
        ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13)
    )
    style_axis(axes[0, 1])

    slosh_data = [
        1000
        * np.asarray([episode["peak_slosh"] for episode in results[key]["episodes"]])
        for key in CONTROLLERS
    ]
    box = axes[1, 0].boxplot(
        slosh_data,
        patch_artist=True,
        widths=0.58,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, color in zip(box["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[1, 0].axhspan(
        SLOSH_LIMIT_MM,
        max(SLOSH_LIMIT_MM * 1.18, max(map(np.max, slosh_data)) * 1.08),
        color="#d62728",
        alpha=0.12,
        label="Spill region",
    )
    axes[1, 0].axhline(
        SLOSH_LIMIT_MM,
        color="#d62728",
        linestyle="--",
        linewidth=1.4,
        label=f"Spill threshold ({SLOSH_LIMIT_MM:.2f} mm)",
    )
    axes[1, 0].set_ylabel("Peak slosh displacement (mm)")
    axes[1, 0].set_title("(c) Peak slosh by episode")
    axes[1, 0].legend(fontsize=8, loc="upper right")
    style_axis(axes[1, 0])

    goal_data = [
        np.asarray(
            [episode["minimum_goal_distance"] for episode in results[key]["episodes"]]
        )
        for key in CONTROLLERS
    ]
    box = axes[1, 1].boxplot(
        goal_data,
        patch_artist=True,
        widths=0.58,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, color in zip(box["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[1, 1].axhspan(0, 0.1, color="#2ca02c", alpha=0.12)
    axes[1, 1].axhline(
        0.1,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.4,
        label="Goal threshold (0.1 m)",
    )
    axes[1, 1].set_ylabel("Minimum distance to goal (m)")
    axes[1, 1].set_title("(d) Closest approach to goal")
    axes[1, 1].legend(fontsize=8, loc="upper right")
    style_axis(axes[1, 1])

    for ax in axes.flat:
        ax.set_xticks(x, DISPLAY_NAMES)

    fig.suptitle(
        "PD and PPO Baselines With and Without the BRT Safety Filter\n"
        "30 matched seeds, fixed-horizon BRT, margin = 0.005",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.6, w_pad=2.0)

    png_path = os.path.join(output_dir, "controller_comparison_summary.png")
    pdf_path = os.path.join(output_dir, "controller_comparison_summary.pdf")
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)


def create_intervention_plot(results, output_dir):
    filtered = ["PD+BRT", "PPO+BRT"]
    names = ["PD + Safety", "PPO + Safety"]
    rates = [
        100
        * np.asarray(
            [episode["intervention_rate"] for episode in results[key]["episodes"]]
        )
        for key in filtered
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    box = ax.boxplot(
        rates,
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, color in zip(box["boxes"], [COLORS[1], COLORS[3]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for index, values in enumerate(rates, start=1):
        jitter = np.linspace(-0.09, 0.09, len(values))
        ax.scatter(
            index + jitter,
            values,
            color="black",
            alpha=0.55,
            s=18,
            zorder=3,
        )
    ax.set_xticks([1, 2], names)
    ax.set_ylabel("Safety-filter intervention rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("How often the BRT changed the nominal action")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "controller_intervention_rates.png"), dpi=220)
    fig.savefig(os.path.join(output_dir, "controller_intervention_rates.pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="controller_results_margin_0p0050.json")
    parser.add_argument(
        "--output-dir", default=os.path.join("figures", "final_comparison")
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = load_results(args.results)
    rows = write_tables(results, args.output_dir)
    create_summary_plot(results, args.output_dir)
    create_intervention_plot(results, args.output_dir)

    print(json.dumps(rows, indent=2))
    print(f"Saved final comparison artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
