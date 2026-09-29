"""Build current README figures from the frozen v3 synthetic test aggregates.

This script only reads the saved aggregate result. It never opens test rows,
re-fits a model, or changes the frozen evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULT_PATH = ROOT / "reports/generalization_v3/independent_results.json"
OUT = ROOT / "reports"
RESULTS = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
BY_MODEL = {row["model"]: row for row in RESULTS["metrics"]}
PALETTE = {"model": "#315B78", "rule": "#A56332", "muted": "#73808C"}


def comparison_plot() -> None:
    rows = [
        ("two_stage", "Selected two-stage model", PALETTE["model"]),
        ("random_forest", "Random Forest", "#648B71"),
        ("original_hyperparameters_refit", "Original XGBoost settings\nrefit on v3 target", "#8C78A6"),
        ("physics_rule", "Fixed 15/35 mm rule", PALETTE["rule"]),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.8), layout="constrained")
    fields = (
        ("critical_recall", "Critical recall", (0.90, 1.005), "%"),
        ("normal_fpr", "Normal false-positive rate", (0, 0.006), "%"),
        ("macro_pr_auc", "Macro average precision", (0.90, 1.005), ""),
        ("macro_f1", "Macro F1", (0.82, 1.005), ""),
    )
    labels = [item[1] for item in rows]
    y = np.arange(len(rows))
    for ax, (key, title, limits, suffix) in zip(axes, fields, strict=True):
        values = [BY_MODEL[name][key] for name, _, _ in rows]
        colors = [color for _, _, color in rows]
        bars = ax.barh(y, values, color=colors, height=0.62)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(*limits)
        ax.set_title(title)
        ax.grid(axis="x", color="#d9dee5", linewidth=0.7)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values, strict=True):
            text = f"{value * 100:.3f}%" if suffix == "%" else f"{value:.3f}"
            ax.text(value + (limits[1] - limits[0]) * 0.018, bar.get_y() + bar.get_height() / 2,
                    text, va="center", fontsize=8)
    axes[0].set_xlabel("Higher is better")
    axes[1].set_xlabel("Lower is better")
    axes[2].set_xlabel("Higher is better")
    axes[3].set_xlabel("Higher is better")
    figure.suptitle("v3 local-severity task · independent synthetic test", fontsize=14, weight="bold")
    figure.text(0.5, -0.015,
                "Same 8,640 windows · 960 events · 480 generating groups. Prototype severity labels; not field validation.",
                ha="center", fontsize=9, color="#45515c")
    figure.savefig(OUT / "readme_v3_model_comparison.png", dpi=190, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def confusion_plot() -> None:
    result = BY_MODEL["two_stage"]
    counts = np.asarray(result["confusion_matrix"], dtype=int)
    rates = counts / counts.sum(axis=1, keepdims=True)
    classes = ("NORMAL", "WARNING", "CRITICAL")
    figure, ax = plt.subplots(figsize=(6.6, 5.5), layout="constrained")
    image = ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    for row in range(3):
        for col in range(3):
            ax.text(col, row, f"{rates[row, col]:.1%}\n(n={counts[row, col]:,})",
                    ha="center", va="center",
                    color="white" if rates[row, col] >= 0.55 else "#17212b", fontsize=10)
    ax.set_xticks(range(3), classes)
    ax.set_yticks(range(3), classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Selected two-stage model · row-normalized test confusion")
    figure.colorbar(image, ax=ax, label="Share within true class")
    figure.text(0.5, -0.02,
                "Local severity: NORMAL ≤15 mm · WARNING >15 to 35 mm · CRITICAL >35 mm. Synthetic only.",
                ha="center", fontsize=8.5, color="#45515c")
    figure.savefig(OUT / "readme_v3_confusion_matrix.png", dpi=190, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def event_timing_plot() -> None:
    rows = RESULTS["events"]["two_stage"]
    persistence = [row["persistence_windows"] for row in rows]
    recall = [row["critical_event_recall"] * 100 for row in rows]
    median = [row["median_detection_delay_hours"] for row in rows]
    p90 = [row["p90_detection_delay_hours"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4), layout="constrained")
    axes[0].plot(persistence, recall, marker="o", color=PALETTE["model"], linewidth=2)
    axes[0].set_xticks(persistence)
    axes[0].set_ylim(90, 100)
    axes[0].set_xlabel("Consecutive windows required")
    axes[0].set_ylabel("Critical-event recall (%)")
    axes[0].set_title("142 Critical events")
    axes[1].plot(persistence, median, marker="o", label="Median", color=PALETTE["model"], linewidth=2)
    axes[1].plot(persistence, p90, marker="s", label="90th percentile", color=PALETTE["rule"], linewidth=2)
    axes[1].set_xticks(persistence)
    axes[1].set_xlabel("Consecutive windows required")
    axes[1].set_ylabel("Delay after 35 mm crossing (hours)")
    axes[1].set_title("Among detected events")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(color="#d9dee5", linewidth=0.7)
        ax.set_axisbelow(True)
    figure.suptitle("Separate local event-alert evaluation · selected two-stage model", fontsize=13, weight="bold")
    figure.text(0.5, -0.015,
                "No false Critical episodes in 798 all-Normal events. Delay is not collapse lead time; 60-step window warmup applies.",
                ha="center", fontsize=8.5, color="#45515c")
    figure.savefig(OUT / "readme_v3_alert_timing.png", dpi=190, facecolor="white", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    comparison_plot()
    confusion_plot()
    event_timing_plot()
