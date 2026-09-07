"""Figures for the write-up.

Colour follows the validated categorical order (blue, orange, aqua) and is
assigned by entity, never cycled by rank.  Every series is direct-labelled --
partly because it is easier to read, partly because the aqua slot sits below
3:1 contrast on a light surface and therefore requires visible labels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # fixed order: blue, orange, aqua

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.edgecolor": GRID,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "figure.dpi": 140,
    }
)


def _style(ax, title=None, subtitle=None, xlabel=None, ylabel=None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", pad=18 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color=INK_SECONDARY)
    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")
    return ax


def plot_model_comparison(table, path: str | Path):
    """v1's two numbers against the correct one, for the same question."""
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    labels = [f"{int(p*100)}%" for p in table["p_win"]]
    x = range(len(labels))
    width = 0.26
    bars = [
        ("v1 “math model”", table["v1_math_7plus"], SERIES[0]),
        ("v1 “simulation”", table["v1_sim_7plus"], SERIES[1]),
        ("Correct conditional model", table["correct_conditional_7plus"], SERIES[2]),
    ]
    for k, (name, values, colour) in enumerate(bars):
        offset = (k - 1) * (width + 0.02)
        ax.bar([i + offset for i in x], values, width, label=name, color=colour, zorder=3)
        for i, v in zip(x, values):
            ax.text(i + offset, v + 1.2, f"{v:.1f}", ha="center", fontsize=9, color=INK_SECONDARY)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(table["v1_sim_7plus"]) * 1.22)
    _style(
        ax,
        "The same question, three different answers",
        "P(a team capped at 2 consecutive wins reaches 7+ wins in 14 matches)",
        "Per-match win probability",
        "Probability (%)",
    )
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_qualification_curve(curves, path: str | Path):
    """What a season win total is actually worth, by league size."""
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for k, (n_teams, grp) in enumerate(curves.groupby("n_teams")):
        grp = grp[grp.wins.between(4, 12)]
        colour = SERIES[k]
        label = f"{n_teams} teams, 4 spots"
        ax.plot(grp.wins, grp.qualify_rate * 100, marker="o", markersize=7,
                linewidth=2, color=colour, label=label, zorder=3)
        anchor = grp[grp.wins == 8].iloc[0]
        ax.annotate(
            f"8 wins → {anchor.qualify_rate*100:.0f}%",
            xy=(8, anchor.qualify_rate * 100),
            xytext=(8.6, 84 if k == 0 else 62),
            color=INK_SECONDARY, fontsize=9.5,
            arrowprops=dict(arrowstyle="-", color=GRID, linewidth=1.2),
        )
    ax.axhline(50, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 4)), zorder=1)
    ax.text(4.05, 52, "coin flip", fontsize=9, color=INK_MUTED)
    ax.set_xticks(range(4, 13))
    ax.set_ylim(-4, 104)
    _style(
        ax,
        "Seven wins stopped being a playoff ticket in 2022",
        "Simulated leagues, Bradley-Terry strengths (σ = 0.4); 4 playoff spots throughout",
        "League-stage wins (of 14)",
        "Chance of finishing top four (%)",
    )
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_confound(marginal, stratified, path: str | Path):
    """The streak effect before and after controlling for the win total."""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharey=True)

    ax = axes[0]
    x = range(len(marginal))
    ax.bar([i - 0.2 for i in x], marginal.qualify_rate_with * 100, 0.38,
           color=SERIES[0], label="with the streak", zorder=3)
    ax.bar([i + 0.2 for i in x], marginal.qualify_rate_without * 100, 0.38,
           color=SERIES[1], label="without it", zorder=3)
    for i, row in enumerate(marginal.itertuples()):
        ax.text(i - 0.2, row.qualify_rate_with * 100 + 1.8, f"{row.qualify_rate_with*100:.0f}",
                ha="center", fontsize=9, color=INK_SECONDARY)
        ax.text(i + 0.2, row.qualify_rate_without * 100 + 1.8, f"{row.qualify_rate_without*100:.0f}",
                ha="center", fontsize=9, color=INK_SECONDARY)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{int(k)}+ in a row" for k in marginal.streak_threshold])
    ax.set_ylim(0, 112)
    _style(ax, "Looks decisive…", "Qualification rate, all teams pooled",
           "Longest win streak", "Chance of finishing top four (%)")
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)

    ax = axes[1]
    for k, wins in enumerate((7, 8, 9)):
        grp = stratified[stratified.wins == wins]
        if grp.empty:
            continue
        ax.errorbar(grp.max_streak, grp.qualified_rate * 100, yerr=grp.se * 200,
                    marker="o", markersize=7, linewidth=2, color=SERIES[k],
                    capsize=3, zorder=3)
        last = grp.iloc[-1]
        ax.text(last.max_streak + 0.18, last.qualified_rate * 100,
                f"{wins} wins", color=INK_SECONDARY, fontsize=9.5, va="center")
    ax.set_xlim(1.5, 9)
    _style(ax, "…and vanishes when you hold wins fixed",
           "Same teams, split by season win total (±2 SE)", "Longest win streak", None)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_streak_frequency(freq, path: str | Path):
    """How often streaks happen at all, by team strength."""
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for k, (p, grp) in enumerate(freq.groupby("p_win")):
        label = {0.4: "Weak team (40%)", 0.5: "Average (50%)", 0.6: "Strong (60%)"}[p]
        ax.plot(grp.streak_length, grp.probability, marker="o", markersize=7,
                linewidth=2, color=SERIES[k], label=label, zorder=3)
        # Short value labels only, at k=3 -- long text here runs into the next line.
        anchor = grp[grp.streak_length == 3].iloc[0]
        ax.text(3.10, anchor.probability + 2.5, f"{anchor.probability:.0f}%",
                color=INK_SECONDARY, fontsize=9.5, va="bottom")
    ax.axvline(3, color=GRID, linewidth=1, zorder=1)
    ax.set_xticks(range(1, 7))
    ax.set_ylim(0, 108)
    _style(ax, "A 3-match win streak is the norm, not a distinguishing feat",
           "Exact probabilities for 14 independent matches — no momentum assumed",
           "Streak length k", "P(longest win streak ≥ k)  (%)")
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
