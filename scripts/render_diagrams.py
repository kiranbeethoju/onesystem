#!/usr/bin/env python3
"""Render pipeline + architecture PNGs for README and GitHub Pages.

Usage:
    python scripts/render_diagrams.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = [ROOT / "docs" / "assets", ROOT / "assets"]

INK = "#12151a"
MUTED = "#5c6570"
PAPER = "#f3efe6"
CARD = "#fffdf8"
ACCENT = "#0f6e56"
ACCENT_SOFT = "#d7ebe3"
LINE = "#c9c2b4"
WARN = "#b45309"


def _setup(fig):
    fig.patch.set_facecolor(PAPER)


def _box(ax, xy, w, h, text, *, face=CARD, edge=LINE, title_color=INK, sub=None, lw=1.2):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        mutation_aspect=0.5,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2 + (0.06 if sub else 0),
        text, ha="center", va="center",
        fontsize=11, fontweight="bold", color=title_color,
        fontfamily="sans-serif",
    )
    if sub:
        ax.text(
            x + w / 2, y + h / 2 - 0.14,
            sub, ha="center", va="center",
            fontsize=8.5, color=MUTED, fontfamily="sans-serif",
        )
    return patch


def _arrow(ax, start, end, color=MUTED):
    ax.add_patch(
        FancyArrowPatch(
            start, end,
            arrowstyle="-|>", mutation_scale=12,
            linewidth=1.3, color=color,
            shrinkA=2, shrinkB=2,
        )
    )


def render_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 7.2), dpi=160)
    _setup(fig)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(
        "In front of the LLM — not instead of it",
        fontsize=15, fontweight="bold", color=INK, pad=10, fontfamily="sans-serif",
    )
    ax.text(
        5, 9.45,
        "Simple requests get a macro. Complex or abstained requests get a chat / reasoning model.",
        ha="center", va="top", fontsize=9, color=MUTED, fontfamily="sans-serif",
    )

    _box(ax, (3.1, 8.05), 3.8, 0.95, "Incoming request", sub="Ticket, note, or chat turn")
    _arrow(ax, (5, 8.05), (5, 7.55))
    _box(
        ax, (2.6, 6.35), 4.8, 1.15, "OneSystem",
        sub="label · probabilities · calibrated confidence · abstain",
        face=ACCENT_SOFT, edge=ACCENT, lw=1.6,
    )
    _arrow(ax, (5, 6.35), (5, 5.85), color=ACCENT)

    # branch labels
    ax.text(2.55, 5.55, "SIMPLE", ha="center", fontsize=8.5, fontweight="700",
            color=WARN, fontfamily="sans-serif")
    ax.text(7.45, 5.55, "COMPLEX / ABSTAIN", ha="center", fontsize=8.5, fontweight="700",
            color=WARN, fontfamily="sans-serif")

    _arrow(ax, (5, 5.85), (2.55, 5.15), color=WARN)
    _arrow(ax, (5, 5.85), (7.45, 5.15), color=WARN)

    _box(ax, (0.7, 3.85), 3.7, 1.05, "Macro / workflow", sub="Refund, route, freeze…")
    _box(ax, (5.6, 3.85), 3.7, 1.05, "LLM", sub="Reason · tools · draft",
         face="#eef2f6", edge="#8a93a0")
    _arrow(ax, (2.55, 3.85), (2.55, 3.35))
    _arrow(ax, (7.45, 3.85), (7.45, 3.35))
    _box(ax, (0.7, 2.15), 3.7, 1.05, "Done", sub="No LLM call",
         face=ACCENT_SOFT, edge=ACCENT, lw=1.5)
    _box(ax, (5.6, 2.15), 3.7, 1.05, "Plan · tools · draft", sub="Open-ended work",
         face="#eef2f6", edge="#8a93a0")

    ax.text(
        5, 1.35,
        "OneSystem wins on fixed schema + latency + abstain    ·    LLMs win on multi-step policy + drafting",
        ha="center", fontsize=8.2, color=MUTED, fontfamily="sans-serif",
    )
    fig.tight_layout(pad=0.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def render_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.4), dpi=160)
    _setup(fig)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title(
        "How it is built — task-conditioned bi-encoder",
        fontsize=15, fontweight="bold", color=INK, pad=8, fontfamily="sans-serif",
    )

    steps = ["{task}: …", "encoder", "mean", "proj", "normalize"]
    # text lane
    ax.text(0.4, 4.85, "TEXT", fontsize=9, fontweight="700", color=ACCENT, fontfamily="sans-serif")
    y_text = 4.0
    for i, label in enumerate(steps):
        x = 0.35 + i * 1.7
        face = ACCENT_SOFT if i == 0 else CARD
        edge = ACCENT if i == 0 else LINE
        _box(ax, (x, y_text), 1.5, 0.85, label, face=face, edge=edge)
        if i < len(steps) - 1:
            _arrow(ax, (x + 1.5, y_text + 0.42), (x + 1.7, y_text + 0.42), color=ACCENT)

    # label lane
    ax.text(0.4, 2.55, "LABEL", fontsize=9, fontweight="700", color=ACCENT, fontfamily="sans-serif")
    y_lab = 1.7
    label_steps = ["{task}: {label}", "encoder", "mean", "proj", "normalize"]
    for i, label in enumerate(label_steps):
        x = 0.35 + i * 1.7
        face = ACCENT_SOFT if i == 0 else CARD
        edge = ACCENT if i == 0 else LINE
        _box(ax, (x, y_lab), 1.5, 0.85, label, face=face, edge=edge)
        if i < len(label_steps) - 1:
            _arrow(ax, (x + 1.5, y_lab + 0.42), (x + 1.7, y_lab + 0.42), color=ACCENT)

    # merge to score
    nx = 0.35 + 4 * 1.7 + 1.5
    _arrow(ax, (nx, y_text + 0.42), (nx + 0.55, 3.35), color=ACCENT)
    _arrow(ax, (nx, y_lab + 0.42), (nx + 0.55, 3.05), color=ACCENT)
    _box(
        ax, (nx + 0.55, 2.55), 2.7, 1.5,
        "cosine × scale / T",
        sub="softmax · single   |   sigmoid · multi",
        face=ACCENT_SOFT, edge=ACCENT, lw=1.6,
    )

    ax.text(
        6, 0.55,
        "Encoder init: Alibaba-NLP/gte-base-en-v1.5 (8192)  ·  shared identity proj  ·  no PEFT at inference",
        ha="center", fontsize=8.2, color=MUTED, fontfamily="sans-serif",
    )
    fig.tight_layout(pad=0.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def main() -> None:
    for out in OUT_DIRS:
        render_pipeline(out / "pipeline.png")
        render_architecture(out / "architecture.png")
        print(f"wrote {out / 'pipeline.png'}")
        print(f"wrote {out / 'architecture.png'}")


if __name__ == "__main__":
    main()
