"""Render a schematic of the compressed-representation horizon analysis.

Run: python3 _illustrations/planning_horizon.py
Based on section 3.1 of https://arxiv.org/html/2509.25260v2.
One rollout illustrates the selected activations; MI is estimated across
many rollouts. No measured scores or individual planning horizon are shown.
"""

from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "images/llm-rl/planning-horizon.svg"
INK, MUTED, LINE = "#23313b", "#63717c", "#b9c3ca"
BLUE, BLUE_BG = "#547ba0", "#edf3f8"
GREEN, GREEN_BG = "#508a7c", "#edf5f2"
parts = []


def text(x, y, value, size=19, color=INK, anchor="middle", extra=""):
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" {extra}>{escape(value)}</text>')


def rect(x, y, w, h, fill="white", stroke=LINE, radius=6):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>')


def path(d, color=LINE, arrow=False, extra=""):
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.5"{marker} {extra}/>')


def compress(cx, color, bg):
    path(f"M {cx} 321 V 346", color, True)
    parts.append(f'<path d="M {cx-55} 359 H {cx+55} L {cx+27} 395 H {cx-27} Z" fill="{bg}" stroke="{color}"/>')
    text(cx, 381, "VQ", 17, color)
    path(f"M {cx} 396 V 425", color, True)


def main():
    parts.clear()
    parts.extend([
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="650" viewBox="0 0 1120 650" role="img" aria-labelledby="title desc">',
        '<title id="title">Shared information between prefix and future computations</title>',
        '<desc id="desc">A completed rollout is split at position t. Prefix activations across positions 1 through t and layers 1 through L minus 1 are compressed into one discrete code. Final-layer states at later positions t plus 1, t plus 2, and t plus k are compressed separately. Code co-occurrences across many rollouts are used to estimate mutual information. The diagram does not assign a planning horizon to an individual rollout.</desc>',
        '<defs><marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 8 4 L 0 8 Z" fill="#8998a3"/></marker></defs>',
        '<rect width="1120" height="650" fill="white"/>',
        '<g font-family="Arial, Helvetica, sans-serif">',
    ])

    text(252, 39, "Prefix", 19, MUTED)
    text(796, 39, "Future continuation", 19, MUTED)
    for x, value in [(64, "x₁"), (162, "…"), (260, "xₜ₋₁"), (358, "xₜ")]:
        rect(x, 65, 88, 48, BLUE_BG, BLUE if value == "xₜ" else "#cfdae4")
        text(x + 44, 96, value, 23, BLUE)
    for cx, value in [(610, "xₜ₊₁"), (790, "xₜ₊₂"), (1000, "xₜ₊ₖ")]:
        rect(cx - 48, 65, 96, 48, GREEN_BG, "#cbded7")
        text(cx, 96, value, 23, GREEN)
    text(895, 96, "…", 23, MUTED)
    path("M 490 53 V 156", LINE, extra='stroke-dasharray="4 5"')
    text(490, 178, "position t", 16, MUTED)

    # The wide prefix block represents many positions and non-final layers.
    path("M 64 127 V 138 H 446 V 127", BLUE)
    path("M 255 138 V 188", BLUE, True)
    for row in range(4):
        for col in range(9):
            rect(153 + col * 23, 233 + row * 18, 19, 13,
                 BLUE_BG if (row + col) % 3 else "#cbdceb", "none", 2)
    text(255, 213, "Prefix activations", 19, BLUE)
    text(255, 319, "positions ≤ t · layers < L", 16, MUTED)

    text(804, 213, "Later decision states · final layer", 19, GREEN)
    for cx, value in [(610, "hₜ₊₁"), (790, "hₜ₊₂"), (1000, "hₜ₊ₖ")]:
        path(f"M {cx} 126 V 188", GREEN, True)
        for row in range(4):
            rect(cx - 22, 233 + row * 18, 44, 13,
                 GREEN_BG if row % 2 else "#c9e0d7", "none", 2)
        text(cx, 319, value, 19, GREEN)

    # Two families of learned compressors: prefix blocks and individual states.
    for cx, color, bg, value in [
        (255, BLUE, BLUE_BG, "Zₚ"),
        (610, GREEN, GREEN_BG, "Z₁"),
        (790, GREEN, GREEN_BG, "Z₂"),
        (1000, GREEN, GREEN_BG, "Zₖ"),
    ]:
        compress(cx, color, bg)
        rect(cx - 46, 438, 92, 44, bg, color)
        text(cx, 467, value, 21, color)
    text(895, 467, "…", 23, MUTED)

    # Undirected comparisons, not prediction or causal arrows.
    path("M 255 482 V 525 H 1000 V 482", LINE)
    path("M 610 482 V 525 M 790 482 V 525", LINE)
    text(627, 558, "Estimate shared information at each future offset", 20)
    text(627, 586, "from code co-occurrences across many rollouts", 17, MUTED)
    text(560, 629, "How much information about future computations is already present in the prefix?", 21)
    parts.append('</g></svg>')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
