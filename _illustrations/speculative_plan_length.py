"""Generate a schematic of accepted draft length as a plan-length proxy.

Run: python3 _illustrations/speculative_plan_length.py
Inspired by DFlash: https://arxiv.org/html/2602.06036v2.
Illustrative B=5, h=3. h counts accepted draft tokens only, excluding
target-generated correction/bonus tokens. The fifth token is discarded
after the first rejection, not independently classified as rejected.
"""

from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "images/llm-rl/speculative-plan-length.svg"
INK, MUTED, LINE = "#23313b", "#63717c", "#b9c3ca"
BLUE, BLUE_BG = "#547ba0", "#edf3f8"
GREEN, GREEN_BG = "#508a7c", "#dceee7"
RED, RED_BG = "#af6657", "#f7e4de"
CENTERS = [280, 420, 560, 700, 840]
TOKENS = ["d₁", "d₂", "d₃", "d₄", "d₅"]
parts = []


def text(x, y, value, size=20, color=INK, extra=""):
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="middle" {extra}>{escape(value)}</text>')


def rect(x, y, width, height, fill="white", stroke=LINE):
    parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')


def path(d, color=LINE, arrow=False, extra=""):
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.5"{marker} {extra}/>')


def main():
    parts.clear()
    parts.extend([
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="710" viewBox="0 0 1120 710" role="img" aria-labelledby="title desc">',
        '<title id="title">Accepted draft length as a proxy for plan length</title>',
        '<desc id="desc">The target LLM extracts context features from a prefix to condition a diffusion drafter. The drafter proposes five tokens in parallel. The target verifies the draft: three consecutive tokens are accepted, the fourth is the first rejection, and the fifth is discarded. The three accepted draft tokens define h, an imperfect candidate plan length. This is a schematic example, not a measured planning horizon. Correction and bonus tokens are omitted.</desc>',
        '<defs><marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 8 4 L 0 8 Z" fill="#8998a3"/></marker></defs>',
        '<rect width="1120" height="710" fill="white"/>',
        '<g font-family="Arial, Helvetica, sans-serif">',
    ])

    rect(125, 40, 180, 50, BLUE_BG, BLUE)
    text(215, 72, "Prefix", color=BLUE)
    path("M 318 65 H 377", arrow=True)
    rect(390, 40, 220, 50, BLUE_BG, BLUE)
    text(500, 72, "Target LLM", color=BLUE)
    path("M 623 65 H 687", arrow=True)
    rect(700, 40, 260, 50, BLUE_BG, BLUE)
    text(830, 72, "Context features", color=BLUE)
    path("M 830 101 V 124 H 560 V 145", BLUE, True)

    rect(430, 158, 260, 52)
    text(560, 191, "Diffusion drafter", 22)
    path("M 560 221 V 238", arrow=True)
    text(560, 264, "B draft tokens · generated in parallel", 19, MUTED)
    path("M 230 290 V 280 H 890 V 290")
    for cx, token in zip(CENTERS, TOKENS):
        rect(cx - 50, 302, 100, 48, "#f7f8f9")
        text(cx, 333, token, 24)
        path(f"M {cx} 360 V 386", arrow=True)

    rect(230, 398, 660, 48, BLUE_BG, BLUE)
    text(560, 429, "Target LLM verification", 21, BLUE)
    for cx in CENTERS:
        path(f"M {cx} 458 V 484", arrow=True)

    for index, (cx, token) in enumerate(zip(CENTERS, TOKENS)):
        color, bg = (GREEN, GREEN_BG) if index < 3 else (RED, RED_BG) if index == 3 else ("#96a1a9", "#f7f8f9")
        rect(cx - 50, 496, 100, 48, bg, color)
        text(cx, 527, token, 24, color)
        if index < 3:
            path(f"M {cx-7} 563 L {cx-2} 568 L {cx+8} 556", GREEN, extra='stroke-linecap="round" stroke-linejoin="round"')
        elif index == 3:
            path(f"M {cx-6} 557 L {cx+6} 569 M {cx+6} 557 L {cx-6} 569", RED, extra='stroke-linecap="round"')
        else:
            text(cx, 569, "Discarded", 17, MUTED)

    path("M 230 584 V 594 H 610 V 584", GREEN)
    text(420, 623, "h accepted · candidate plan length", 20, GREEN)
    text(700, 608, "First rejection", 17, RED)
    text(560, 680, "Accepted draft length as a proxy for plan length", 22)
    parts.append('</g></svg>')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
