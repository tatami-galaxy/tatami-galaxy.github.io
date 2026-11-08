"""Generate the illustrative SVG with Python's standard library.

Run: python3 _illustrations/token_advantages.py
Token boundaries and signed advantages are illustrative, not model outputs.
Edit SEGMENTS to change the text, token boundaries (|), or advantage pattern.
This underscore-prefixed source directory is excluded by Jekyll by default.
"""

from html import escape
from pathlib import Path
import random


SEGMENTS = [
    "Let| us| solve| this| step| by| step|.| A| train| travels| ",
    "120| km| in| ",
    "2| hours|.| Its| average| speed| is| distance| divided| by| time|.| So| we| compute| ",
    "120| /| 2| =| 60| km|/|h|.| ",
    "To| check| the| result|,| multiply| the| speed| by| the| time|:| ",
    "60| times| 2| gives| 120| km|,| ",
    "which| matches| the| original| distance|.| Therefore|,| the| average| speed| is| ",
    "60| km|/|h|.",
]

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images" / "llm-rl" / "token-advantages.svg"
WIDTH, MARGIN, CHAR, FONT, LINE = 1100, 48, 13.2, 22, 46
TOKENS = [token for segment in SEGMENTS for token in segment.split("|")]
rng = random.Random(17)
VALUES = [rng.uniform(-1, 1) for _ in TOKENS]


def color(value):
    neutral = (248, 248, 245)
    end = (112, 190, 166) if value >= 0 else (232, 147, 131)
    return "#" + "".join(f"{round(a + abs(value) * (b - a)):02x}" for a, b in zip(neutral, end))


def label(x, y, text, size=18, fill="#53616b", extra=""):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" {extra}>{escape(text)}</text>'


def main():
    x, y = MARGIN, 176
    shapes = []
    for index, (token, value) in enumerate(zip(TOKENS, VALUES)):
        width = len(token) * CHAR
        if x + width > WIDTH - MARGIN:
            x, y = MARGIN, y + LINE
        shapes.append(
            f'<g><title>Token {index + 1}: {escape(repr(token))}; advantage {value:+.3f}</title>'
            f'<rect x="{x:.1f}" y="{y - 25}" width="{width - 1:.1f}" height="34" '
            f'rx="2" fill="{color(value)}"/>'
            + label(f"{x:.1f}", y, token, FONT, "#172b36",
                    f'font-family="monospace" xml:space="preserve" textLength="{width:.1f}" lengthAdjust="spacingAndGlyphs"')
            + '</g>'
        )
        x += width

    legend_y = y + 64
    height = legend_y + 116
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Token-level advantages: one token, one action</title>',
        '<desc id="desc">A sample response about a train travelling 120 kilometres in two hours. Each illustrative token has a synthetic signed advantage. Coral is negative, off-white is zero, and green is positive. Values do not come from a training run.</desc>',
        f'<rect width="{WIDTH}" height="{height}" fill="white"/>',
        '<g font-family="Arial, Helvetica, sans-serif">',
        label(MARGIN, 51, "One token, one action", 30, "#172b36", 'font-weight="600"'),
        label(MARGIN, 85, "A separate advantage for each generated token", 20),
        f'<path d="M{MARGIN} 112 H{WIDTH-MARGIN}" stroke="#dce3e6"/>',
        *shapes,
        label(MARGIN, legend_y + 17, "Advantage", 18, "#172b36"),
        '<defs><linearGradient id="scale"><stop offset="0" stop-color="'+color(-1)+'"/><stop offset="0.5" stop-color="'+color(0)+'"/><stop offset="1" stop-color="'+color(1)+'"/></linearGradient></defs>',
        f'<rect x="180" y="{legend_y}" width="330" height="20" rx="3" fill="url(#scale)"/>',
        label(180, legend_y + 44, "−1", 16),
        label(345, legend_y + 44, "0", 16, extra='text-anchor="middle"'),
        label(510, legend_y + 44, "+1", 16, extra='text-anchor="end"'),
        label(560, legend_y + 17, "Negative  ←  zero  →  positive", 18),
        label(MARGIN, height - 26, "Schematic · synthetic advantages · illustrative token boundaries", 17),
        '</g></svg>',
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(svg) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
