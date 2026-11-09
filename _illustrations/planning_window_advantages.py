"""Render the original token heatmap with variable-length action windows.

Run: python3 _illustrations/planning_window_advantages.py
Text, token positions, font, and palette match token_advantages.py.
Window boundaries and shared advantages are hand-chosen illustrations;
values are neither learned nor averages of the original token advantages.
"""

from html import escape
from pathlib import Path

from token_advantages import WIDTH, MARGIN, CHAR, FONT, LINE, TOKENS, color, label

OUT = Path(__file__).resolve().parents[1] / "images/llm-rl/planning-window-advantages.svg"
# Exclusive token endpoints: windows contain 8, 11, 9, 13, 6, 15, 7, 12 tokens.
WINDOW_ENDS = [8, 19, 28, 41, 47, 62, 69, 81]
ADVANTAGES = [-0.18, 0.30, 0.55, 0.82, -0.12, 0.67, 0.25, 0.90]


def main():
    assert WINDOW_ENDS[-1] == len(TOKENS)
    assert len(WINDOW_ENDS) == len(ADVANTAGES)
    assert all(a < b for a, b in zip([0] + WINDOW_ENDS, WINDOW_ENDS))
    x, y, window = MARGIN, 57, 0
    fragments = []
    texts = []
    for index, token in enumerate(TOKENS):
        if index == WINDOW_ENDS[window]:
            window += 1
        width = len(token) * CHAR
        if x + width > WIDTH - MARGIN:
            x, y = MARGIN, y + LINE
        if not fragments or fragments[-1]['window'] != window or fragments[-1]['y'] != y:
            fragments.append(dict(window=window, x=x, y=y, width=width))
        else:
            fragments[-1]['width'] += width
        texts.append(label(f"{x:.1f}", y, token, FONT, "#172b36",
                           f'font-family="monospace" xml:space="preserve" textLength="{width:.1f}" lengthAdjust="spacingAndGlyphs"'))
        x += width

    backgrounds = []
    for fragment in fragments:
        window = fragment['window']
        start = 0 if window == 0 else WINDOW_ENDS[window-1]
        count = WINDOW_ENDS[window] - start
        value = ADVANTAGES[window]
        tooltip = f"Action {window+1}: {count} tokens; illustrative shared advantage {value:+.2f}"
        backgrounds.append(
            f'<g data-window="{window+1}"><title>{escape(tooltip)}</title>'
            f'<rect x="{fragment["x"]:.1f}" y="{fragment["y"]-25}" '
            f'width="{fragment["width"]-2:.1f}" height="34" rx="3" '
            f'fill="{color(value)}"/></g>'
        )

    caption_y = y + 66
    height = caption_y + 32
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Each planning window is an action</title>',
        '<desc id="desc">The same sample response as the token-level advantage figure, now grouped into eight variable-length planning windows of 8, 11, 9, 13, 6, 15, 7, and 12 tokens. Each window shares one advantage color, with continuous highlighting within each line. Matching colors continue windows across line breaks. Window boundaries and advantage values are illustrative, not learned.</desc>',
        f'<rect width="{WIDTH}" height="{height}" fill="white"/>',
        '<g font-family="Arial, Helvetica, sans-serif">',
        *backgrounds,
        *texts,
        label(WIDTH/2, caption_y, "Each planning window is an action", 22, "#172b36", 'text-anchor="middle"'),
        '</g></svg>',
    ]
    OUT.write_text("\n".join(svg) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
