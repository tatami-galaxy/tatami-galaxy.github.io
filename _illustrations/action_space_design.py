"""Generate two minimal action-space figures using Python's standard library.

Run: python3 _illustrations/action_space_design.py
Sources: https://arxiv.org/html/2004.00980v2 (Doom and Obstacle Tower)
and https://arxiv.org/html/1705.07269v1 (FAR and Appendix C).
Controls are schematic; outcome statements and the 9/14 task count are
reported findings. The task-count strip is not an effect-size chart.
"""

from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "images/llm-rl"
INK, MUTED, LINE = "#23313b", "#63717c", "#c3ccd2"
BLUE, BLUE_BG = "#547ba0", "#edf3f8"
GREEN, GREEN_BG = "#508a7c", "#dceee7"


class Figure:
    def __init__(self, height, title, desc):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{height}" viewBox="0 0 1120 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>',
            '<defs><marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 8 4 L 0 8 Z" fill="#8998a3"/></marker></defs>',
            f'<rect width="1120" height="{height}" fill="white"/>',
            '<g font-family="Arial, Helvetica, sans-serif">',
        ]

    def text(self, x, y, value, size=20, color=INK, anchor="middle", extra=""):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" {extra}>{escape(value)}</text>')

    def path(self, d, color=LINE, arrow=False, extra=""):
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.5"{marker} {extra}/>')

    def box(self, x, y, w, h, fill="white", stroke=LINE):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{fill}" stroke="{stroke}"/>')

    def tile(self, x, y, w, label, color=BLUE, bg=BLUE_BG, size=17, h=44):
        self.box(x, y, w, h, bg, color)
        self.text(x+w/2, y+h/2+6, label, size, color)

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        file = OUT / name
        file.write_text("\n".join(self.parts + ['</g></svg>']) + "\n")
        print(file)


def shaping():
    f = Figure(560, "The useful action set depends on the task",
               "Two reported examples from Action Space Shaping in Deep Reinforcement Learning. In Obstacle Tower, allowing backward movement slowed learning. In harder Doom scenarios, expanding the available controls improved performance. The control diagrams are schematic, not complete action-space specifications.")
    f.text(408, 49, "Initial controls", 20, BLUE)
    f.text(827, 49, "Modified controls", 20, GREEN)
    f.text(52, 128, "Obstacle Tower", 21, anchor="start")
    f.text(52, 156, "Remove backward", 16, MUTED, "start")
    f.tile(272, 108, 144, "Other controls")
    f.tile(432, 108, 112, "Backward")
    f.path("M 568 130 H 632", arrow=True)
    f.tile(655, 108, 144, "Other controls", GREEN, GREEN_BG)
    f.tile(815, 108, 112, "Backward", "#9aa5ad", "#f7f8f9")
    f.path("M 822 145 L 920 115", "#9aa5ad")
    f.text(793, 206, "Faster learning", 22, GREEN)
    f.path("M 52 252 H 1068", "#e4e9ec")

    f.text(52, 343, "Doom", 21, anchor="start")
    f.text(52, 371, "Add movement options", 16, MUTED, "start")
    for x, label in [(272, "Forward"), (366, "Turn L/R"), (460, "Attack")]:
        f.tile(x, 323, 84, label, size=16)
    f.path("M 568 345 H 632", arrow=True)
    for i, label in enumerate(["Forward", "Turn L/R", "Attack", "Backward", "Strafe L", "Strafe R"]):
        f.tile(655 + (i % 3)*112, 298 + (i // 3)*58, 100, label,
               GREEN if i >= 3 else BLUE, GREEN_BG if i >= 3 else BLUE_BG, 16)
    f.text(817, 449, "Better performance in harder scenarios", 21, GREEN)
    f.text(560, 526, "The useful action set depends on the task", 22)
    f.save("action-space-shaping.svg")


def factoring():
    f = Figure(690, "The same actions can be learned through different representations",
               "Flat and factored representations of the same 18 Atari actions. Horizontal movement has three choices, vertical movement has three choices, and firing has two choices. Up-right-fire is highlighted in both representations. Shared factor outputs also affect other combinations. The paper reports higher FARA3C scores than A3C on nine of fourteen tasks, with the baseline higher on the other five; marks summarize counts, not effect sizes.")
    f.text(288, 49, "Flat representation", 22, BLUE)
    f.text(832, 49, "Factored representation", 22, GREEN)
    f.path("M 550 75 V 470", "#e4e9ec")
    directions = ["↖", "↑", "↗", "←", "·", "→", "↙", "↓", "↘"]
    for start, name, fire in [(68, "No fire", False), (310, "+ fire", True)]:
        f.text(start+86, 106, name, 18, MUTED)
        for i, label in enumerate(directions):
            selected = fire and i == 2
            f.tile(start+(i % 3)*64, 126+(i // 3)*61, 54, label,
                   GREEN if selected else BLUE, GREEN_BG if selected else BLUE_BG, 26, 48)
    f.text(282, 347, "18 composite actions", 21, BLUE)
    f.text(282, 381, "One output score per action", 17, MUTED)

    for row, (label, choices, chosen) in enumerate([
        ("Horizontal", ["←", "·", "→"], 2),
        ("Vertical", ["↑", "·", "↓"], 0),
        ("Fire", ["off", "on"], 1),
    ]):
        y = 121 + row*72
        f.text(588, y+29, label, 17, MUTED, "start")
        for i, choice in enumerate(choices):
            f.tile(716+i*90, y, 74, choice, GREEN if i == chosen else MUTED,
                   GREEN_BG if i == chosen else "#f7f8f9", 22)
        end = 716+(len(choices)-1)*90+74
        f.path(f"M {end+8} {y+22} H 1020", GREEN)
    f.path("M 1020 143 V 335 H 842 V 353", GREEN, True)
    f.tile(722, 365, 240, "Up + right + fire", GREEN, GREEN_BG, 21, 48)
    f.text(842, 451, "3 × 3 × 2 = 18 actions", 21, GREEN)

    f.path("M 68 487 H 1052", "#e4e9ec")
    f.text(560, 526, "FARA3C outperformed A3C on 9 of 14 Atari tasks", 22)
    for i in range(14):
        f.box(355+i*30, 549, 20, 20, GREEN if i < 9 else "#dce2e6", "none")
    f.text(560, 600, "9 favored FARA3C · 5 favored A3C", 17, MUTED)
    f.text(560, 659, "The same actions can be learned through different representations", 22)
    f.save("factored-action-representations.svg")


if __name__ == "__main__":
    shaping()
    factoring()
