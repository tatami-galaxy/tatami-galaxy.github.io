"""Render the response-generation comparison using Manim Community.

Requires Manim Community 0.22.1 (pip install manim==0.22.1).
On macOS, install Cairo and pkg-config first (brew install cairo pkg-config).
Run with Manim installed:
    python _illustrations/iterative_response_generation.py

The MP4 and poster are written to assets/videos/. Intermediate files stay in
a temporary directory. All text is scripted illustration, not a model trace;
animation timing does not represent inference speed or compute usage.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import av
from manim import (
    Scene, Text, RoundedRectangle, Line, Arrow, VMobject,
    Dot, FadeOut, Transform, Create, MoveAlongPath, Succession,
    AddTextLetterByLetter, LEFT, linear, tempconfig,
)

INK = "#23313B"
MUTED = "#63717C"
LINE = "#D9E0E5"
BLUE = "#547BA0"
GREEN = "#357B67"
AMBER = "#AA6C19"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/videos"
PLAYBACK_SPEED = 1.5


def label(value, size=24, color=INK):
    return Text(value, font="Arial", font_size=size, color=color)


def at_left(mobject, x, y):
    return mobject.move_to([x, y, 0], aligned_edge=LEFT)


class IterativeResponseGeneration(Scene):
    def construct(self):
        self.camera.background_color = "#FAFBFC"
        self.add(label("Two ways to build a response", 38).move_to([0, 4.5, 0]))
        self.add(label("Prompt: How should we get to the museum?", 25, MUTED)
                 .move_to([0, 3.87, 0]))
        self.add(Line([0, 3.25, 0], [0, -3.95, 0], color=LINE))
        panels = []
        for cx, title, subtitle, color in [
            (-4, "Standard generation", "Think, then write", BLUE),
            (4, "Iterative generation", "Think, write, revise", GREEN),
        ]:
            self.add(label(title, 30, color).move_to([cx, 3.08, 0]))
            self.add(label(subtitle, 22, MUTED).move_to([cx, 2.61, 0]))
            thinking = RoundedRectangle(width=6.55, height=2.34, corner_radius=.14,
                                        stroke_color=LINE, fill_color="#EEF3F7",
                                        fill_opacity=1).move_to([cx, 1.05, 0])
            self.add(thinking)
            self.add(at_left(label("REASONING", 18, color), cx-2.98, 1.89))
            if cx > 0:
                self.add(at_left(label("+ current document", 17, GREEN), cx-.4, 1.89))
            rows = [at_left(label(s, 24), cx-2.98, 1.35-i*.40) for i, s in enumerate([
                "Compare the routes.", "Walking is direct.",
                "Check the forecast: rain.", "Choose a sheltered route.",
            ])]
            write_arrow = Arrow([cx, -.18, 0], [cx, -.87, 0], buff=.05,
                                color=color, stroke_width=3)
            self.add(write_arrow)
            status = label("Thinking…", 20, color).move_to([cx+1.65, -.52, 0])
            self.add(status)
            document = RoundedRectangle(width=6.55, height=2.66, corner_radius=.14,
                                        stroke_color=LINE, fill_color="#FFFFFF",
                                        fill_opacity=1).move_to([cx, -2.3, 0])
            self.add(document)
            self.add(at_left(label("response.txt", 20, MUTED), cx-2.98, -1.31))
            self.add(Line([cx-2.98, -1.65, 0], [cx+2.98, -1.65, 0], color=LINE))
            empty = at_left(label("Empty", 25, MUTED), cx-2.98, -2.17)
            self.add(empty)
            panels.append(dict(cx=cx, rows=rows, status=status, empty=empty,
                               document=document, arrow=write_arrow))

        left, right = panels
        # The return path terminates in the reasoning/context box.
        feedback = VMobject(color=GREEN, stroke_width=3).set_points_as_corners([
            [7.29, -2.4, 0], [7.65, -2.4, 0], [7.65, .65, 0], [7.29, .65, 0],
        ])
        feedback_tip = Arrow([7.65, .65, 0], [7.23, .65, 0], buff=0,
                             color=GREEN, stroke_width=3, max_tip_length_to_length_ratio=.3)
        sync_text = label("Sync to context", 18, GREEN).move_to([4, -3.97, 0])
        self.add(feedback.set_stroke(opacity=.5), feedback_tip.set_opacity(.5),
                 sync_text.set_opacity(.8))

        def status(panel, text, color):
            return Transform(panel["status"], label(text, 20, color)
                             .move_to(panel["status"].get_center()))

        def think(index):
            self.play(*[AddTextLetterByLetter(p["rows"][index]) for p in panels],
                      *[status(p, "Thinking…", BLUE if p is left else GREEN) for p in panels],
                      run_time=1.3 / PLAYBACK_SPEED)
            self.wait(.6 / PLAYBACK_SPEED)

        def sync():
            pulse = Dot(radius=.065, color=GREEN).move_to(feedback.get_start())
            self.play(feedback.animate.set_stroke(opacity=1), feedback_tip.animate.set_opacity(1),
                      sync_text.animate.set_opacity(1), status(right, "Syncing…", GREEN),
                      run_time=.25 / PLAYBACK_SPEED)
            self.add(pulse)
            self.play(MoveAlongPath(pulse, feedback), run_time=.65 / PLAYBACK_SPEED, rate_func=linear)
            self.remove(pulse)
            self.play(feedback.animate.set_stroke(opacity=.5), feedback_tip.animate.set_opacity(.5),
                      sync_text.animate.set_opacity(.8), run_time=.2 / PLAYBACK_SPEED)

        self.wait(.8 / PLAYBACK_SPEED)
        think(0)
        think(1)
        draft = at_left(label("Walk to the museum.", 27, GREEN), 1.02, -2.17)
        self.play(FadeOut(right["empty"]), status(right, "Adding…", GREEN), run_time=.25 / PLAYBACK_SPEED)
        self.play(AddTextLetterByLetter(draft), run_time=1.25 / PLAYBACK_SPEED)
        self.wait(.7 / PLAYBACK_SPEED)
        sync()
        think(2)
        second = at_left(label("Rain is expected.", 27, GREEN), 1.02, -2.89)
        self.play(status(right, "Adding…", GREEN), AddTextLetterByLetter(second),
                  draft.animate.set_color(INK), run_time=1.2 / PLAYBACK_SPEED)
        sync()
        think(3)

        # Make the change visible: strike out the old sentence, then replace it.
        self.play(status(right, "Revising…", AMBER), draft.animate.set_color(AMBER),
                  second.animate.set_color(INK), run_time=.3 / PLAYBACK_SPEED)
        strike = Line(draft.get_left(), draft.get_right(), color=AMBER, stroke_width=2)
        self.play(Create(strike), run_time=.35 / PLAYBACK_SPEED)
        self.wait(.65 / PLAYBACK_SPEED)
        self.play(FadeOut(draft), FadeOut(strike), run_time=.25 / PLAYBACK_SPEED)
        revised = at_left(label("Take the bus to the museum.", 27, AMBER), 1.02, -2.17)
        self.play(AddTextLetterByLetter(revised), run_time=1.4 / PLAYBACK_SPEED)
        sync()

        self.play(status(left, "Writing answer…", BLUE),
                  status(right, "Final response", GREEN),
                  revised.animate.set_color(INK),
                  right["document"].animate.set_stroke(GREEN, width=2),
                  FadeOut(left["empty"]), run_time=.4 / PLAYBACK_SPEED)
        final_first = at_left(label("Take the bus to the museum.", 27), -6.98, -2.17)
        final_second = at_left(label("Rain is expected.", 27), -6.98, -2.89)
        # A single forward answer pass, only after the reasoning phase ends.
        self.play(Succession(AddTextLetterByLetter(final_first, run_time=1.7 / PLAYBACK_SPEED),
                             AddTextLetterByLetter(final_second, run_time=1.1 / PLAYBACK_SPEED)))
        self.play(status(left, "Final response", BLUE),
                  left["document"].animate.set_stroke(BLUE, width=2), run_time=.4 / PLAYBACK_SPEED)
        self.wait(3 / PLAYBACK_SPEED)


def render():
    OUT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="iterative-response-") as media:
        with tempconfig(dict(media_dir=media, pixel_width=1600, pixel_height=1000,
                             frame_width=16, frame_height=10, frame_rate=30,
                             format="mp4", output_file="comparison",
                             disable_caching=True, verbosity="WARNING")):
            scene = IterativeResponseGeneration()
            scene.render()
            source = Path(scene.renderer.file_writer.movie_file_path)
        # Put the MP4 index first for progressive playback on the static site.
        with av.open(str(source)) as original:
            stream = original.streams.video[0]
            with av.open(str(OUT / "iterative-response-generation.mp4"), "w",
                         options={"movflags": "+faststart"}) as target:
                out_stream = target.add_stream_from_template(stream)
                for packet in original.demux(stream):
                    if packet.dts is not None:
                        packet.stream = out_stream
                        target.mux(packet)
        # A mid-animation poster exposes the difference before playback starts.
        with av.open(str(OUT / "iterative-response-generation.mp4")) as video:
            for frame in video.decode(video=0):
                if frame.time >= 15 / PLAYBACK_SPEED:
                    frame.to_image().save(OUT / "iterative-response-generation.png")
                    break


if __name__ == "__main__":
    render()
