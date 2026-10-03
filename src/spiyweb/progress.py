"""How far a background job has got, read off the lines it prints.

`/index` runs `spiyweb index` in a child process so the screen keeps moving;
the child says what it is doing on stdout (`build_index` logs every stage
before it starts it), and this module turns those lines into a stage count.
Progress bars (tqdm, the model download) are folded into a percentage
instead of reaching the transcript - one `\\r` update per line would flood it.

Pure and stdlib-only: lines in, a status string out. The rules below are
pinned against the real log calls by a test, so a new message in
`build_index` that no rule recognises fails the suite instead of silently
freezing the stage counter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "EDGE_LAYERS",
    "INDEX_STAGES",
    "IndexProgress",
    "JobProgress",
    "bar_percent",
    "classify",
    "clock_text",
]

INDEX_STAGES = ("read", "propositions", "embed", "entities", "edges")
"""What `spiyweb index` does, in order; `propositions` only with the flag."""

EDGE_LAYERS = ("semantic", "entity", "structural", "derivation")
"""The edge layers `build_index` writes one after another, in this order."""

_LAYERS = "|".join(EDGE_LAYERS)
RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^\d+ document\(s\), \d+ unit\(s\) -> "), "read"),
    (re.compile(r"^sync: \d+ chunks reused"), "read"),
    (re.compile(r"^no previous index here"), "read"),
    (re.compile(r"^propositions=False: "), "propositions"),
    (re.compile(r"^extracting propositions: "), "propositions"),
    (re.compile(r"^propositions exist, skipping"), "propositions"),
    (re.compile(r"^proposition layer: "), "propositions"),
    (re.compile(r"^loading the embedding model"), "model"),
    (re.compile(r"^embedding \d+ passages"), "embed"),
    (re.compile(r"^vectors exist, skipping"), "embed"),
    (re.compile(r"^extracting entities from "), "entities"),
    (re.compile(r"^\d+ of \d+ passages fall below"), "entities"),
    (re.compile(r"^entities exist, skipping"), "entities"),
    (re.compile(rf"^({_LAYERS}) edges exist, skipping"), "edges"),
    (re.compile(rf"^({_LAYERS}) layer: \d+ edges"), "edges"),
    (re.compile(r"^nli "), "nli"),
    (re.compile(r"^no nli_model given: "), "nli"),
    (re.compile(r"^done: "), "done"),
)
"""Line pattern -> stage. `model` (loading the embedder) is a pause with a
name rather than a numbered stage - with the proposition layer it comes
before extraction, and a counter that ran backwards would lie. `nli` is
classified for completeness: the CLI never builds the contradiction layer,
but `build_index` can log it."""

_EDGE_STEP = re.compile(rf"^({_LAYERS}) (edges exist|layer:)")
_BAR = re.compile(r"(\d{1,3})%\|")


def classify(line: str) -> str | None:
    """The stage a line announces, or `None` for any other line."""
    text = line.strip()
    for pattern, stage in RULES:
        if pattern.search(text):
            return stage
    return None


def bar_percent(line: str) -> int | None:
    """The percentage of a progress-bar line (`45%|####  | 9/20`), else `None`."""
    found = _BAR.search(line)
    if found is None:
        return None
    return min(100, int(found.group(1)))


def clock_text(seconds: float) -> str:
    """`1:05`, `12:40`, `1:02:03` - elapsed time the way a person reads it."""
    whole = max(0, int(seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


@dataclass
class JobProgress:
    """A job with a name and a clock, plus whatever bar it last printed."""

    label: str
    started: float
    percent: int | None = None

    def feed(self, line: str) -> bool:
        """Note what `line` says; `True` keeps it out of the transcript."""
        found = bar_percent(line)
        if found is None:
            return False
        self.percent = found
        return True

    def parts(self, now: float) -> list[str]:
        """The status pieces, joined by the caller with its own separator."""
        out = [self.label, clock_text(now - self.started)]
        if self.percent is not None:
            out.append(f"{self.percent}%")
        return out


@dataclass
class IndexProgress(JobProgress):
    """`spiyweb index`: which of its stages is running."""

    propositions: bool = False
    stage: str = ""
    edges_done: int = 0

    @property
    def stages(self) -> tuple[str, ...]:
        if self.propositions:
            return INDEX_STAGES
        return tuple(stage for stage in INDEX_STAGES if stage != "propositions")

    def feed(self, line: str) -> bool:
        if super().feed(line):
            return True
        stage = classify(line)
        if stage is None:
            return False
        if stage != self.stage:
            self.percent = None  # a new stage's bar starts from nothing
        self.stage = stage
        if _EDGE_STEP.search(line.strip()):
            self.edges_done += 1
        return False

    def parts(self, now: float) -> list[str]:
        out = [self.label]
        if self.stage in self.stages:
            number = self.stages.index(self.stage) + 1
            step = f"stage {number}/{len(self.stages)} {self.stage}"
            if self.stage == "edges":
                step += f" {self.edges_done}/{len(EDGE_LAYERS)}"
            out.append(step)
        elif self.stage == "model":
            out.append("loading the embedding model")
        elif self.stage == "done":
            out.append("finishing")
        else:
            out.append("starting")
        out.append(clock_text(now - self.started))
        if self.percent is not None:
            out.append(f"{self.percent}%")
        return out
