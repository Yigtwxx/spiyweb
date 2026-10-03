"""The SPIYWEB wordmark in the monitor's welcome box.

The landing page's mark (`site/src/data/demo.ts`), cell for cell: block
letters for the word and double box lines for their shadow, in the figlet
style *ANSI Shadow* - six rows, the same size as the wordmarks of the
owner's other terminal tools, so the family reads as one. The letters shade
from frost-white into glacier blue left to right; the shadow keeps one dark
blue. With colour off the shape alone still reads.

Colours are 256-colour codes, the nearest tinted cell of the xterm cube to
each of the page's hex values: truecolor is not on every terminal this runs
in (macOS Terminal, older Windows consoles), 256 colours are, and the rest of
the monitor already speaks it.

Pure functions over strings, no dependency: the monitor decides whether
there is room for it, this module only draws it.
"""

from __future__ import annotations

from spiyweb.terminal import RESET

__all__ = [
    "BANDS",
    "BLOCK",
    "SHADOW",
    "WORDMARK",
    "wordmark_lines",
    "wordmark_width",
]

BLOCK = "█"
"""The cell the letters are made of; every other glyph is shadow."""

WORDMARK: tuple[str, ...] = (
    "███████╗██████╗ ██╗██╗   ██╗██╗    ██╗███████╗██████╗ ",
    "██╔════╝██╔══██╗██║╚██╗ ██╔╝██║    ██║██╔════╝██╔══██╗",
    "███████╗██████╔╝██║ ╚████╔╝ ██║ █╗ ██║█████╗  ██████╔╝",
    "╚════██║██╔═══╝ ██║  ╚██╔╝  ██║███╗██║██╔══╝  ██╔══██╗",
    "███████║██║     ██║   ██║   ╚███╔███╔╝███████╗██████╔╝",
    "╚══════╝╚═╝     ╚═╝   ╚═╝    ╚══╝╚══╝ ╚══════╝╚═════╝ ",
)
"""Fifty-four columns by six rows - a rectangle, so nothing has to guess."""

BANDS: tuple[str, ...] = (
    "38;5;195",  # #e3f2f9 frost-white
    "38;5;153",  # #c4e6f4
    "38;5;117",  # #9fd6ec
    "38;5;110",  # #74bde3
    "38;5;74",  # #4f9fd6 glacier blue
)
"""Five gradient bands across the word, left to right."""

SHADOW = "38;5;24"
"""#2a527d: the shadow glyphs."""


def wordmark_width() -> int:
    """Columns the mark takes."""
    return max(len(line) for line in WORDMARK)


def wordmark_lines(*, color: bool) -> list[str]:
    """The mark, each line exactly `wordmark_width()` printed columns."""
    width = wordmark_width()
    return [_paint_row(line.ljust(width), color=color) for line in WORDMARK]


def _band(column: int) -> str:
    """The gradient band a column of the mark falls in."""
    return BANDS[min(len(BANDS) - 1, column * len(BANDS) // wordmark_width())]


def _paint_row(line: str, *, color: bool) -> str:
    """One row of the mark, one escape sequence per run of equal colour."""
    if not color:
        return line
    out: list[str] = []
    run, code = "", ""
    for column, char in enumerate(line):
        here = "" if char == " " else _band(column) if char == BLOCK else SHADOW
        if here != code and run:
            out.append(f"\x1b[{code}m{run}{RESET}" if code else run)
            run = ""
        code = here
        run += char
    if run:
        out.append(f"\x1b[{code}m{run}{RESET}" if code else run)
    return "".join(out)
