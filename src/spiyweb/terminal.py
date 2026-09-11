"""Colour and shape for the terminal, without a dependency to draw them with.

`rich` would be the obvious answer and it is the wrong one here: this package
promises `dependencies = []`, the wheel job measures that promise on the
artifact, and a pretty CLI is not worth trading it for. So this is ANSI SGR
by hand - a few hundred bytes of stdlib - and it buys the same three things
that matter:

- **Energy as length.** A bar whose width is a node's share of the strongest
  activation shows the decay the whole project is about. A column of numbers
  does not; you have to divide them in your head.
- **Hop as colour.** First contact, one hop out, two hops out: the gradient
  is the web spreading, and it is the one thing a reader should see first.
  The whole palette is two families - **blue for the web working, red for
  something wrong** - so a report can be judged before it is read.
- **Structure as rules.** Sections a person can skim past.

Three rules it obeys without exception, because a decorated CLI that breaks a
pipe is worse than a plain one:

1. **Not a terminal, no colour.** Piping to `jq` or a log file gets clean
   text. This is not a nicety - `spiyweb lint --json | jq` was broken once
   already by a progress line, and escape codes would break it again.
2. **`NO_COLOR` is honoured**, per <https://no-color.org>. Any value.
3. **Windows gets its VT mode enabled**, because Python does not do it and
   this project's author is on Windows. If enabling fails, colour is off -
   never raw escape codes on screen.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TextIO

__all__ = [
    "CLEAR_SCREEN",
    "ERASE_LINE",
    "HIDE_CURSOR",
    "HOME",
    "SHOW_CURSOR",
    "bar",
    "clip",
    "cursor_to",
    "hop_color",
    "pad",
    "paint",
    "printed_width",
    "rule",
    "supports_color",
    "supports_screen",
    "supports_unicode",
    "terminal_size",
    "terminal_width",
]

RESET = "\x1b[0m"

HIDE_CURSOR, SHOW_CURSOR = "\x1b[?25l", "\x1b[?25h"
"""Hidden while a screen is being redrawn, shown again in a `finally`,
always: a program that exits with the cursor hidden leaves the next shell
looking broken."""

CLEAR_SCREEN = "\x1b[2J\x1b[3J\x1b[H"
"""Screen AND scrollback, then home - what typing `claude` does to a shell.
The monitor owns the whole window from then on."""

HOME, ERASE_LINE = "\x1b[H", "\x1b[2K"

# Two families and a grey, and the split carries meaning rather than taste:
# **blue is the web working, red is something wrong.** A reader who learns
# nothing else can still tell a report that is fine from one that is not, at
# a glance and without reading a word.
#
# 256-colour codes, picked to stay legible on a dark AND a light background -
# no pure black, no pure white, nothing that vanishes on either. Within each
# family the steps are far enough apart to survive a cheap terminal palette;
# a test asserts every entry is distinct.
_CODES = {
    "bold": "1",
    "dim": "2",
    "italic": "3",
    # --- blue: structure, energy, things that went right ---
    "heading": "38;5;39",  # azure, always bold - section titles
    "accent": "38;5;75",  # periwinkle - prompts, numbers, the command to copy
    "good": "38;5;51",  # bright cyan - installed, balanced, stopped cleanly
    # --- red: findings, absences, failures ---
    "warn": "38;5;210",  # coral - dense but not structural
    "bad": "38;5;196",  # red - a layer doing nothing, an island, a crash
    # --- neither: text that must not compete for attention ---
    "muted": "38;5;245",  # grey - ids, counts, parenthetical notes
}

_HOP_COLORS = (
    "38;5;51",  # hop 0 - first contact, brightest cyan
    "38;5;45",  # hop 1
    "38;5;39",  # hop 2
    "38;5;32",  # hop 3
    "38;5;25",  # hop 4 and beyond - deep blue, nearly spent
)
"""One ramp down the blue family: bright cyan at first contact, deep blue
where the energy has nearly run out. Depth reads at a glance, and it never
borrows red - red means a finding, and a distant atom is not a finding.

Five steps and then it stops: past hop 4 the energy is nearly gone and finer
shading would be a distinction without a difference."""

_BLOCKS = " ▏▎▍▌▋▊▉█"
"""Eighth-width blocks, so a bar has sub-character resolution. Plain ASCII
fallback is not needed - these are in every font a terminal ships with, and
the alternative (`#`) makes a chart of hashes."""

MIN_WIDTH = 40
"""Below this the terminal is too narrow for bars; callers fall back to
numbers. Not a guess - a bar of four characters carries no information."""

DEFAULT_WIDTH = 80
"""Assumed width when the terminal will not say, e.g. in a pipe or CI."""


def supports_color(stream: TextIO | None = None) -> bool:
    """Whether to emit escape codes at all.

    Checked once per call rather than cached: a caller may hand us a
    different stream, and the cost is three environment lookups.
    """
    target = stream if stream is not None else sys.stdout
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    try:
        if not target.isatty():
            return False
    except (AttributeError, ValueError):
        return False
    if sys.platform == "win32":
        return _enable_windows_vt()
    return True


def _enable_windows_vt() -> bool:
    """Turn on virtual-terminal processing, or report that colour is off.

    Python does not do this for us, and without it Windows prints the escape
    codes literally - which is strictly worse than plain text. Modern Windows
    Terminal and PowerShell 7 already have it on; this is for the console
    hosts that do not.
    """
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        enable_vt = 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        if mode.value & enable_vt:
            return True
        return bool(kernel32.SetConsoleMode(handle, mode.value | enable_vt))
    except Exception:  # pragma: no cover - depends on the console host
        return False


def paint(text: str, *styles: str, enabled: bool = True) -> str:
    """Wrap `text` in the named styles, or return it untouched.

    Unknown style names are ignored rather than raised on: a typo should
    cost a colour, not a crash in the middle of someone's report.
    """
    if not enabled or not styles or not text:
        return text
    codes = [_CODES[name] for name in styles if name in _CODES]
    if not codes:
        return text
    return f"\x1b[{';'.join(codes)}m{text}{RESET}"


def hop_color(hop: int) -> str:
    """The style code for a hop distance, saturating at the last step."""
    index = min(max(hop, 0), len(_HOP_COLORS) - 1)
    return _HOP_COLORS[index]


def paint_hop(text: str, hop: int, *, enabled: bool = True) -> str:
    """Colour `text` by how far the energy travelled to reach it."""
    if not enabled or not text:
        return text
    return f"\x1b[{hop_color(hop)}m{text}{RESET}"


def terminal_width(fallback: int = DEFAULT_WIDTH) -> int:
    """Usable width, clamped so a very wide terminal does not sprawl."""
    try:
        columns = shutil.get_terminal_size(fallback=(fallback, 24)).columns
    except (OSError, ValueError):  # pragma: no cover - exotic environments
        columns = fallback
    return max(MIN_WIDTH, min(columns, 120))


def bar(value: float, maximum: float, width: int, blocks: str = _BLOCKS) -> str:
    """A proportional bar with eighth-character resolution.

    Any positive value gets at least one visible mark. A node that activated
    at all is not the same as a node that did not, and rounding it to an
    empty bar would say it was. `blocks` is the ramp from empty to full; the
    default is the eighth blocks, an ASCII console passes a shorter one.
    """
    if width <= 0 or maximum <= 0.0 or value <= 0.0:
        return ""
    steps = len(blocks) - 1
    share = min(value / maximum, 1.0)
    units = round(share * width * steps)
    full, remainder = divmod(max(units, 1), steps)
    return blocks[-1] * full + (blocks[remainder] if remainder else "")


def rule(title: str, width: int, *, enabled: bool = True) -> str:
    """A section heading with a line to its right, sized to the terminal."""
    label = title.upper()
    # `- 1` for the single space between the label and the line, and nothing
    # else: an earlier `- 3` left every heading two columns short of the
    # width it was handed, which is invisible on one line and obvious once
    # two rules sit above each other.
    dashes = max(width - len(label) - 1, 0)
    painted = paint(label, "heading", "bold", enabled=enabled)
    return f"{painted} {paint('─' * dashes, 'muted', enabled=enabled)}"


def columns(rows: Sequence[Sequence[str]], gap: int = 2) -> list[str]:
    """Align rows into columns on their PRINTED width.

    Escape codes have no width on screen but plenty in `len()`, so padding
    computed from the raw string would misalign every coloured cell. The
    widths here are measured on the stripped text.
    """
    if not rows:
        return []
    widths = [0] * max(len(row) for row in rows)
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], _printed_width(cell))
    lines: list[str] = []
    for row in rows:
        parts: list[str] = []
        for index, cell in enumerate(row):
            pad = widths[index] - _printed_width(cell)
            parts.append(cell + " " * pad if index < len(row) - 1 else cell)
        lines.append((" " * gap).join(parts).rstrip())
    return lines


def cursor_to(row: int) -> str:
    """Move to the first column of `row` (1-based)."""
    return f"\x1b[{row};1H"


def terminal_size(fallback: tuple[int, int] = (DEFAULT_WIDTH, 24)) -> tuple[int, int]:
    """`(columns, rows)` of the terminal, uncapped - a full-screen layout
    wants every column, unlike a report that reads better narrow."""
    size = shutil.get_terminal_size(fallback=fallback)
    return max(MIN_WIDTH, size.columns), max(8, size.lines)


def supports_screen(stream: TextIO | None = None) -> bool:
    """Whether a full-screen redraw is worth attempting: a terminal that is
    not dumb, with VT sequences on. Independent of colour - `NO_COLOR` turns
    paint off, not cursor movement."""
    target = stream if stream is not None else sys.stdout
    try:
        if not target.isatty():
            return False
    except (AttributeError, ValueError):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    if sys.platform == "win32":
        return _enable_windows_vt()
    return True


def supports_unicode(stream: TextIO | None = None) -> bool:
    """Whether box drawing and braille are safe to print. `SPIYWEB_ASCII=1`
    forces the plain glyph set on any terminal."""
    if os.environ.get("SPIYWEB_ASCII"):
        return False
    target = stream if stream is not None else sys.stdout
    encoding = (getattr(target, "encoding", None) or "").lower().replace("-", "")
    return encoding in ("utf8", "utf16", "utf32")


def clip(text: str, width: int) -> str:
    """Cut `text` to `width` printed columns, keeping escape sequences intact
    and closing any colour that was still open at the cut."""
    if width <= 0:
        return ""
    if "\x1b" not in text:
        return text[:width]
    out: list[str] = []
    seen = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\x1b":
            end = text.find("m", index)
            end = len(text) - 1 if end < 0 else end
            out.append(text[index : end + 1])
            index = end + 1
            continue
        if seen < width:
            out.append(char)
            seen += 1
        index += 1
    clipped = "".join(out)
    return clipped if clipped.endswith(RESET) else clipped + RESET


def pad(text: str, width: int) -> str:
    """`text` clipped and space-padded to exactly `width` printed columns."""
    text = clip(text, width)
    return text + " " * (width - printed_width(text))


def printed_width(text: str) -> int:
    """Length with escape sequences removed."""
    return _printed_width(text)


def _printed_width(text: str) -> int:
    """Length with escape sequences removed."""
    if "\x1b" not in text:
        return len(text)
    width = 0
    inside = False
    for char in text:
        if inside:
            inside = char != "m"
            continue
        if char == "\x1b":
            inside = True
            continue
        width += 1
    return width
