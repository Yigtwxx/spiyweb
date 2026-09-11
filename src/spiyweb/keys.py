"""Raw keypresses with a timeout, for a screen that also watches a file.

The wizard reads one key and waits for it - fine for a menu. The monitor
cannot wait: every few milliseconds it must look at the trace file, so it
needs "a key, or nothing, within `timeout`". That is the only thing here.

Keys come back RAW: a named key (`enter`, `backspace`, `escape`, arrows) or
the literal character. No `q`-means-quit, no vim letters - the person is
typing a command line, and `/query` contains a `q`.

Platform notes, both stdlib:

- Windows: `msvcrt.kbhit()` polls, `getwch()` reads without echo. Arrows
  arrive as two calls, the first being `\\x00` or `\\xe0`.
- POSIX: `select()` on a cooked tty only wakes on a full LINE, so the
  terminal is put in cbreak mode around the `select` + `read` pair - at most
  one poll slice, nothing printed inside - and restored in a `finally`.
  Ctrl-C still arrives as a character here (`\\x03`) and is raised.
"""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "BACKSPACE",
    "DELETE",
    "DISABLE_FOCUS",
    "DOWN",
    "ENABLE_FOCUS",
    "END_KEY",
    "ENTER",
    "ESCAPE",
    "FOCUS_IN",
    "FOCUS_OUT",
    "HOME_KEY",
    "LEFT",
    "NAMED",
    "RIGHT",
    "TAB",
    "UP",
    "decode_escape",
    "poll_raw",
]

UP, DOWN, LEFT, RIGHT = "up", "down", "left", "right"
ENTER, BACKSPACE, ESCAPE, TAB = "enter", "backspace", "escape", "tab"
FOCUS_IN, FOCUS_OUT = "focus_in", "focus_out"
HOME_KEY, END_KEY, DELETE = "home", "end", "delete"
"""What a terminal with focus reporting on (`ENABLE_FOCUS`) sends when the
window gains or loses the keyboard - the caret follows."""
NAMED = frozenset(
    {
        UP,
        DOWN,
        LEFT,
        RIGHT,
        ENTER,
        BACKSPACE,
        ESCAPE,
        TAB,
        FOCUS_IN,
        FOCUS_OUT,
        HOME_KEY,
        END_KEY,
        DELETE,
    }
)
ENABLE_FOCUS, DISABLE_FOCUS = "\x1b[?1004h", "\x1b[?1004l"

_WINDOWS_ARROWS = {
    "H": UP,
    "P": DOWN,
    "K": LEFT,
    "M": RIGHT,
    "G": HOME_KEY,
    "O": END_KEY,
    "S": DELETE,
}
_POSIX_ARROWS = {
    "A": UP,
    "B": DOWN,
    "C": RIGHT,
    "D": LEFT,
    "I": FOCUS_IN,
    "O": FOCUS_OUT,
    "H": HOME_KEY,
    "F": END_KEY,
}
_ESCAPE_WAIT_S = 0.05
_WINDOWS_SLICE_S = 0.01


def poll_raw(timeout_s: float) -> str | None:
    """A named key, a literal character, or `None` once `timeout_s` passed."""
    if sys.platform == "win32":
        import msvcrt

        return _poll_windows(
            timeout_s, kbhit=msvcrt.kbhit, getwch=msvcrt.getwch, sleep=time.sleep
        )
    return _poll_posix(timeout_s)  # pragma: no cover - exercised off Windows


def name_windows_key(
    first: str,
    second: Callable[[], str],
    pending: Callable[[], bool] = lambda: False,
) -> str:
    """Turn what `getwch` returned into a raw key, reading the second half of
    a two-part key (arrows) only when the first half announces one. An ESC
    with more already waiting is a VT sequence the console passed through -
    focus reports arrive that way - and is decoded like on POSIX."""
    if first in ("\x00", "\xe0"):
        return _WINDOWS_ARROWS.get(second(), "")
    if first == "\x1b" and pending():
        return decode_escape(pending=pending, read=lambda _n: second())
    return _name_char(first)


def _poll_windows(
    timeout_s: float,
    *,
    kbhit: Callable[[], bool],
    getwch: Callable[[], str],
    sleep: Callable[[float], None],
) -> str | None:
    deadline = time.monotonic() + timeout_s
    while True:
        if kbhit():
            return name_windows_key(getwch(), getwch, pending=kbhit)
        if time.monotonic() >= deadline:
            return None
        sleep(_WINDOWS_SLICE_S)


def _poll_posix(timeout_s: float) -> str | None:  # pragma: no cover
    import select
    import termios
    import tty

    descriptor = sys.stdin.fileno()
    saved = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        if not select.select([sys.stdin], [], [], timeout_s)[0]:
            return None
        char = sys.stdin.read(1)
        if char == "\x1b":
            return decode_escape(
                pending=lambda: bool(
                    select.select([sys.stdin], [], [], _ESCAPE_WAIT_S)[0]
                ),
                read=sys.stdin.read,
            )
        return _name_char(char)
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, saved)


def decode_escape(pending: Callable[[], bool], read: Callable[[int], str]) -> str:
    """What follows an ESC byte on a POSIX terminal, without ever blocking.

    An arrow arrives as `ESC [ A` in one burst; a bare ESC is the Escape key
    and NOTHING follows it. So the caller asks `pending` first and only reads
    what is actually there. Anything else after the ESC is dropped: a
    function key is not a command-line character.
    """
    if not pending():
        return ESCAPE
    if read(1) not in ("[", "O"):
        return ""
    third = read(1)
    if third == "3" and pending() and read(1) == "~":  # ESC [ 3 ~ is Delete
        return DELETE
    return _POSIX_ARROWS.get(third, "")


def _name_char(char: str) -> str:
    if char in ("\r", "\n"):
        return ENTER
    if char in ("\x7f", "\x08"):
        return BACKSPACE
    if char == "\x1b":
        return ESCAPE
    if char == "\t":
        return TAB
    if char == "\x03":
        raise KeyboardInterrupt
    return char
