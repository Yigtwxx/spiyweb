"""`spiyweb` with no arguments: one word to type, then questions.

The four verbs are discoverable from `--help`, and `--help` is where nobody
looks. Someone trying this library for the first time has an index they built
five minutes ago and a question they want to ask it; making them learn a
subcommand and type a path first is a tax on exactly the moment that decides
whether they keep going.

So a bare `spiyweb` asks. What do you want to do, which index, what question -
and it finds the indexes itself, because the path is the part people get
wrong.

Two rules hold this together, and the first one is not negotiable:

1. **It only asks when someone is there to answer.** stdin and stdout must
   both be a terminal. A script, a pipe or a CI job that runs bare `spiyweb`
   gets the usage text and a non-zero exit - never a prompt, because a
   prompt with nobody at it is a hung build.
2. **Every answer is a subcommand.** The wizard builds an argv and hands it
   to the same parser everything else goes through; it runs nothing of its
   own. It also PRINTS that argv, so the second time you know what to type
   and the wizard has made itself unnecessary.

Arrow keys where the terminal allows it, numbers everywhere else. Raw-mode
input is `msvcrt` on Windows and `termios` on POSIX - both stdlib, so the
zero-dependency promise is untouched - but neither exists in a pipe, an
editor's output pane or a CI log. So the numbered prompt is not a lesser
fallback that rots: it is the same function, tested, and it is what runs
whenever raw input is unavailable.

Raw mode is entered only around the single `read` of one keypress and
restored immediately. Printing inside raw mode is the classic way to get a
staircase of half-indented lines on POSIX, and holding the terminal in raw
mode across a redraw is how a crash leaves somebody's shell unusable.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from spiyweb import __version__

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ["interactive", "is_interactive", "run_wizard"]

QUIT = ("q", "quit", "exit")

SEARCH_ROOTS = (".", "data", "indexes")
"""Where to look for indexes, nearest first. Not a recursive walk: a scan of
somebody's whole home directory is a surprise, and an index one level down is
where they actually are."""

MAX_DISCOVERED = 12
"""Enough to choose from, few enough to read. Past this the wizard asks for a
path instead of printing a wall of options."""


@dataclass(frozen=True)
class Found:
    """One thing the wizard discovered and can act on."""

    path: Path
    kind: str
    detail: str


def is_interactive() -> bool:
    """Whether there is a person to answer. Both ends must be a terminal."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _color() -> bool:
    from spiyweb.terminal import supports_color

    return supports_color()


UP, DOWN, ENTER, QUIT_KEY = "up", "down", "enter", "quit"

HIDE_CURSOR, SHOW_CURSOR = "\x1b[?25l", "\x1b[?25h"
"""The blinking block sitting on a menu line looks like a text field nobody
can type in. Restored in a `finally`, always: a program that exits with the
cursor hidden leaves the next shell looking broken."""

FILLED, HOLLOW = "●", "○"
"""A filled dot for the option under the cursor, hollow for the rest. The
whole point of an arrow menu is that the answer to "where am I" costs no
reading, and a marker that changes SHAPE survives a terminal with no colour
at all - which is exactly the terminal a colourless fallback runs in."""


def supports_raw_input() -> bool:
    """Whether single keypresses can be read without waiting for Enter."""
    if not is_interactive():
        return False
    if sys.platform == "win32":
        try:
            import msvcrt  # noqa: F401
        except ImportError:  # pragma: no cover - not a Windows build
            return False
        return True
    try:
        import termios  # noqa: F401
        import tty  # noqa: F401

        return sys.stdin.fileno() >= 0
    except (ImportError, ValueError, OSError):  # pragma: no cover
        return False


def read_key() -> str:
    """One keypress: `up`, `down`, `enter`, `quit`, or the character itself.

    Raw mode is entered and left around this single read. Anything longer -
    a redraw, a print - happens with the terminal back in its normal state,
    because raw mode plus a newline is how POSIX output turns into a
    staircase, and a crash inside raw mode leaves a shell that echoes
    nothing.
    """
    if sys.platform == "win32":
        return _read_key_windows()
    return _read_key_posix()


def _read_key_windows() -> str:
    import msvcrt

    char = msvcrt.getwch()
    if char in ("\x00", "à"):  # a two-part key: arrows, function keys
        return {"H": UP, "P": DOWN}.get(msvcrt.getwch(), "")
    return _classify(char)


def _read_key_posix() -> str:  # pragma: no cover - exercised off Windows
    import select
    import termios
    import tty

    descriptor = sys.stdin.fileno()
    saved = termios.tcgetattr(descriptor)
    try:
        tty.setraw(descriptor)
        char = sys.stdin.read(1)
        if char == "\x1b":
            return _decode_escape(
                pending=lambda: bool(select.select([sys.stdin], [], [], 0.05)[0]),
                read=sys.stdin.read,
            )
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, saved)
    return _classify(char)


def _decode_escape(pending: Callable[[], bool], read: Callable[[int], str]) -> str:
    """What follows an ESC byte on a POSIX terminal, without ever blocking.

    An arrow arrives as `ESC [ A` (or `ESC O A` in application mode) in one
    burst; a bare ESC is somebody leaving, and NOTHING follows it. A blocking
    read after the ESC would therefore wait for two keys that never come and
    the menu would freeze - so the caller asks `pending` first, with a short
    timeout, and only reads what is actually there.
    """
    if not pending():
        return QUIT_KEY
    if read(1) not in ("[", "O"):
        return ""
    return {"A": UP, "B": DOWN}.get(read(1), "")


def _classify(char: str) -> str:
    """Turn one character into an intent, or hand it back untouched."""
    if char in ("\r", "\n"):
        return ENTER
    if char == "\x03":  # Ctrl-C means stop, here as everywhere
        raise KeyboardInterrupt
    if char in ("\x1b", "\x04"):  # ESC, Ctrl-D
        return QUIT_KEY
    if char.lower() in QUIT:
        return QUIT_KEY
    # vim and wasd, because muscle memory is cheaper than a legend.
    if char in ("k", "w"):
        return UP
    if char in ("j", "s"):
        return DOWN
    return char


def _ask(prompt: str) -> str:
    """One line from the person, or a quit on EOF / Ctrl-C.

    Ctrl-C and Ctrl-D both mean "stop", and both should leave the terminal
    tidy rather than dumping a traceback over the menu.
    """
    from spiyweb.terminal import paint

    try:
        return input(paint(prompt, "accent", "bold", enabled=_color())).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "q"


def choose(question: str, options: Sequence[tuple[str, ...]]) -> str | None:
    """Ask a question; return the chosen key, or `None` to quit.

    Each option is `(key, label)` or `(key, label, note)`. The note is
    printed in grey parentheses beside the label, because a menu that only
    names its options makes the reader guess what they do - and guessing is
    what the wizard exists to remove.

    Arrow keys where the terminal can read one keypress, numbers where it
    cannot. Both paths return the same thing, so the caller never learns
    which one ran.
    """
    if supports_raw_input():
        return _choose_by_arrows(question, options)
    return _choose_by_number(question, options)


def _render(
    options: Sequence[tuple[str, ...]], cursor: int, *, color: bool
) -> list[str]:
    """The option lines, with a filled dot on the one under the cursor.

    The marker changes SHAPE and not only colour: this menu has to work on a
    terminal that refused colour, and on that terminal a purely chromatic
    cursor is no cursor at all.
    """
    from spiyweb.terminal import paint

    lines: list[str] = []
    for index, option in enumerate(options):
        chosen = index == cursor
        dot = paint(
            FILLED if chosen else HOLLOW,
            "good" if chosen else "muted",
            enabled=color,
        )
        label = option[1]
        note = option[2] if len(option) > 2 else ""
        body = paint(label, "accent", "bold", enabled=color) if chosen else label
        described = paint(f"  ({note})", "muted", enabled=color) if note else ""
        lines.append(f"  {dot} {body}{described}")
    return lines


def _choose_by_arrows(question: str, options: Sequence[tuple[str, ...]]) -> str | None:
    """Move with the arrows, choose with Enter, leave with q.

    Redrawn in place: the cursor is moved back up over the option block and
    each line is cleared before it is written again, so the menu stays one
    block instead of scrolling a new copy of itself on every keypress.
    """
    from spiyweb.terminal import paint

    color = _color()
    cursor = 0
    print()
    print(paint(question, "heading", "bold", enabled=color))
    print(paint("  arrows to move, enter to choose, q to quit", "muted", enabled=color))
    for line in _render(options, cursor, color=color):
        print(line)

    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.flush()
    try:
        while True:
            try:
                key = read_key()
            except KeyboardInterrupt:
                return None
            if key == ENTER:
                return options[cursor][0]
            if key == QUIT_KEY:
                return None
            if key == UP:
                cursor = (cursor - 1) % len(options)
            elif key == DOWN:
                cursor = (cursor + 1) % len(options)
            elif key.isdigit() and 1 <= int(key) <= len(options):
                # Numbers still work: the muscle memory of every other menu.
                return options[int(key) - 1][0]
            else:
                continue
            sys.stdout.write(f"\x1b[{len(options)}A")
            for line in _render(options, cursor, color=color):
                sys.stdout.write(f"\x1b[2K{line}\n")
            sys.stdout.flush()
    finally:
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()


def _choose_by_number(question: str, options: Sequence[tuple[str, ...]]) -> str | None:
    """The fallback, and the tested path: type a number and press Enter.

    Not a lesser menu that rots - it is what runs in a pipe, a CI log and an
    editor's output pane, and every test of `choose` goes through it.

    Empty input takes the first option: the common case should cost one
    keystroke, and the first option is always the one most people want.
    """
    from spiyweb.terminal import paint

    color = _color()
    print()
    print(paint(question, "heading", "bold", enabled=color))
    for index, option in enumerate(options, start=1):
        label = option[1]
        note = option[2] if len(option) > 2 else ""
        # Right-aligned: a two-digit option must not shove its label
        # one column further than every single-digit one above it.
        number = paint(f"  {index:>2})", "accent", enabled=color)
        described = paint(f"  ({note})", "muted", enabled=color) if note else ""
        default = paint("  [enter]", "muted", enabled=color) if index == 1 else ""
        print(f"{number} {label}{described}{default}")
    print(paint("   q)  quit", "muted", enabled=color))

    while True:
        answer = _ask("> ")
        if answer.lower() in QUIT:
            return None
        if not answer:
            return options[0][0]
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1][0]
        print(paint(f"  pick 1-{len(options)}, or q", "warn", enabled=color))


def ask_text(question: str, *, default: str = "") -> str | None:
    """Ask for free text; `None` means quit."""
    from spiyweb.terminal import paint

    color = _color()
    print()
    print(paint(question, "heading", "bold", enabled=color))
    if default:
        print(paint(f"  (enter for: {default})", "muted", enabled=color))
    answer = _ask("> ")
    if answer.lower() in QUIT:
        return None
    return answer or default


def discover(cwd: Path | None = None) -> list[Found]:
    """Find indexes nearby, so nobody has to type a path.

    An index is a directory holding `nodes.json` - the same test the CLI's
    own verbs use, so what the wizard offers is exactly what they accept.
    `cwd` is where "nearby" starts; the process's own directory by default.
    """
    indexes: list[Found] = []
    seen: set[Path] = set()
    for root in SEARCH_ROOTS:
        base = Path(root) if cwd is None else cwd / root
        if not base.is_dir():
            continue
        # Case-folded so the menu order is the same on every platform.
        for entry in sorted(base.iterdir(), key=lambda p: p.name.casefold()):
            if not entry.is_dir() or entry.resolve() in seen:
                continue
            seen.add(entry.resolve())
            if (entry / "nodes.json").is_file():
                indexes.append(Found(entry, "index", _atom_count(entry)))
    return indexes[:MAX_DISCOVERED]


def _atom_count(index: Path) -> str:
    """How many atoms an index holds, or nothing if it will not say."""
    import json

    try:
        nodes = json.loads((index / "nodes.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return f"{len(nodes):,} atoms" if isinstance(nodes, list) else ""


def _pick_index(question: str) -> str | None:
    """Choose a discovered index, or type a path. `None` means quit."""
    from spiyweb.terminal import paint

    indexes = discover()
    if not indexes:
        print()
        print(
            paint(
                "No index found nearby. Build one with option 4, or give a path.",
                "warn",
                enabled=_color(),
            )
        )
        return ask_text("Path to an index directory:")

    options: list[tuple[str, ...]] = [
        (str(found.path), str(found.path), found.detail) for found in indexes
    ]
    options.append(("", "Somewhere else", "type a path yourself"))
    chosen = choose(question, options)
    if chosen is None:
        return None
    return chosen or ask_text("Path to an index directory:")


def run_wizard() -> int:
    """Ask what to do, then run it through the ordinary parser."""
    from spiyweb.terminal import paint, rule, terminal_width

    color = _color()
    print(rule(f"spiyweb {__version__}", terminal_width(), enabled=color))
    print(
        paint(
            "Graph retrieval: the query is an energy seed that spreads and decays.",
            "muted",
            enabled=color,
        )
    )

    action = choose(
        "What would you like to do?",
        [
            ("query", "Ask a question", "the web spreads and you see what lit up"),
            ("lint", "Inspect a corpus", "islands, hubs, duplicates - no query needed"),
            ("index", "Build an index", "a folder of .txt/.md becomes a graph"),
            ("version", "What is installed", "which extras you have, which you lack"),
        ],
    )
    if action is None:
        return 0

    argv = _build(action)
    if argv is None:
        return 0

    print()
    print(
        paint("running: ", "muted", enabled=color)
        + paint(
            "spiyweb " + " ".join(_quote(part) for part in argv),
            "accent",
            enabled=color,
        )
    )
    print()
    from spiyweb.cli import main

    return main(argv)


def _build(action: str) -> list[str] | None:
    """Turn one answer into the argv for a real subcommand."""
    if action == "version":
        return ["version"]

    if action == "index":
        source = ask_text("Folder of .txt/.md files to index:", default="docs")
        if source is None:
            return None
        target = ask_text("Where should the index go?", default="my-index")
        if target is None:
            return None
        return ["index", source, target]

    if action == "query":
        index = _pick_index("Which index?")
        if index is None:
            return None
        question = ask_text("What do you want to ask it?")
        if not question:
            return None
        profile = choose(
            "How should the web spread?",
            [
                ("", "Explore", "the default - damps slowly, travels furthest"),
                ("precise", "Precise", "damps fast, stays near the question"),
                ("compare", "Compare", "wider seed, for two-sided questions"),
            ],
        )
        if profile is None:
            return None
        argv = ["query", index, question]
        if profile:
            argv += ["--profile", profile]
        return argv

    if action == "lint":
        index = _pick_index("Which corpus?")
        return None if index is None else ["lint", index]

    return None


def _quote(part: str) -> str:
    """Quote an argument the way a person would have to type it."""
    return f'"{part}"' if " " in part else part


def interactive() -> int:
    """Entry point for a bare `spiyweb`, guarded by the terminal check."""
    if not is_interactive():
        from spiyweb.cli import build_parser

        build_parser().print_help()
        print(
            "\nspiyweb: pick a command. The guided menu only runs in a "
            "terminal - there is nobody to answer a prompt here.",
            file=sys.stderr,
        )
        return 2
    return run_wizard()
