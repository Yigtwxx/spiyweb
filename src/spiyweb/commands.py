"""The `/` commands of the monitor - every question the old menu asked.

`/query`, `/lint`, `/index` and `/install` are the four menu items as
commands; `/find` looks for the application that imports spiyweb and the
index it opens; `/config` is an arrow-key list of the monitor's own knobs;
`/replay` plays a recorded query again. Text without a slash is a question
for the current index, once one is known.

A command that maps to a CLI verb builds an argv and runs the same parser
everything else goes through (`Monitor.run_captured`), with the output
landing in the transcript. The one exception is `/query`, which the monitor
runs itself so it can PLAY the record instead of printing a table.

Nothing here touches the terminal: commands talk to the `Monitor` through
`log`, `reply`, `ask`, `pick` and `play`, so a fake monitor tests them.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from spiyweb.watch import Monitor

__all__ = [
    "COMMANDS",
    "FindReport",
    "Usage",
    "dispatch",
    "find_usages",
    "parse",
    "pip_command",
    "problem_hint",
]

INSTALLABLE = ("store", "embed", "entity", "view", "nli", "index")
PRUNED_DIRS = frozenset(
    {"venv", ".venv", "node_modules", "__pycache__", "site-packages", "build", "dist"}
)
IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+spiyweb\b")
OPEN_RE = re.compile(r"(?:open_index|SpiywebIndex\.open)\(\s*[rRuU]?(['\"])(.+?)\1")
PIP_HINT_RE = re.compile(r'pip install "spiyweb\[([a-z,]+)\]"')


@dataclass(frozen=True)
class Invocation:
    name: str
    args: tuple[str, ...]
    raw: str


@dataclass(frozen=True)
class Command:
    name: str
    usage: str
    summary: str
    handler: Callable[[Monitor, Invocation], None]


@dataclass(frozen=True)
class Usage:
    path: Path
    line: int
    text: str


@dataclass(frozen=True)
class FindReport:
    usages: tuple[Usage, ...]
    index_hints: tuple[str, ...]
    scanned: int
    truncated: bool


def parse(line: str) -> Invocation | None:
    """`/query idx what now` -> `("query", ("idx", "what", "now"))`; text
    without a slash is not a command (it is a question)."""
    stripped = line.strip()
    if not stripped.startswith("/"):
        return None
    parts = stripped[1:].split()
    if not parts:
        return Invocation(name="help", args=(), raw=stripped)
    return Invocation(name=parts[0].lower(), args=tuple(parts[1:]), raw=stripped)


def dispatch(monitor: Monitor, line: str) -> None:
    monitor.echo(line)
    invocation = parse(line)
    if invocation is None:
        _question(monitor, line.strip())
        return
    for command in COMMANDS:
        if command.name == invocation.name:
            command.handler(monitor, invocation)
            return
    monitor.reply(
        monitor.paint(f"no such command /{invocation.name}", "warn")
        + monitor.paint(" - /help lists them", "muted"),
        tone="warn",
    )


def problem_hint(message: str) -> str:
    """A CLI error that says `pip install "spiyweb[x]"` says `/install x` here."""
    return PIP_HINT_RE.sub(lambda m: f"/install {m.group(1)}", message)


# --- /help ----------------------------------------------------------------


def _help(monitor: Monitor, _: Invocation) -> None:
    lines = [monitor.paint("commands", "bold")]
    for command in COMMANDS:
        lines.append(
            monitor.paint(command.usage.ljust(30), "accent")
            + monitor.paint(command.summary, "muted")
        )
    lines.append(
        monitor.paint("anything else".ljust(30), "accent")
        + monitor.paint("a question for the current index", "muted")
    )
    monitor.reply(*lines)


# --- /find ----------------------------------------------------------------


def find_usages(
    root: Path, *, max_depth: int, max_files: int, max_file_bytes: int
) -> FindReport:
    """Every `.py` under `root` that imports spiyweb, and the index paths
    those files open with a string literal."""
    usages: list[Usage] = []
    hints: list[str] = []
    scanned = 0
    truncated = False
    root = root.resolve()
    for current, dirs, files in os.walk(root):
        depth = len(Path(current).relative_to(root).parts)
        dirs[:] = sorted(
            d for d in dirs if not d.startswith(".") and d not in PRUNED_DIRS
        )
        if depth >= max_depth:
            dirs[:] = []
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            if scanned >= max_files:
                truncated = True
                break
            scanned += 1
            path = Path(current) / name
            try:
                if path.stat().st_size > max_file_bytes:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if IMPORT_RE.match(line):
                    usages.append(Usage(path=path, line=number, text=line.strip()))
                for match in OPEN_RE.finditer(line):
                    hint = match.group(2)
                    if hint not in hints:
                        hints.append(hint)
        if truncated:
            break
    return FindReport(
        usages=tuple(usages),
        index_hints=tuple(hints),
        scanned=scanned,
        truncated=truncated,
    )


def _find(monitor: Monitor, _: Invocation) -> None:
    cfg = monitor.config
    report = find_usages(
        monitor.cwd,
        max_depth=cfg.find_max_depth,
        max_files=cfg.find_max_files,
        max_file_bytes=cfg.find_max_file_bytes,
    )
    g = monitor.glyphs
    if not report.usages:
        monitor.reply(
            monitor.paint("nothing imports spiyweb under this folder", "warn")
            + monitor.paint(f"  ({report.scanned} files read)", "muted"),
            tone="warn",
        )
        return
    lines: list[str] = []
    for usage in report.usages:
        try:
            shown = usage.path.relative_to(monitor.cwd.resolve())
        except ValueError:
            shown = usage.path
        lines.append(
            monitor.paint(f"{shown}:{usage.line}", "accent")
            + "   "
            + monitor.paint(usage.text, "muted")
        )
    for hint in report.index_hints:
        resolved = _resolve_hint(monitor, hint)
        if resolved is not None:
            monitor.suggested_index = str(resolved)
            lines.append(
                monitor.paint(f"index {g.to} {resolved}", "good")
                + monitor.paint("  (used by /query and /lint)", "dim")
            )
        else:
            lines.append(
                monitor.paint(f"index {g.to} {hint}", "muted")
                + monitor.paint("  (not found from here)", "dim")
            )
    if report.truncated:
        lines.append(monitor.paint(f"stopped after {report.scanned} files", "warn"))
    monitor.reply(*lines)


def _resolve_hint(monitor: Monitor, hint: str) -> Path | None:
    from spiyweb.cli import _is_index

    candidate = Path(hint)
    if not candidate.is_absolute():
        candidate = monitor.cwd / candidate
    return candidate if _is_index(candidate) else None


# --- /install ---------------------------------------------------------------


def pip_command(
    extra: str,
    *,
    python: str = sys.executable,
    has_pip: bool | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> list[str] | None:
    """How to install `spiyweb[extra]` into THIS interpreter: pip when it has
    one, `uv pip` when uv is around, otherwise nothing (say so)."""
    if has_pip is None:
        from importlib.util import find_spec

        has_pip = find_spec("pip") is not None
    if has_pip:
        return [python, "-m", "pip", "install", f"spiyweb[{extra}]"]
    uv = which("uv")
    if uv:
        return [uv, "pip", "install", "--python", python, f"spiyweb[{extra}]"]
    return None


def _extras_table(monitor: Monitor) -> list[str]:
    from spiyweb.cli import EXTRAS, _installed

    rows = {name: _installed(modules) for name, modules in sorted(EXTRAS.items())}
    rows["index"] = all(rows[name] for name in ("store", "embed", "entity"))
    g = monitor.glyphs
    lines = [monitor.paint("extras", "bold")]
    for name, present in rows.items():
        mark = (
            monitor.paint(g.live, "good") if present else monitor.paint(g.off, "muted")
        )
        state = "installed" if present else "not installed"
        lines.append(
            f"{mark} "
            + monitor.paint(f"[{name}]".ljust(10), "accent")
            + monitor.paint(state, "muted" if present else "warn")
        )
    lines.append(monitor.paint("/install <extra> installs one from here", "dim"))
    return lines


def _install(monitor: Monitor, invocation: Invocation) -> None:
    if not invocation.args:
        monitor.reply(*_extras_table(monitor))
        return
    extra = invocation.args[0].strip("[]").lower()
    if extra not in INSTALLABLE:
        monitor.reply(
            monitor.paint(f"no extra called {extra!r}", "warn")
            + monitor.paint(f" - one of {', '.join(INSTALLABLE)}", "muted"),
            tone="warn",
        )
        return
    command = pip_command(extra)
    if command is None:
        monitor.reply(
            monitor.paint("neither pip nor uv is available here", "warn"),
            monitor.paint(f'run by hand: pip install "spiyweb[{extra}]"', "muted"),
            tone="warn",
        )
        return
    shown = " ".join(command)

    def answered(answer: str) -> None:
        if answer.strip().lower() not in ("y", "yes"):
            monitor.reply(monitor.paint("not installed", "muted"), tone="muted")
            return
        monitor.reply(monitor.paint(f"running {shown}", "muted"))
        monitor.reply(monitor.paint("this can take a few minutes...", "dim"))
        monitor.draw(monitor.clock(), force=True)
        result = _run_install(command)
        tail = [line for line in result.output.splitlines() if line.strip()][-6:]
        monitor.reply(*(monitor.paint(line, "muted") for line in tail))
        monitor.reply(*_extras_table(monitor))

    monitor.ask(f"run {shown} ? (y/n)", answered, default="n")


@dataclass(frozen=True)
class InstallResult:
    code: int
    output: str


def _run_install(command: list[str]) -> InstallResult:
    try:
        done = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as failure:
        return InstallResult(code=1, output=str(failure))
    return InstallResult(code=done.returncode, output=done.stdout + done.stderr)


# --- indexes ----------------------------------------------------------------


def _with_index(
    monitor: Monitor, given: str | None, question: str, then: Callable[[str], None]
) -> None:
    """Find the index to act on: the argument, the suggestion, a choice from
    what `discover()` sees nearby, or a typed path - in that order."""
    from spiyweb.cli import _is_index
    from spiyweb.wizard import discover

    if given and _is_index(Path(given)):
        then(given)
        return
    if given:
        monitor.reply(
            monitor.paint(f"{given} is not an index (no nodes.json in it)", "warn"),
            tone="warn",
        )
        return

    def _typed(path: str) -> None:
        if path and _is_index(Path(path)):
            then(path)
        elif path:
            monitor.reply(monitor.paint(f"{path} is not an index", "warn"), tone="warn")

    if monitor.suggested_index and _is_index(Path(monitor.suggested_index)):
        then(monitor.suggested_index)
        return
    found = discover(monitor.cwd)
    if found:
        options = [(str(f.path), f"{f.path}  {f.detail}") for f in found]
        options.append(("", "somewhere else (type a path)"))

        def chosen(value: str) -> None:
            if value:
                then(value)
            else:
                monitor.ask("path to an index directory:", _typed)

        monitor.pick(question, options, chosen)
        return
    monitor.ask("path to an index directory (none found nearby):", _typed)


def _lint(monitor: Monitor, invocation: Invocation) -> None:
    def run(index: str) -> None:
        monitor.run_captured(["lint", index])

    _with_index(
        monitor, invocation.args[0] if invocation.args else None, "which corpus?", run
    )


def _index(monitor: Monitor, invocation: Invocation) -> None:
    args = list(invocation.args)

    def have_docs(docs: str) -> None:
        def have_out(out: str) -> None:
            monitor.run_captured(["index", docs, out])

        if len(args) >= 2:
            have_out(args[1])
        else:
            monitor.ask("where should the index go?", have_out, default="my-index")

    if args:
        have_docs(args[0])
    else:
        monitor.ask("folder of .txt/.md files to index:", have_docs, default="docs")


def _query(monitor: Monitor, invocation: Invocation) -> None:
    from spiyweb.cli import _is_index

    args = list(invocation.args)
    given: str | None = None
    if args and _is_index(Path(args[0])):
        given = args.pop(0)
    question = " ".join(args)

    def with_index(index: str) -> None:
        if question:
            run_query(monitor, index, question)
        else:
            monitor.ask(
                "what do you want to ask it?", lambda q: run_query(monitor, index, q)
            )

    _with_index(monitor, given, "which index?", with_index)


def _question(monitor: Monitor, text: str) -> None:
    if not text:
        return
    if monitor.suggested_index:
        run_query(monitor, monitor.suggested_index, text)
        return
    monitor.reply(
        monitor.paint("no index chosen yet", "warn")
        + monitor.paint(" - /find looks for one, /query lets you pick", "muted"),
        tone="warn",
    )


def run_query(monitor: Monitor, index_path: str, question: str) -> None:
    """Ask `index_path` and play the record here, never through the file."""
    from spiyweb.cli import Problem, _open
    from spiyweb.config import TraceConfig

    monitor.suggested_index = index_path
    try:
        index = monitor.indexes.get(index_path)
        if index is None:
            index = _open(index_path, trace=TraceConfig(attach_dir=None))
            monitor.indexes[index_path] = index
        answer = index.retrieve(question, profile=monitor.settings.profile)  # type: ignore[attr-defined]
    except Problem as problem:
        monitor.reply(monitor.paint(problem_hint(str(problem)), "warn"), tone="warn")
        return
    except Exception as failure:
        monitor.reply(
            monitor.paint(f"{type(failure).__name__}: {failure}", "warn"), tone="warn"
        )
        return
    passages = list(answer.passages)[:3]
    after = [monitor.paint("top passages", "bold")]
    for passage in passages:
        text = " ".join(passage.text.split())
        if len(text) > 160:
            text = text[:159] + "…"
        after.append(
            monitor.paint(f"{passage.energy:5.2f} ", "accent")
            + monitor.paint(passage.node_id, "muted")
            + "  "
            + text
        )
    if answer.trace is None:
        monitor.reply(*after)
        return
    monitor.play(answer.trace, after=after)


# --- /replay, /clear, /config, /menu, /version -------------------------------


def _replay(monitor: Monitor, invocation: Invocation) -> None:
    count = 1
    if invocation.args and invocation.args[0].isdigit():
        count = max(1, int(invocation.args[0]))
    records = list(monitor.played)
    if not records:
        from spiyweb.trace import load_traces

        try:
            records = list(load_traces(monitor.directory))
        except (OSError, ValueError) as failure:
            monitor.reply(monitor.paint(str(failure), "warn"), tone="warn")
            return
    if not records:
        monitor.reply(monitor.paint("nothing recorded yet", "muted"), tone="muted")
        return
    for record in records[-count:]:
        monitor.play(record)


def _clear(monitor: Monitor, _: Invocation) -> None:
    monitor.transcript.clear()
    monitor.dirty = True


def _config(monitor: Monitor, _: Invocation) -> None:
    from spiyweb.watch import HOP_DELAYS, PROFILES

    def options() -> list[tuple[str, str]]:
        s = monitor.settings
        on = monitor.paint
        return [
            (
                "profile",
                f"profile        {on(s.profile, 'accent')}   (how the web spreads)",
            ),
            ("map", f"ring map       {on('on' if s.map_enabled else 'off', 'accent')}"),
            ("pet", f"spider         {on('on' if s.pet_enabled else 'off', 'accent')}"),
            (
                "hop",
                f"hop delay      {on(str(s.hop_delay_ms) + ' ms', 'accent')}"
                + "   (0 = last frame only)",
            ),
            (
                "ascii",
                f"glyphs         {on('ascii' if s.ascii else 'unicode', 'accent')}",
            ),
            ("color", f"colour         {on('on' if s.color else 'off', 'accent')}"),
            ("done", monitor.paint("done", "muted")),
        ]

    def chosen(value: str) -> None:
        s = monitor.settings
        if value == "profile":
            s.profile = PROFILES[(PROFILES.index(s.profile) + 1) % len(PROFILES)]
        elif value == "map":
            s.map_enabled = not s.map_enabled
        elif value == "pet":
            s.pet_enabled = not s.pet_enabled
        elif value == "hop":
            current = (
                HOP_DELAYS.index(s.hop_delay_ms) if s.hop_delay_ms in HOP_DELAYS else 0
            )
            s.hop_delay_ms = HOP_DELAYS[(current + 1) % len(HOP_DELAYS)]
        elif value == "ascii":
            s.ascii = not s.ascii
        elif value == "color":
            s.color = not s.color
        else:
            monitor.picker = None
            monitor.save_settings()
            monitor.reply(monitor.paint("settings saved", "muted"), tone="muted")
            return
        monitor.save_settings()
        picker = monitor.picker
        if picker is not None:
            picker.options = options()

    monitor.pick(
        "config  (enter or space toggles, esc closes)", options(), chosen, sticky=True
    )


def _menu(monitor: Monitor, _: Invocation) -> None:
    from spiyweb.wizard import run_wizard

    monitor.suspended(run_wizard)


def _version(monitor: Monitor, _: Invocation) -> None:
    monitor.run_captured(["version"])


def _quit(monitor: Monitor, _: Invocation) -> None:
    monitor.reply(
        monitor.paint(
            "the monitor lives as long as this terminal - close the tab to leave",
            "muted",
        ),
        tone="muted",
    )


COMMANDS: tuple[Command, ...] = (
    Command("help", "/help", "this list", _help),
    Command(
        "find", "/find", "files that import spiyweb, and the index they open", _find
    ),
    Command(
        "query", "/query [index] <question>", "ask an index and watch it spread", _query
    ),
    Command(
        "lint", "/lint [index]", "islands, hubs, duplicates - no query needed", _lint
    ),
    Command(
        "index", "/index [docs] [out]", "a folder of .txt/.md becomes a graph", _index
    ),
    Command(
        "install", "/install [extra]", "what is installed; install an extra", _install
    ),
    Command("replay", "/replay [n]", "play the last n queries again", _replay),
    Command(
        "config", "/config", "profile, map, spider, speed - with arrow keys", _config
    ),
    Command("clear", "/clear", "wipe the transcript", _clear),
    Command("menu", "/menu", "the old numbered menu", _menu),
    Command("version", "/version", "version and extras", _version),
    Command("quit", "/quit", "(close the terminal instead)", _quit),
)
