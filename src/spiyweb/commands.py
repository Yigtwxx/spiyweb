"""The `/` commands of the monitor - everything, without leaving it.

`/query`, `/lint`, `/index` and `/install` do what the CLI verbs do;
`/doctor` says what is installed and what is missing; `/find` looks for
the application that imports spiyweb and the index it opens; `/config` is
an arrow-key list of the monitor's own knobs; `/replay` plays a recorded
query again. Text without a slash is a question for the current index.

A long verb (`/index`, `/lint`, `/install`) runs as a background job - the
same `spiyweb` command in a child process, its output streaming into the
transcript and its progress into the status line - so the screen never
freezes and `/kill` can stop it. `/query` the monitor runs itself, so it can
PLAY the record instead of printing a table.

Nothing here touches the terminal: commands talk to the `Monitor` through
`log`, `reply`, `ask`, `pick` and `play`, so a fake monitor tests them.
"""

from __future__ import annotations

import importlib
import io
import json
import os
import re
import shutil
import sys
import time
import warnings
from contextlib import redirect_stderr
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from spiyweb.nearby import (
    PRUNED_DIRS,
    IndexInfo,
    find_indexes,
    is_index,
    text_folders,
)
from spiyweb.progress import IndexProgress, JobProgress
from spiyweb.results import (
    markdown,
    novelty,
    passage_detail,
    passage_of,
    preview,
    ranked_passages,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from spiyweb.session import Answer, SpiywebIndex
    from spiyweb.trace import TraceRecord
    from spiyweb.watch import Job, Monitor

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
INDEX_PARTS = ("store", "embed", "entity")
"""What `[index]` is the union of."""
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
    parts = split_args(stripped[1:])
    if not parts:
        return Invocation(name="help", args=(), raw=stripped)
    return Invocation(name=parts[0].lower(), args=tuple(parts[1:]), raw=stripped)


def split_args(text: str) -> list[str]:
    """Words, with `"..."` or `'...'` keeping a path with spaces together.

    No backslash escapes, on purpose: `shlex` would turn `C:\\Users\\me`
    into `C:Usersme`, and a Windows path is the likeliest thing typed here.
    An unclosed quote runs to the end of the line.
    """
    words: list[str] = []
    current: list[str] = []
    quote = ""
    started = False
    for char in text:
        if quote:
            if char == quote:
                quote = ""
            else:
                current.append(char)
        elif char in "\"'" and not current:
            # Only at the start of a word: the apostrophe in "Tesla's" is
            # a letter, not the opening of a quoted path.
            quote, started = char, True
        elif char.isspace():
            if started:
                words.append("".join(current))
                current, started = [], False
        else:
            current.append(char)
            started = True
    if started:
        words.append("".join(current))
    return words


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
        monitor.paint("! <command>".ljust(30), "accent")
        + monitor.paint("run a shell command here, in the background", "muted")
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
            d
            for d in dirs
            if not d.startswith(".")
            and d not in PRUNED_DIRS
            and not _is_spiyweb_itself(Path(current) / d)
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


def _is_spiyweb_itself(directory: Path) -> bool:
    """The library's own package imports itself on every line; a checkout of
    spiyweb is not an application that uses it."""
    return (
        directory.name == "spiyweb" and (directory / "core" / "propagate.py").is_file()
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
    by_file: dict[Path, list[Usage]] = {}
    for usage in report.usages:
        by_file.setdefault(usage.path, []).append(usage)
    for path, usages in by_file.items():
        try:
            shown = path.relative_to(monitor.cwd.resolve())
        except ValueError:
            shown = path
        first = usages[0]
        count = f"{len(usages)} imports" if len(usages) > 1 else "1 import"
        lines.append(
            monitor.paint(str(shown), "accent")
            + "   "
            + monitor.paint(f"{count}, first at line {first.line}", "muted")
        )
    for hint in report.index_hints:
        resolved = _resolve_hint(monitor, hint)
        if resolved is not None:
            if str(resolved) not in monitor.find_hints:
                monitor.find_hints.append(str(resolved))
            if not monitor.active_index:
                monitor.set_active(resolved)
            lines.append(
                monitor.paint(f"index {g.to} {resolved}", "good")
                + monitor.paint("  (/indexes chooses among them)", "dim")
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
    candidate = Path(hint)
    if not candidate.is_absolute():
        candidate = monitor.cwd / candidate
    return candidate if is_index(candidate) else None


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


def doctor_lines(monitor: Monitor) -> list[str]:
    """Everything that decides whether `/index` and a question will work,
    each missing piece next to the command that fixes it.

    Nothing heavy is imported to find out: extras and the spaCy model are
    `find_spec` lookups. The device is the one exception - knowing it means
    importing torch - so it is checked only when torch is there, and the
    screen says so before the pause.
    """
    import platform

    from spiyweb import __version__
    from spiyweb.cli import EXTRAS, _installed
    from spiyweb.config import EntityExtractionConfig, LLMConfig

    g = monitor.glyphs

    def row(ok: bool, name: str, state: str, fix: str = "") -> str:
        mark = monitor.paint(g.live, "good") if ok else monitor.paint(g.off, "warn")
        return (
            f"{mark} "
            + monitor.paint(name.ljust(16), "accent")
            + monitor.paint(state, "muted" if ok else "warn")
            + (monitor.paint(f"   {fix}", "dim") if fix else "")
        )

    lines = [
        monitor.paint(f"spiyweb {__version__}", "bold")
        + monitor.paint(f"  {g.dot}  python {platform.python_version()}", "muted")
    ]
    present = {name: _installed(modules) for name, modules in sorted(EXTRAS.items())}
    present["index"] = all(present[name] for name in INDEX_PARTS)
    for name, ok in present.items():
        lines.append(
            row(
                ok,
                f"[{name}]",
                "installed" if ok else "not installed",
                "" if ok else f"/install {name}",
            )
        )
    model = EntityExtractionConfig().spacy_model
    has_model = _installed((model,))
    lines.append(
        row(
            has_model,
            "spaCy model",
            model if has_model else f"{model} missing - /index needs it",
            "" if has_model else "/install index fetches it",
        )
    )
    server = LLMConfig().base_url
    answering = _answers(server, monitor.config.doctor_timeout_s)
    lines.append(
        row(
            answering,
            "LLM server",
            server if answering else f"{server} not answering",
            "" if answering else "only the proposition layer needs it (ollama serve)",
        )
    )
    if _installed(("torch",)):
        monitor.reply(monitor.paint("checking the device (imports torch) ...", "dim"))
        monitor.draw(monitor.clock(), force=True)
        from spiyweb.embedding import detect_device

        device = detect_device()
        lines.append(
            row(True, "device", device + ("" if device != "cpu" else " (slow)"))
        )
    else:
        lines.append(
            row(False, "device", "unknown - no torch yet", "it comes with [embed]")
        )
    return lines


def _answers(base_url: str, timeout_s: float) -> bool:
    """Whether anything answers at `base_url` - even an error is an answer."""
    from urllib.error import HTTPError
    from urllib.request import urlopen

    try:
        with urlopen(base_url.rstrip("/") + "/models", timeout=timeout_s):
            return True
    except HTTPError:
        return True
    except (OSError, ValueError):
        return False


def _doctor(monitor: Monitor, _: Invocation) -> None:
    monitor.reply(*doctor_lines(monitor))


def _install(monitor: Monitor, invocation: Invocation) -> None:
    if not invocation.args:
        _doctor(monitor, invocation)
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

    def installed(job: Job) -> None:
        # The running process cached "not installed" for every module it
        # looked up before; without this it keeps believing it.
        importlib.invalidate_caches()
        if job.exit_code != 0:
            monitor.reply(
                monitor.paint(f"installing [{extra}] failed", "warn"), tone="warn"
            )
            return
        if extra in ("index", "entity") and _fetch_spacy_model(monitor):
            return
        monitor.reply(*doctor_lines(monitor))

    def answered(answer: str) -> None:
        if answer.strip().lower() not in ("y", "yes"):
            monitor.reply(monitor.paint("not installed", "muted"), tone="muted")
            return
        monitor.spawn(
            command,
            shown=shown,
            progress=JobProgress(label=f"installing [{extra}]", started=0.0),
            on_done=installed,
        )

    monitor.ask(f"run {shown} ? (y/n)", answered, default="n")


def _fetch_spacy_model(monitor: Monitor) -> bool:
    """Chain the spaCy model download after `[index]`/`[entity]` - without it
    the first `/index` fails, and nothing in the monitor would say why.
    `True` when a download was started."""
    from spiyweb.cli import _installed
    from spiyweb.config import EntityExtractionConfig

    model = EntityExtractionConfig().spacy_model
    if _installed((model,)):
        return False

    def fetched(job: Job) -> None:
        importlib.invalidate_caches()
        monitor.reply(*doctor_lines(monitor))

    started = monitor.spawn(
        [sys.executable, "-m", "spacy", "download", model],
        shown=f"python -m spacy download {model}",
        progress=JobProgress(label=f"fetching {model}", started=0.0),
        on_done=fetched,
    )
    return started is not None


# --- indexes ----------------------------------------------------------------


def nearby_indexes(monitor: Monitor) -> list[IndexInfo]:
    """The one list every index choice is made from: the active index, what
    `/find` saw an application open, then what sits nearby."""
    first = [p for p in (monitor.active_index, *monitor.find_hints) if p]
    return find_indexes(monitor.cwd, first)


def index_row(monitor: Monitor, info: IndexInfo) -> str:
    """`my-index   42 nodes · semantic, entity · 2026-10-03 14:02   active`"""
    from spiyweb.config import EmbeddingConfig

    g = monitor.glyphs
    facts = []
    if info.nodes is not None:
        facts.append(f"{info.nodes} nodes")
    if info.propositions:
        facts.append(f"{info.propositions} propositions")
    if info.layers:
        facts.append(", ".join(info.layers))
    if info.built_ns:
        facts.append(
            time.strftime("%Y-%m-%d %H:%M", time.localtime(info.built_ns / 1e9))
        )
    row = monitor.paint(shown_path(monitor, info.path).ljust(18), "accent") + (
        monitor.paint("  " + f" {g.dot} ".join(facts), "muted") if facts else ""
    )
    active = monitor.active_index
    if active and Path(active).resolve() == info.path.resolve():
        row += monitor.paint("   active", "good")
    if info.model and info.model != EmbeddingConfig().model:
        row += monitor.paint(
            f"   built with {info.model}: needs its own embedder", "warn"
        )
    return row


def _with_index(
    monitor: Monitor, given: str | None, question: str, then: Callable[[str], None]
) -> None:
    """Find the index to act on: the argument, the active one, the only one
    nearby, a choice among several, or a typed path - in that order."""
    if given and is_index(given):
        then(given)
        return
    if given:
        monitor.reply(
            monitor.paint(f"{given} is not an index (no nodes.json in it)", "warn"),
            tone="warn",
        )
        return
    if monitor.active_index:
        then(monitor.active_index)
        return

    def typed(path: str) -> None:
        if path and is_index(path):
            then(path)
        elif path:
            monitor.reply(monitor.paint(f"{path} is not an index", "warn"), tone="warn")

    found = nearby_indexes(monitor)
    if len(found) == 1:
        monitor.reply(
            monitor.paint(f"using {shown_path(monitor, found[0].path)}", "muted"),
            tone="muted",
        )
        then(str(found[0].path))
        return
    if found:
        options = [(str(info.path), index_row(monitor, info)) for info in found]
        options.append(("", monitor.paint("somewhere else (type a path)", "muted")))

        def chosen(value: str) -> None:
            if value:
                then(value)
            else:
                monitor.ask("path to an index folder:", typed)

        monitor.pick(question, options, chosen)
        return
    monitor.ask("path to an index folder (none found nearby):", typed)


def _indexes(monitor: Monitor, _: Invocation) -> None:
    found = nearby_indexes(monitor)
    if not found:
        monitor.reply(
            monitor.paint("no index nearby", "warn")
            + monitor.paint(" - /index builds one from a folder of .txt/.md", "muted"),
            tone="warn",
        )
        return

    def chosen(value: str) -> None:
        monitor.set_active(value)
        monitor.reply(
            monitor.paint(f"questions now go to {shown_path(monitor, value)}", "good"),
            tone="good",
        )

    monitor.pick(
        "indexes - enter makes one the active index",
        [(str(info.path), index_row(monitor, info)) for info in found],
        chosen,
    )


def _lint(monitor: Monitor, invocation: Invocation) -> None:
    def run(index: str) -> None:
        monitor.spawn_spiyweb(
            ["lint", index],
            progress=JobProgress(label=f"linting {Path(index).name}", started=0.0),
        )

    _with_index(
        monitor, invocation.args[0] if invocation.args else None, "which corpus?", run
    )


def _index(monitor: Monitor, invocation: Invocation) -> None:
    """`/index` asks: which folder, then the options. `/index docs` opens the
    options for that folder; `/index docs out [flags]` starts at once."""
    args = list(invocation.args)
    if len(args) >= 2:
        start_index(monitor, args[0], args[1], args[2:])
    elif args:
        index_form(monitor, args[0])
    else:
        pick_folder(monitor)


def pick_folder(monitor: Monitor) -> None:
    from spiyweb.cli import TEXT_SUFFIXES

    cfg = monitor.config
    folders = text_folders(
        monitor.cwd,
        TEXT_SUFFIXES,
        max_depth=cfg.find_max_depth,
        max_files=cfg.find_max_files,
    )

    def typed(path: str) -> None:
        if path:
            index_form(monitor, path)

    if not folders:
        monitor.ask("folder of .txt/.md files to index (none found under here):", typed)
        return
    options = []
    for folder in folders:
        name = shown_path(monitor, folder.path)
        shown = "./" if name == "." else name + "/"
        count = f"{folder.files} file" + ("" if folder.files == 1 else "s")
        options.append(
            (str(folder.path), shown.ljust(24) + monitor.paint(count, "muted"))
        )
    options.append(("", monitor.paint("somewhere else (type a path)", "muted")))

    def chosen(value: str) -> None:
        if value:
            index_form(monitor, shown_path(monitor, value))
        else:
            monitor.ask("folder of .txt/.md files to index:", typed)

    monitor.pick("which folder becomes the index?", options, chosen)


def index_form(monitor: Monitor, docs: str) -> None:
    """The options of one build, as a list toggled with enter - the same
    feel as `/config`. Nothing starts until "build it" is chosen."""
    name = Path(docs).resolve().name or "corpus"
    state: dict[str, object] = {
        "out": f"{name}-index",
        "propositions": False,
        "force": False,
        "whole": False,
        "glob": "",
    }

    def flags() -> list[str]:
        out = []
        if state["propositions"]:
            out.append("--propositions")
        if state["force"]:
            out.append("--force")
        if state["whole"]:
            out.append("--whole-file")
        if state["glob"]:
            out += ["--glob", str(state["glob"])]
        return out

    def options() -> list[tuple[str, str]]:
        on = monitor.paint

        def toggle(key: str) -> str:
            return on("on" if state[key] else "off", "accent")

        target = monitor.cwd / str(state["out"])
        reuse = (
            on("   exists: finished stages are reused", "dim")
            if is_index(target) and not state["force"]
            else ""
        )
        return [
            ("out", f"output folder        {on(str(state['out']), 'accent')}{reuse}"),
            (
                "propositions",
                f"proposition layer    {toggle('propositions')}"
                + on("   local LLM (Ollama), one call per passage - slow", "dim"),
            ),
            ("force", f"rebuild from scratch {toggle('force')}"),
            (
                "whole",
                f"one passage per file {toggle('whole')}"
                + on("   off: blank lines split passages", "dim"),
            ),
            (
                "glob",
                "only files matching  "
                + on(str(state["glob"]) or "every .txt/.md", "accent"),
            ),
            ("build", on("build it", "good", "bold")),
            ("cancel", on("cancel", "muted")),
        ]

    def reopen() -> None:
        monitor.pick(title, options(), chosen, sticky=True)

    def chosen(value: str) -> None:
        if value in ("propositions", "force", "whole"):
            state[value] = not state[value]
        elif value in ("out", "glob"):
            monitor.picker = None

            def answered(text: str) -> None:
                state[value] = (
                    text.strip() if value == "glob" else (text.strip() or state[value])
                )
                reopen()

            question = (
                "where should the index go?"
                if value == "out"
                else "only files matching (e.g. **/*.md, empty for all):"
            )
            # Glob starts empty: Enter on nothing means every file again.
            default = str(state[value]) if value == "out" else ""
            monitor.ask(question, answered, default=default)
            return
        elif value == "build":
            monitor.picker = None
            start_index(monitor, docs, str(state["out"]), flags())
            return
        else:
            monitor.picker = None
            monitor.reply(monitor.paint("not built", "muted"), tone="muted")
            return
        picker = monitor.picker
        if picker is not None:
            picker.options = options()

    title = f"index {docs} - enter toggles, esc closes"
    reopen()


def start_index(
    monitor: Monitor, docs: str, out: str, flags: Sequence[str] = ()
) -> None:
    """`spiyweb index` in the background; the index it builds becomes the
    one a typed question goes to."""

    def built(job: Job) -> None:
        if job.exit_code == 0:
            target = Path(out)
            if not target.is_absolute():
                target = monitor.cwd / target
            monitor.set_active(target)
            monitor.warm_from = None  # a fresh build: nothing of it is warm
            monitor.reply(
                monitor.paint(f"{out} is ready", "good")
                + monitor.paint(" - type a question and press enter", "muted"),
                tone="good",
            )
        elif job.stopped:
            monitor.reply(
                monitor.paint("stopped - a stage may be half written", "warn"),
                monitor.paint(
                    f"run /index {docs} {out} --force to build it cleanly", "muted"
                ),
                tone="warn",
            )

    monitor.spawn_spiyweb(
        ["index", docs, out, *flags],
        progress=IndexProgress(
            label=f"indexing {Path(docs).name} {monitor.glyphs.to} {Path(out).name}",
            started=0.0,
            propositions="--propositions" in flags,
        ),
        on_done=built,
    )


def _query(monitor: Monitor, invocation: Invocation) -> None:
    args = list(invocation.args)
    given: str | None = None
    if args and is_index(args[0]):
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
    """Plain text is a question: for the active index, the only one nearby,
    or one chosen from a list - and with none at all, how to get one."""
    if not text:
        return
    if monitor.active_index or nearby_indexes(monitor):
        _with_index(
            monitor, None, "which index?", lambda index: run_query(monitor, index, text)
        )
        return
    monitor.reply(
        monitor.paint("no index here yet", "warn")
        + monitor.paint(
            " - /index builds one from your .txt/.md files, /demo shows one", "muted"
        ),
        tone="warn",
    )


def open_cached(monitor: Monitor, index_path: str) -> SpiywebIndex:
    """The opened index for `index_path`, reopened when it was rebuilt.

    Keyed by the resolved path, so `idx` and `./idx` are one index; stamped
    with the receipt's modification time, so a `/index --force` - here or in
    another terminal - is seen by the next query instead of answered from
    the graph that was in memory. The replaced index is closed.
    """
    from spiyweb.cli import _open
    from spiyweb.config import TraceConfig
    from spiyweb.nearby import index_stamp

    key = str(Path(index_path).resolve())
    stamp = index_stamp(key)
    held = monitor.indexes.get(key)
    if held is not None and held[0] == stamp:
        return held[1]
    if held is not None:
        held[1].close()
        del monitor.indexes[key]
        monitor.warm_from = None  # the old graph's node ids mean nothing now
    monitor.reply(
        monitor.paint(f"opening {shown_path(monitor, key)} ...", "dim"), tone="muted"
    )
    monitor.draw(monitor.clock(), force=True)
    options: dict[str, object] = {"trace": TraceConfig(attach_dir=None)}
    embedder = shared_embedder(monitor, key)
    if embedder is not None:
        options["embedder"] = embedder
    index = _open(index_path, **options)
    monitor.indexes[key] = (stamp, index)
    return index


def shown_path(monitor: Monitor, path: str | Path) -> str:
    """A path the way the person would type it: relative to where the
    monitor runs when it is inside, as given otherwise."""
    target = Path(path)
    try:
        return target.resolve().relative_to(monitor.cwd.resolve()).as_posix()
    except (ValueError, OSError):
        return str(target)


def shared_embedder(monitor: Monitor, index_path: str) -> object | None:
    """The monitor's one embedding model, when the index names the default.

    An index that names another model, or none (built before the store
    recorded one), gets nothing here and decides for itself - the model
    check inside `SpiywebIndex` still refuses a mismatch either way.
    """
    from spiyweb.cli import Problem
    from spiyweb.config import EmbeddingConfig
    from spiyweb.nearby import read_meta

    if read_meta(index_path).get("embedding_model") != EmbeddingConfig().model:
        return None
    if monitor.embedder is None:
        monitor.reply(
            monitor.paint("loading the embedding model (first query only) ...", "dim"),
            tone="muted",
        )
        monitor.draw(monitor.clock(), force=True)
        try:
            from spiyweb.embedding import SentenceTransformerEmbedder

            monitor.embedder = SentenceTransformerEmbedder()
        except ImportError as missing:
            raise Problem(
                f'{missing}\nasking an index needs: pip install "spiyweb[index]"'
            ) from missing
    return monitor.embedder


def answer_for(
    monitor: Monitor, index_path: str, question: str, *, follow_up: bool = True
) -> Answer | None:
    """Ask `index_path` the way the monitor is set up to ask: the chosen
    profile or the `/tune` knobs, on warm ground when that is switched on.
    `follow_up=False` asks cold and leaves the warm ground as it was - a
    knob change or a comparison must be judged on the question alone, not
    on the residue of its own previous answer. `None`, with the reason
    said, when the index could not answer."""
    from spiyweb.cli import Problem

    if (
        not monitor.active_index
        or Path(monitor.active_index).resolve() != Path(index_path).resolve()
    ):
        monitor.set_active(Path(index_path).resolve())
    monitor.last_question = question
    caught: list[warnings.WarningMessage] = []
    try:
        # A warning printed to stderr lands on top of the full-screen frame;
        # caught here, it becomes a line under the answer instead.
        with (
            warnings.catch_warnings(record=True) as caught,
            redirect_stderr(io.StringIO()),
        ):
            warnings.simplefilter("always")
            index = open_cached(monitor, index_path)
            answer = index.retrieve(question, **ask_options(monitor, follow_up))
    except Problem as problem:
        monitor.reply(monitor.paint(problem_hint(str(problem)), "warn"), tone="warn")
        return None
    finally:
        for warning in caught:
            monitor.reply(monitor.paint(str(warning.message), "warn"), tone="warn")
    if monitor.settings.warm and follow_up:
        monitor.warm_from = answer.result.propagation
    if monitor.tuned is not None and answer.trace is not None:
        # A raw config runs nameless; the head of the picture would otherwise
        # call it by the default profile's name.
        answer = replace(answer, trace=replace(answer.trace, profile="tuned"))
    return answer


def ask_options(monitor: Monitor, follow_up: bool = True) -> dict[str, object]:
    """`retrieve()` keywords for how this monitor asks."""
    from spiyweb.config import RetrievalConfig, ThermalConfig

    options: dict[str, object] = {}
    if monitor.tuned is not None:
        # A config supplied by the caller is never overlaid by a profile.
        options["config"] = monitor.tuned.as_retrieval(RetrievalConfig())
    else:
        options["profile"] = monitor.settings.profile
    if follow_up and monitor.settings.warm and monitor.warm_from is not None:
        from spiyweb.thermal import residue_of

        residue = residue_of(monitor.warm_from, ThermalConfig())
        if residue:
            options["residue"] = residue
    return options


def run_query(
    monitor: Monitor, index_path: str, question: str, *, follow_up: bool = True
) -> None:
    """Ask `index_path` and play the record here, never through the file."""
    answer = answer_for(monitor, index_path, question, follow_up=follow_up)
    if answer is None:
        return
    texts = {passage.node_id: passage.text for passage in answer.passages}
    if answer.trace is None:
        monitor.reply(
            monitor.paint("tracing is off for this index - nothing to play", "warn"),
            tone="warn",
        )
        return
    after = monitor.passage_lines(answer.trace, texts)
    record = answer.trace
    warmed = record.injected_energy > record.settings.get(
        "seed_energy", record.injected_energy
    )
    if record.hops_used == 0 and not warmed:
        # The web never left the seeds: say why, from the graph, without an
        # LLM (the structural refusal) - this is the one place the monitor
        # holds the graph the report needs.
        after += [
            monitor.paint(line, "muted") for line in answer.refusal().text.splitlines()
        ]
    monitor.play(answer.trace, after=after)


# --- /demo -------------------------------------------------------------------


DEMO_RESOURCE = "demo.jsonl"
"""Three queries recorded by `examples/make_demo.py`, shipped in the wheel."""


def demo_records() -> list[TraceRecord]:
    """The recorded demo queries - plain JSON, so playing them needs nothing."""
    from importlib.resources import files

    from spiyweb.trace import TraceRecord

    text = files("spiyweb").joinpath(DEMO_RESOURCE).read_text(encoding="utf-8")
    return [
        TraceRecord.from_dict(json.loads(line)) for line in text.splitlines() if line
    ]


def _demo(monitor: Monitor, _: Invocation) -> None:
    from spiyweb.cli import EXTRAS, _installed

    records = demo_records()
    monitor.reply(
        monitor.paint(
            f"{len(records)} questions about Wardenclyffe Tower, recorded with the "
            "real pipeline on a small sample corpus",
            "muted",
        )
    )
    ready = all(_installed(EXTRAS[name]) for name in INDEX_PARTS)
    yours = "/index <folder>" if ready else "/install index, then /index <folder>"
    for number, record in enumerate(records, 1):
        after = monitor.passage_lines(record)
        if number == len(records):
            after += [
                "",
                monitor.paint("that was a recording", "bold")
                + monitor.paint(f" - to ask your own documents: {yours}", "muted"),
            ]
        monitor.play(record, after=after)


# --- /compare, /tune, /reset ----------------------------------------------------


def _compare(monitor: Monitor, invocation: Invocation) -> None:
    question = " ".join(invocation.args)

    def with_index(index: str) -> None:
        if question:
            run_compare(monitor, index, question)
        else:
            monitor.ask(
                "what should both of them answer?",
                lambda q: run_compare(monitor, index, q) if q else None,
            )

    _with_index(monitor, None, "which index?", with_index)


def run_compare(monitor: Monitor, index_path: str, question: str) -> None:
    """The same question, two ways: the web, and plain top-k at the same
    `k` - and which passages only one of them found."""
    from spiyweb.evaluation.baseline import topk_retrieve

    key = str(Path(index_path).resolve())
    embedder = shared_embedder(monitor, key)
    if embedder is None:
        monitor.reply(
            monitor.paint(
                "/compare needs an index built with the default model", "warn"
            ),
            monitor.paint(
                "top-k embeds the question again, and only that model is held here",
                "muted",
            ),
            tone="warn",
        )
        return
    answer = answer_for(monitor, index_path, question, follow_up=False)
    if answer is None or answer.trace is None:
        return
    web = [node.id for node in ranked_passages(answer.trace)]
    k = min(len({passage_of(n) for n in web}), monitor.config.max_rows)
    vector = embedder.embed_queries([question])[0]  # type: ignore[attr-defined]
    # Over-fetched, then folded: on a two-layer index several propositions
    # stand for one passage, and k counts passages.
    topk = topk_retrieve(vector, open_cached(monitor, index_path).store, 3 * k)
    found = novelty(web, topk, k)
    texts = {node.id: node.text for node in answer.trace.nodes}
    hops = {node.id: node.hop for node in answer.trace.nodes}
    after = monitor.passage_lines(answer.trace, texts)
    after += [
        "",
        monitor.paint(f"the web against plain top-k  (k={k}, same index)", "bold"),
        monitor.paint(f"  both found       {found.both}/{k}", "muted"),
        monitor.paint(f"  only the web     {len(found.only_web)}", "good"),
    ]
    for passage in found.only_web:
        after.append(
            "     "
            + monitor.paint(passage, "accent")
            + monitor.paint(f"  hop {hops.get(passage, '?')}  ", "muted")
            + preview(texts.get(passage, ""), 70, monitor.glyphs.ellipsis)
        )
    after.append(
        monitor.paint(f"  only top-k       {len(found.only_topk)}", "muted")
        + (
            monitor.paint("   " + ", ".join(found.only_topk), "dim")
            if found.only_topk
            else ""
        )
    )
    monitor.play(answer.trace, after=after)


def _tune(monitor: Monitor, invocation: Invocation) -> None:
    """`/tune damping 0.8 seed 6`: the three knobs a profile sets, for this
    session only, and the last question asked again with them."""
    from spiyweb.profiles import PROFILES, Profile

    args = [arg.lower() for arg in invocation.args]
    base = monitor.tuned or PROFILES[monitor.settings.profile]
    if not args:
        source = "tuned" if monitor.tuned else f"profile {base.name}"
        monitor.reply(
            monitor.paint(f"{source}: ", "muted")
            + f"damping {base.damping}  threshold {base.threshold_ratio}  "
            f"seed {base.seed_width}",
            monitor.paint(
                "/tune damping 0.8 seed 6 changes them, /tune reset drops them - "
                "this session only",
                "dim",
            ),
        )
        return
    if args == ["reset"]:
        monitor.tuned = None
        monitor.reply(
            monitor.paint(f"back to the {monitor.settings.profile} profile", "muted"),
            tone="muted",
        )
        _again(monitor)
        return
    knobs = {
        "damping": base.damping,
        "threshold": base.threshold_ratio,
        "seed": float(base.seed_width),
    }
    if len(args) % 2:
        monitor.reply(
            monitor.paint("pairs of name and value: /tune damping 0.8", "warn"),
            tone="warn",
        )
        return
    for name, value in zip(args[::2], args[1::2], strict=True):
        if name not in knobs:
            monitor.reply(
                monitor.paint(f"no knob {name!r} - damping, threshold, seed", "warn"),
                tone="warn",
            )
            return
        try:
            knobs[name] = float(value)
        except ValueError:
            monitor.reply(
                monitor.paint(f"{value!r} is not a number", "warn"), tone="warn"
            )
            return
    try:
        monitor.tuned = Profile(
            name="tuned",
            damping=knobs["damping"],
            threshold_ratio=knobs["threshold"],
            seed_width=int(knobs["seed"]),
        )
    except ValueError as invalid:
        monitor.reply(monitor.paint(str(invalid), "warn"), tone="warn")
        return
    _again(monitor)


def _again(monitor: Monitor) -> None:
    """Ask the last question again, so a changed knob shows what it does."""
    if monitor.last_question and monitor.active_index:
        run_query(monitor, monitor.active_index, monitor.last_question, follow_up=False)


def _reset(monitor: Monitor, _: Invocation) -> None:
    monitor.warm_from = None
    monitor.reply(
        monitor.paint("cold again - the next question starts from nothing", "muted"),
        tone="muted",
    )


# --- /show, /save --------------------------------------------------------------


def _show(monitor: Monitor, invocation: Invocation) -> None:
    """`/show 2` opens the second passage of the query on screen; `/show`
    lists them to choose from, and stays open to open several."""
    record = monitor.current_record()
    if record is None:
        monitor.reply(
            monitor.paint("nothing has been asked yet - type a question", "muted"),
            tone="muted",
        )
        return
    ranked = ranked_passages(record)
    if not ranked:
        monitor.reply(monitor.paint("that query activated nothing", "muted"))
        return
    width = monitor.size()[0]

    def open_one(number: int) -> None:
        if not 1 <= number <= len(ranked):
            monitor.reply(
                monitor.paint(f"there are {len(ranked)} passages", "warn"), tone="warn"
            )
            return
        monitor.reply(
            *passage_detail(record, ranked[number - 1], monitor.style, width=width)
        )

    if invocation.args:
        if invocation.args[0].isdigit():
            open_one(int(invocation.args[0]))
            return
        monitor.reply(
            monitor.paint("/show takes a number: /show 2", "warn"), tone="warn"
        )
        return
    shown = ranked[: max(monitor.config.max_rows, monitor.config.after_passages)]
    options = [
        (
            str(number),
            f"{number:>2}  "
            + monitor.paint(f"{node.energy:5.2f} ", "accent")
            + monitor.paint(node.id, "muted")
            + "  "
            + preview(node.text, 60, monitor.glyphs.ellipsis),
        )
        for number, node in enumerate(shown, 1)
    ]
    monitor.pick(
        f"passages of {preview(record.query, 50, monitor.glyphs.ellipsis)!r}"
        " - enter opens, esc closes",
        options,
        lambda value: open_one(int(value)),
        sticky=True,
    )


def _save(monitor: Monitor, invocation: Invocation) -> None:
    """The query on screen as a Markdown file in this folder."""
    record = monitor.current_record()
    if record is None:
        monitor.reply(monitor.paint("nothing to save yet", "muted"), tone="muted")
        return
    name = (
        invocation.args[0]
        if invocation.args
        else time.strftime("spiyweb-%Y%m%d-%H%M%S.md")
    )
    target = Path(name)
    if not target.is_absolute():
        target = monitor.cwd / target
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(markdown(record), encoding="utf-8")
    except OSError as failure:
        monitor.reply(monitor.paint(str(failure), "warn"), tone="warn")
        return
    monitor.reply(
        monitor.paint(f"saved {shown_path(monitor, target)}", "good"), tone="good"
    )


# --- /replay, /clear, /config ---------------------------------------------------


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
                "banner",
                f"banner         {on('on' if s.banner_enabled else 'off', 'accent')}",
            ),
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
            (
                "mouse",
                f"mouse          {on('on' if s.mouse else 'off', 'accent')}"
                + "   (on: wheel scrolls, click picks; copy with shift+drag)",
            ),
            (
                "warm",
                f"follow-ups     {on('warm' if s.warm else 'cold', 'accent')}"
                + "   (warm: a question starts where the last one spread)",
            ),
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
        elif value == "banner":
            s.banner_enabled = not s.banner_enabled
        elif value == "hop":
            current = (
                HOP_DELAYS.index(s.hop_delay_ms) if s.hop_delay_ms in HOP_DELAYS else 0
            )
            s.hop_delay_ms = HOP_DELAYS[(current + 1) % len(HOP_DELAYS)]
        elif value == "ascii":
            s.ascii = not s.ascii
        elif value == "color":
            s.color = not s.color
        elif value == "mouse":
            s.mouse = not s.mouse
        elif value == "warm":
            s.warm = not s.warm
            monitor.warm_from = None  # switching starts from cold either way
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


def _jobs(monitor: Monitor, _: Invocation) -> None:
    if not monitor.jobs:
        monitor.reply(
            monitor.paint("no jobs - `! python app.py` starts one", "muted"),
            tone="muted",
        )
        return
    lines = []
    for job in monitor.jobs:
        state = (
            monitor.paint(f"exit {job.exit_code}", "muted")
            if job.done
            else monitor.paint("running", "good")
        )
        lines.append(
            monitor.paint(f"{job.number}", "accent")
            + "  "
            + state
            + "  "
            + monitor.paint(job.command, "muted")
        )
    monitor.reply(*lines)


def _kill(monitor: Monitor, invocation: Invocation) -> None:
    running = [job for job in monitor.jobs if not job.done and not job.stopped]
    if not running:
        monitor.reply(monitor.paint("nothing is running", "muted"), tone="muted")
        return
    target = running[-1]
    if invocation.args and invocation.args[0].isdigit():
        wanted = int(invocation.args[0])
        matches = [job for job in running if job.number == wanted]
        if not matches:
            monitor.reply(
                monitor.paint(f"no running job {wanted}", "warn"), tone="warn"
            )
            return
        target = matches[0]
    target.stop()
    monitor.reply(monitor.paint(f"job {target.number} stopped", "warn"), tone="warn")


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
    Command("demo", "/demo", "watch three recorded questions spread", _demo),
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
        "indexes", "/indexes", "the indexes nearby; choose the active one", _indexes
    ),
    Command(
        "doctor", "/doctor", "what is installed, what is missing, the fix", _doctor
    ),
    Command("install", "/install [extra]", "install an extra from here", _install),
    Command("show", "/show [n]", "a passage in full: text, source, path", _show),
    Command("compare", "/compare <question>", "the web against plain top-k", _compare),
    Command("tune", "/tune [knob value]", "damping, threshold, seed - live", _tune),
    Command("reset", "/reset", "forget the warm ground of earlier questions", _reset),
    Command("save", "/save [file]", "the query on screen as a Markdown file", _save),
    Command("replay", "/replay [n]", "play the last n queries again", _replay),
    Command(
        "config", "/config", "profile, map, spider, speed - with arrow keys", _config
    ),
    Command("clear", "/clear", "wipe the transcript", _clear),
    Command("jobs", "/jobs", "shell commands started with !", _jobs),
    Command("kill", "/kill [n]", "stop a running job (the last by default)", _kill),
    Command("quit", "/quit", "(close the terminal instead)", _quit),
)
