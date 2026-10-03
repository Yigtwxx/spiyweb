"""Bare `spiyweb`: the terminal is the interface.

Open a terminal in the project folder, type `spiyweb`, and the window is
taken over - a welcome box with the SPIYWEB wordmark and the spider, a
transcript, a status line, a boxed prompt for `/` commands, a status bar.
Nothing is asked. In another terminal the application runs; every
query it makes lands here within a poll and is played hop by hop.

How the two processes meet: the monitor drops `<attach_dir>/watch` and keeps
touching it (the heartbeat); the library's `TraceStore` sees a fresh marker
and appends each record to `<attach_dir>/traces.jsonl`; the monitor tails
that file. No socket, no server, no change to the application's code, and a
crashed monitor stops being obeyed by itself once the marker goes stale.

The loop never blocks on anything but one poll slice: keys and file growth
are checked together, so typing stays responsive while a query plays and a
query arriving mid-command does not wait for Enter. Only the terminal
closing ends it - the owner's decision - with ctrl-c twice inside a second
kept as the emergency exit.

Output goes through one `write` and time comes from one `clock`, both
injectable, which is what makes the whole screen testable without a tty.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from spiyweb import __version__
from spiyweb.animate import Glyphs, Style, block_height, query_block
from spiyweb.banner import wordmark_lines, wordmark_width
from spiyweb.config import WatchConfig
from spiyweb.keys import (
    BACKSPACE,
    CTRL_C,
    DELETE,
    DISABLE_FOCUS,
    DISABLE_MOUSE,
    DOWN,
    ENABLE_FOCUS,
    ENABLE_MOUSE,
    END_KEY,
    ENTER,
    ESCAPE,
    FOCUS_IN,
    FOCUS_OUT,
    HOME_KEY,
    LEFT,
    NAMED,
    PAGE_DOWN,
    PAGE_UP,
    RIGHT,
    TAB,
    UP,
    is_mouse,
    parse_mouse,
    poll_raw,
)
from spiyweb.nearby import complete_path, find_indexes, is_index, split_last_token
from spiyweb.pet import FULL, pet_lines, pet_width
from spiyweb.progress import JobProgress
from spiyweb.results import honesty_lines, preview, ranked_passages
from spiyweb.terminal import (
    CLEAR_SCREEN,
    ERASE_LINE,
    HIDE_CURSOR,
    HOME,
    SHOW_CURSOR,
    cursor_to,
    enable_windows_vt_input,
    pad,
    printed_width,
    restore_windows_input,
    supports_color,
    supports_screen,
    supports_unicode,
    terminal_size,
)
from spiyweb.trace import (
    TRACE_FILENAME,
    WATCH_MARKER,
    TraceRecord,
    ensure_private_dir,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from spiyweb.core.propagate import PropagationResult
    from spiyweb.profiles import Profile
    from spiyweb.session import SpiywebIndex

__all__ = ["Job", "Marker", "Monitor", "TraceTail", "interactive", "run_monitor"]

SETTINGS_FILENAME = "monitor.json"
"""Where `/config` choices persist, next to the marker."""

HISTORY_FILENAME = "history"
"""Typed lines, one per row, for up/down across sessions."""

BORDER_TAIL = 2
"""Border characters right of a box label, before the corner."""

LABEL_MIN = 12
"""A box label with fewer columns than this left to it says nothing."""

STOP_HOLD_MS = 1200
"""How long the finished picture stays live before it joins the transcript."""

CAUGHT_MS = 500
"""The "query caught" beat before the first hop starts."""

MAP_MIN_ROWS = 7
"""A map shorter than this is a smudge; below it the ranking stands alone."""

BELL = "\a"
"""The terminal bell: a long job finished while nobody was looking."""

DOUBLE_INTERRUPT_S = 1.0
"""Two ctrl-c inside this window leave; one clears the prompt."""

JOB_LINES_PER_TICK = 20
"""How much of a job's output one tick may log; the rest waits its turn."""


MOUSE_SCROLL_ROWS = 3
"""Transcript rows one wheel notch moves."""

PAGE_ROWS = 10
"""Transcript rows page up / page down move."""


PROFILES = ("explore", "precise", "compare")
HOP_DELAYS = (0, 300, 650, 1000)

MARK_GAP = 2
"""Columns between the wordmark, the spider and the text beside them - two,
as the owner's other tools set their text beside their marks."""

TEXT_BESIDE_MIN = 40
"""The welcome text goes beside the wordmark and the spider only with this
many columns to spare; under them otherwise."""


class Marker:
    """The monitor's presence, as a file the library can `stat`."""

    def __init__(
        self,
        directory: Path,
        *,
        heartbeat_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.directory = directory
        self.path = directory / WATCH_MARKER
        self._heartbeat_s = heartbeat_s
        self._clock = clock
        self._last = -1e9

    def start(self) -> None:
        ensure_private_dir(self.directory)
        payload = {
            "pid": os.getpid(),
            "version": __version__,
            "started_at": time.time(),
        }
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        self._last = self._clock()

    def beat(self) -> bool:
        """Touch the marker when a heartbeat is due; say whether it was."""
        now = self._clock()
        if now - self._last < self._heartbeat_s:
            return False
        try:
            os.utime(self.path, None)
        except OSError:
            self.start()
        self._last = now
        return True

    def stop(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass


class TraceTail:
    """New complete records at the end of a JSONL file, poll after poll."""

    def __init__(self, path: Path, *, start_at_end: bool = True) -> None:
        self.path = path
        self.offset = 0
        self.skipped = 0
        self._buffer = b""
        if start_at_end:
            try:
                self.offset = self.path.stat().st_size
            except OSError:
                self.offset = 0

    def poll(self) -> list[TraceRecord]:
        try:
            size = self.path.stat().st_size
        except OSError:
            self.offset, self._buffer = 0, b""
            return []
        if size < self.offset:  # truncated or replaced: start over
            self.offset, self._buffer = 0, b""
        if size == self.offset:
            return []
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            chunk = handle.read(size - self.offset)
        self.offset = size
        self._buffer += chunk
        *lines, self._buffer = self._buffer.split(b"\n")
        records: list[TraceRecord] = []
        for raw in lines:
            if not raw.strip():
                continue
            try:
                records.append(TraceRecord.from_dict(json.loads(raw.decode("utf-8"))))
            except (ValueError, KeyError, TypeError):
                self.skipped += 1
        return records


@dataclass
class Play:
    """One record being revealed: which hop, how far into it, done yet."""

    record: TraceRecord
    started: float
    hop: int = 0
    t: float = 0.0
    caught: bool = True
    stopped: bool = False
    phase_start: float = 0.0
    after: tuple[str, ...] = ()
    """Lines logged under the committed picture - a query's top passages."""

    def final(self, now: float) -> None:
        self.caught = False
        self.hop = self.record.hops_used
        self.t = 1.0
        self.stopped = True
        self.phase_start = now


@dataclass
class Picker:
    """An arrow-key list: `/config`, an index to query, a profile."""

    title: str
    options: list[tuple[str, str]]
    on_choose: Callable[[str], None]
    cursor: int = 0
    hint: str = "up/down to move, enter to choose, esc to close"
    sticky: bool = False


@dataclass
class Prompt:
    """A question waiting on the input line, and what to do with the answer."""

    question: str
    on_answer: Callable[[str], None]
    default: str = ""


PYTHON_NAMES = ("python", "python3", "py")


def same_python(command: str) -> str:
    """`! python app.py` means THIS python - the one that has spiyweb - not
    whichever one the shell finds first. Anything else runs as typed."""
    parts = command.split(None, 1)
    if parts and parts[0].lower() in PYTHON_NAMES:
        rest = parts[1] if len(parts) > 1 else ""
        return f'"{sys.executable}" {rest}'.rstrip()
    return command


@dataclass
class Job:
    """A child process running beside the monitor: a `!` shell command, or
    a verb the monitor started itself (`/index`, `/lint`, `/install`).

    Its output arrives through a queue fed by a reader thread and is logged
    a few lines per tick, so a chatty server never stalls the screen. The
    child gets no stdin: the keyboard belongs to the monitor.
    """

    number: int
    command: str
    process: subprocess.Popen[str]
    lines: queue.Queue[str | None] = field(default_factory=queue.Queue)
    done: bool = False
    exit_code: int | None = None
    started: float = 0.0
    progress: JobProgress | None = None
    """Reads the job's lines into a status - and keeps bar lines out of the
    transcript. `None` for a `!` command: its output is the point."""
    on_done: Callable[[Job], None] | None = None
    """Called once when the process has ended and its output is logged."""
    stopped: bool = False
    """Ended by `/kill` rather than by itself."""
    eof: bool = False
    """Its output has ended. `done` follows once the exit code is in - two
    steps, so an interrupt between them is retried, never lost."""

    @classmethod
    def start(cls, number: int, command: str, cwd: Path) -> Job:
        """`! <command>`: through the shell, exactly as typed."""
        command = same_python(command)
        return cls._launch(number, command, command, cwd, shell=True)

    @classmethod
    def spawn(
        cls,
        number: int,
        argv: Sequence[str],
        cwd: Path,
        *,
        shown: str,
        started: float = 0.0,
        progress: JobProgress | None = None,
        on_done: Callable[[Job], None] | None = None,
    ) -> Job:
        """An argv, no shell: paths with spaces stay one argument, nothing
        typed is ever interpreted, and stopping it stops the real child."""
        job = cls._launch(number, list(argv), shown, cwd, shell=False)
        job.started, job.progress, job.on_done = started, progress, on_done
        return job

    @classmethod
    def _launch(
        cls,
        number: int,
        command: str | list[str],
        shown: str,
        cwd: Path,
        *,
        shell: bool,
    ) -> Job:
        environment = dict(os.environ)
        here = str(Path(sys.executable).parent)
        environment["PATH"] = here + os.pathsep + environment.get("PATH", "")
        # A piped Python child buffers its stdout in blocks: without this a
        # job's progress would arrive all at once, when it ends.
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.Popen(
            command,
            shell=shell,
            cwd=str(cwd),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **own_process_group(),
        )
        job = cls(number=number, command=shown, process=process)

        def pump() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                job.lines.put(line.rstrip("\r\n"))
            job.lines.put(None)

        threading.Thread(target=pump, daemon=True).start()
        return job

    def drain(self, limit: int) -> tuple[list[str], bool]:
        """Up to `limit` waiting lines, and whether the process has ended."""
        out: list[str] = []
        while len(out) < limit:
            try:
                item = self.lines.get_nowait()
            except queue.Empty:
                break
            if item is None:
                self.eof = True
                break
            out.append(item)
        if self.eof and not self.done:
            self.exit_code = self.process.wait()
            self.done = True
            stream = getattr(self.process, "stdout", None)
            if stream is not None:
                stream.close()
        return out, self.done

    def stop(self) -> None:
        """The whole group, not only its leader: a shell's child or a
        server's workers would otherwise outlive the job - and, holding its
        output open, keep it "running" after the leader is gone."""
        if self.done:
            return
        self.stopped = True
        stop_group(self.process)
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()


def own_process_group() -> dict[str, object]:
    """Popen options that give a job its own process group, so the ctrl-c
    meant for the monitor never reaches it - the monitor decides whether a
    ctrl-c leaves, and a job is stopped with `/kill`, not by accident."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def stop_group(process: subprocess.Popen[str]) -> None:
    if sys.platform == "win32":
        # `terminate()` would stop only the shell; taskkill takes the tree.
        done = subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
        )
        if done.returncode != 0:
            process.terminate()
        return
    import signal  # pragma: no cover - exercised off Windows

    try:  # pragma: no cover
        os.killpg(process.pid, signal.SIGTERM)
    except OSError:  # pragma: no cover
        process.terminate()


@dataclass
class Settings:
    """What `/config` can change; persisted as JSON next to the marker."""

    profile: str = "explore"
    map_enabled: bool = True
    pet_enabled: bool = True
    banner_enabled: bool = True
    hop_delay_ms: int = 650
    ascii: bool = False
    color: bool = True
    mouse: bool = False
    """Off: the terminal keeps its own selection and copy. On: the wheel
    scrolls here and a click picks - and copying needs shift+drag."""
    index: str = ""
    """The index a typed question goes to; `""` while none is chosen."""
    warm: bool = False
    """A follow-up lands on the ground the last question warmed (D22). Off
    by default: an unrelated question would inherit the last one's region."""

    @classmethod
    def load(cls, path: Path, *, base: Settings | None = None) -> Settings:
        """The saved choices over `base` - the caller's config - so a config
        passed in code is the default and the file only records changes."""
        base = base if base is not None else cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return base
        known = {f: raw[f] for f in cls.__dataclass_fields__ if f in raw}
        try:
            return replace(base, **known)
        except TypeError:
            return base

    def save(self, path: Path) -> None:
        try:
            ensure_private_dir(path.parent)
            path.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")
        except OSError:
            pass


@dataclass
class Monitor:
    """The screen, the loop, and the state the commands act on."""

    config: WatchConfig = field(default_factory=WatchConfig)
    directory: Path | None = None
    write: Callable[[str], None] | None = None
    flush: Callable[[], None] | None = None
    poll: Callable[[float], str | None] | None = None
    clock: Callable[[], float] = time.monotonic
    size: Callable[[], tuple[int, int]] = terminal_size
    color: bool | None = None
    unicode: bool | None = None
    cwd: Path = field(default_factory=Path.cwd)

    def __post_init__(self) -> None:
        self.directory = Path(self.directory or self.config.attach_dir)
        if not self.directory.is_absolute():
            self.directory = self.cwd / self.directory
        if self.write is None:
            self.write = sys.stdout.write
            self.flush = sys.stdout.flush
        self.flush = self.flush or (lambda: None)
        self.poll = self.poll or poll_raw
        base = Settings(
            map_enabled=self.config.map_enabled,
            pet_enabled=self.config.pet_enabled,
            banner_enabled=self.config.banner_enabled,
            hop_delay_ms=self.config.hop_delay_ms,
        )
        self.settings = Settings.load(self.directory / SETTINGS_FILENAME, base=base)
        if self.settings.index and not is_index(self.settings.index):
            self.settings.index = ""  # moved or deleted since the last session
        if self.color is None:
            self.color = supports_color() and self.settings.color
        if self.unicode is None:
            self.unicode = supports_unicode() and not self.settings.ascii
        self.apply_settings()
        self.marker = Marker(
            self.directory, heartbeat_s=self.config.heartbeat_s, clock=self.clock
        )
        self.tail = TraceTail(self.directory / TRACE_FILENAME)
        self.transcript: list[str] = []
        self._buffer = ""
        self.cursor = 0
        self.history: list[str] = load_history(
            self.directory / HISTORY_FILENAME, self.config.history_max
        )
        self.history_at = len(self.history)
        self.tab_seed: str | None = None
        self.queue: deque[tuple[TraceRecord, tuple[str, ...]]] = deque()
        self.played: list[TraceRecord] = []
        self.playing: Play | None = None
        self.picker: Picker | None = None
        self.prompt: Prompt | None = None
        self.last_record_at: float | None = None
        self.last_interrupt = -1e9
        self.find_hints: list[str] = []
        self.last_question = ""
        """Asked again by `/tune`, so a changed knob shows what it does."""
        self.tuned: Profile | None = None
        """`/tune`'s knobs - this session only, never saved: a saved knob
        would quietly change every later answer."""
        self.warm_from: PropagationResult | None = None
        """The last answer's activations, while `/config` warm is on."""
        """Index folders `/find` saw an application open - offered first."""
        self.hint_dismissed = False
        """The first message sent: the input box stops saying how to start."""
        self.has_nearby = bool(self.active_index) or bool(find_indexes(self.cwd))
        self.indexes: dict[str, tuple[int, SpiywebIndex]] = {}
        """Opened indexes by resolved path, with the `meta.json` stamp they
        were opened at: a rebuild changes the stamp and the next query
        reopens instead of answering from the old graph."""
        self.embedder: object | None = None
        """One embedding model for every index that names the default one -
        each `SpiywebIndex` would otherwise load its own copy."""
        self.jobs: list[Job] = []
        self.running = True
        self.dirty = True
        self.drawn: list[str] = []
        self.last_size = (0, 0)
        self.focused = True
        self.scroll = 0
        self.hits: dict[int, tuple[str, str]] = {}
        self.debug = _debug_log(self.directory)
        self.tick_count = 0

    # --- the input line ------------------------------------------------------

    @property
    def buffer(self) -> str:
        return self._buffer

    @buffer.setter
    def buffer(self, text: str) -> None:
        """Setting the whole line puts the cursor at its end - history, tab
        completion and clearing all mean "start typing from here"."""
        self._buffer = text
        self.cursor = len(text)

    def insert(self, text: str) -> None:
        self._buffer = self._buffer[: self.cursor] + text + self._buffer[self.cursor :]
        self.cursor += len(text)

    def delete_before(self) -> None:
        if self.cursor:
            self._buffer = self._buffer[: self.cursor - 1] + self._buffer[self.cursor :]
            self.cursor -= 1

    def delete_under(self) -> None:
        self._buffer = self._buffer[: self.cursor] + self._buffer[self.cursor + 1 :]

    def move(self, step: int) -> None:
        self.cursor = max(0, min(len(self._buffer), self.cursor + step))

    # --- the active index --------------------------------------------------

    @property
    def active_index(self) -> str | None:
        """Where a typed question goes; remembered across sessions."""
        return self.settings.index or None

    def set_active(self, path: str | Path | None) -> None:
        if str(path or "") != self.settings.index:
            self.warm_from = None  # another corpus: nothing of it is warm
        self.settings.index = str(path) if path else ""
        self.settings.save(self.directory / SETTINGS_FILENAME)
        if path:
            self.has_nearby = True
        self.dirty = True

    def asking_as(self) -> str:
        """How a question is asked right now, for the status bar."""
        how = "tuned" if self.tuned is not None else self.settings.profile
        if self.settings.warm and self.warm_from is not None:
            how += f" {self.glyphs.dot} warm"
        return how

    def refresh_nearby(self) -> None:
        """After an index was built or chosen: does the start hint still apply."""
        self.has_nearby = bool(self.active_index) or bool(find_indexes(self.cwd))

    # --- settings ----------------------------------------------------------

    def apply_settings(self) -> None:
        """Rebuild the style after `/config` changed something."""
        self.config = replace(
            self.config,
            map_enabled=self.settings.map_enabled,
            pet_enabled=self.settings.pet_enabled,
            banner_enabled=self.settings.banner_enabled,
            hop_delay_ms=self.settings.hop_delay_ms,
        )
        unicode = bool(self.unicode) and not self.settings.ascii
        color = bool(self.color) and self.settings.color
        self.glyphs = Glyphs.unicode() if unicode else Glyphs.ascii()
        self.style = Style(color=color, glyphs=self.glyphs, config=self.config)
        self.dirty = True

    def save_settings(self) -> None:
        self.settings.save(self.directory / SETTINGS_FILENAME)
        self.apply_settings()
        self.set_mouse(self.settings.mouse)

    def set_mouse(self, wanted: bool) -> None:
        """Mouse capture on or off, right now - never both ways at once."""
        if wanted == getattr(self, "mouse_on", False):
            return
        assert self.write is not None
        self.write(ENABLE_MOUSE if wanted else DISABLE_MOUSE)
        self.mouse_on = wanted

    # --- painting helpers --------------------------------------------------

    def paint(self, text: str, *styles: str) -> str:
        return self.style.paint(text, *styles)

    def log(self, *lines: str) -> None:
        """Append to the transcript, the way a reply appears under a prompt."""
        self.transcript.extend(lines)
        self.scroll = 0
        self.dirty = True

    def echo(self, command: str) -> None:
        self.log(
            "", " " + self.paint(self.glyphs.prompt, "accent", "bold") + " " + command
        )

    def reply(self, *lines: str, tone: str = "accent") -> None:
        g = self.glyphs
        first = True
        for line in lines:
            lead = self.paint(g.bullet, tone) + " " if first else "  "
            self.log(lead + line)
            first = False

    # --- screen ------------------------------------------------------------

    def boxed(self, rows: list[str], width: int, *, label: str = "") -> list[str]:
        """A rounded box `width` wide; `label` sits in the top border at the
        right, the way a hint does - clipped, or dropped when there is no
        room for it to say anything."""
        g = self.glyphs
        inner = width - 4
        edge = self.paint(g.v, "dim")
        room = width - 8 - BORDER_TAIL  # corners, spaces, a few h on the left
        if label and room >= LABEL_MIN:
            if len(label) > room:
                label = label[: room - len(g.ellipsis)] + g.ellipsis
            run = width - 2 - BORDER_TAIL - len(label) - 2
            top = (
                self.paint(g.tl + g.h * run + " ", "dim")
                + self.paint(label, "muted")
                + self.paint(" " + g.h * BORDER_TAIL + g.tr, "dim")
            )
        else:
            top = self.paint(g.tl + g.h * (width - 2) + g.tr, "dim")
        bottom = self.paint(g.bl + g.h * (width - 2) + g.br, "dim")
        return [top, *[edge + " " + pad(r, inner) + " " + edge for r in rows], bottom]

    def welcome(self, width: int, rows: int) -> list[str]:
        text = [
            self.paint("Welcome to spiyweb", "bold")
            + self.paint(f" {__version__}", "muted")
            + self.paint("  live monitor", "dim"),
            "",
            self.paint(f"listening in {self._shown_dir()}", "muted")
            + self.paint(f"   cwd: {self.cwd}", "dim"),
            self.paint("/help for commands", "dim"),
        ]
        marked = self.marked_welcome(width, rows, text)
        if marked:
            return marked
        if not self.config.pet_enabled:
            return self.boxed(text, width)
        frames = pet_lines(
            rows=rows, unicode=bool(self.unicode) and not self.settings.ascii
        )
        left_width = pet_width(frames) + 4
        # Blue, like everything the web does right: a gradient down the hop
        # ramp, bright at the head and deep at the last legs.
        left = [
            pad(self.style.hop(line, index * 5 // len(frames)), left_width)
            for index, line in enumerate(frames)
        ]
        right = text if len(frames) <= len(text) else ["", *text]
        height = max(len(left), len(right))
        left += [" " * left_width] * (height - len(left))
        right += [""] * (height - len(right))
        return self.boxed(
            [a + "  " + b for a, b in zip(left, right, strict=True)], width
        )

    def marked_welcome(self, width: int, rows: int, text: list[str]) -> list[str]:
        """The welcome box with the SPIYWEB wordmark, the spider to its right
        and the text beside both or under them - or nothing, and the plain
        box stands: switched off, an ASCII console (the mark is block and box
        glyphs), too narrow, or so short that the box would take a query's
        map away."""
        unicode = bool(self.unicode) and not self.settings.ascii
        if not self.config.banner_enabled or not unicode:
            return []
        mark = wordmark_lines(color=self.style.color)
        spider = FULL if self.config.pet_enabled else ()
        height = max(len(mark), len(spider))
        # The mark sits on the spider's last row: its own last row is shadow,
        # and the spider's first is only the tips of its front legs.
        mark = [" " * wordmark_width()] * (height - len(mark)) + mark
        art = [
            line
            + (
                " " * MARK_GAP + self.style.hop(spider[row], row * 5 // len(spider))
                if spider
                else ""
            )
            for row, line in enumerate(mark)
        ]
        art_width = wordmark_width() + (MARK_GAP + pet_width(spider) if spider else 0)
        inner = width - 4
        if inner < art_width:
            return []
        beside = inner - art_width - MARK_GAP
        if beside >= TEXT_BESIDE_MIN:
            right = ["", *text] if len(text) < height else text
            right += [""] * (height - len(right))
            box = self.boxed(
                [
                    line + " " * MARK_GAP + pad(words, beside)
                    for line, words in zip(art, right, strict=False)
                ],
                width,
            )
        else:
            box = self.boxed([*art, "", *[t for t in text if t]], width)
        # The bottom of the screen (status, boxed input, status bar) and the
        # line `screen()` keeps free, against the smallest map worth drawing.
        room = rows - 1 - len(box) - 5
        return box if room >= MAP_MIN_ROWS + 4 else []

    def _shown_dir(self) -> str:
        try:
            return "./" + str(self.directory.relative_to(self.cwd)).replace("\\", "/")
        except ValueError:
            return str(self.directory)

    def status_line(self, now: float) -> str:
        g = self.glyphs
        play = self.playing
        if play is not None and play.caught:
            return (
                " "
                + self.paint(g.live, "warn")
                + " "
                + self.paint("query caught", "warn", "bold")
            )
        if play is not None and not play.stopped:
            glyph = g.spin[self.tick_count % len(g.spin)]
            hint = self.paint(
                f"(hop {play.hop}/{play.record.hops_used} {g.dot} any key to skip)",
                "dim",
            )
            return (
                " "
                + self.paint(glyph, "accent")
                + " "
                + self.paint("Spreading", "accent")
                + self.paint("...", "muted")
                + " "
                + hint
            )
        job = self.busy()
        if job is not None and job.progress is not None:
            glyph = g.spin[self.tick_count % len(g.spin)]
            parts = job.progress.parts(now)
            return (
                " "
                + self.paint(glyph, "accent")
                + " "
                + self.paint(parts[0], "accent")
                + self.paint(f" {g.dot} ", "dim")
                + self.paint(f" {g.dot} ".join(parts[1:]), "muted")
                + self.paint(f"   /kill {job.number} stops it", "dim")
            )
        if self.last_record_at is None:
            return (
                " "
                + self.paint(g.off, "muted")
                + " "
                + self.paint("no app attached yet", "muted")
                + self.paint(" - run your project in another terminal", "dim")
            )
        quiet = now - self.last_record_at
        if quiet > self.config.sleep_after_s:
            return (
                " "
                + self.paint(g.off, "muted")
                + " "
                + self.paint(f"quiet for {int(quiet // 60)} min", "muted")
                + self.paint(" - waiting for the next query", "dim")
            )
        return (
            " "
            + self.paint(g.live, "good")
            + " "
            + self.paint("attached", "good")
            + self.paint(" - waiting for the next query", "muted")
        )

    def input_box(self, width: int, blink: bool) -> list[str]:
        g = self.glyphs
        caret = self.paint(g.caret, "accent", "bold") if blink and self.focused else " "
        if self.prompt is not None and not self.buffer:
            placeholder = self.prompt.question
            if self.prompt.default:
                placeholder += f"  (enter for {self.prompt.default})"
            body = caret + self.paint(placeholder, "muted")
        elif self.buffer:
            before, under, after = (
                self.buffer[: self.cursor],
                self.buffer[self.cursor : self.cursor + 1],
                self.buffer[self.cursor + 1 :],
            )
            if not under:
                body = before + caret
            elif blink and self.focused:
                body = before + self.paint(under, "reverse") + after
            else:
                body = before + under + after
        else:
            body = caret + self.paint(
                "ask a question - / lists the commands, ! runs a shell command", "dim"
            )
        hint = input_hint(
            dismissed=self.hint_dismissed,
            modal=self.prompt is not None or self.picker is not None,
            busy=self.busy() is not None,
            has_index=self.has_nearby,
        )
        return self.boxed(
            [self.paint(g.prompt, "accent", "bold") + " " + body], width, label=hint
        )

    def status_bar(self, width: int) -> str:
        """Left: how to get help. Right: which index a question goes to and
        how, then the attach state - least useful dropped first when the
        terminal is narrow."""
        g = self.glyphs
        left = self.paint("  ? for shortcuts", "dim") + self.paint(
            f"  {g.dot}  /help", "dim"
        )
        live = self.last_record_at is not None
        dot = self.paint(g.live, "good") if live else self.paint(g.off, "muted")
        index = self.active_index
        segments: list[tuple[int, str]] = [
            (
                0,
                self.paint(Path(index).name, "accent")
                if index
                else self.paint("no index", "muted"),
            ),
            (1, self.paint(self.asking_as(), "muted")),
            (0, dot + " " + self.paint("attached" if live else "listening", "muted")),
            (3, self.paint(self._shown_dir(), "muted")),
            (4, self.paint(f"{len(self.played)} played", "muted")),
        ]
        if self.tail.skipped:
            segments.append((2, self.paint(f"{self.tail.skipped} unreadable", "warn")))
        separator = " " + self.paint(g.dot, "dim") + " "

        def joined(kept: list[tuple[int, str]]) -> str:
            return separator.join(text for _, text in kept) + "  "

        kept = list(segments)
        while (
            printed_width(left) + 1 + printed_width(joined(kept)) > width
            and max(rank for rank, _ in kept) > 0
        ):
            worst = max(rank for rank, _ in kept)
            kept = [segment for segment in kept if segment[0] != worst]
        right = joined(kept)
        gap = width - printed_width(left) - printed_width(right)
        return left + " " * max(1, gap) + right

    def picker_lines(self, width: int) -> list[str]:
        picker = self.picker
        if picker is None:
            return []
        g = self.glyphs
        lines = ["", " " + self.paint(picker.title, "bold")]
        for index, (_, label) in enumerate(picker.options):
            chosen = index == picker.cursor
            dot = self.paint(
                g.pick if chosen else g.unpick, "accent" if chosen else "muted"
            )
            lines.append(
                f"   {dot} "
                + (self.paint(label, "accent", "bold") if chosen else label)
            )
        lines.append(" " + self.paint(picker.hint, "dim"))
        return [pad(line, width) for line in lines]

    def map_rows_for(self, record: TraceRecord, width: int, room: int) -> int:
        """How tall the map may be here: the configured height when it fits,
        shrunk to the room left otherwise, and zero when even a small map
        would push the ranking off the screen."""
        if width < self.config.map_min_width or not self.config.map_enabled:
            return 0
        without = block_height(record, self.style, with_map=False)
        spare = room - without
        if spare < 0:
            return 0
        rows = min(self.config.map_rows, room - 4)
        return rows if rows >= MAP_MIN_ROWS else 0

    def live_block(self, width: int, room: int) -> list[str]:
        play = self.playing
        if play is None or play.caught:
            return []
        map_rows = self.map_rows_for(play.record, width, room)
        return query_block(
            play.record,
            play.hop,
            play.t,
            self.style,
            width=width,
            with_map=map_rows > 0,
            final=play.stopped,
            map_rows=map_rows,
        )

    def screen(self, now: float) -> list[str]:
        width, rows = self.size()
        top = self.welcome(width, rows)
        blink = (self.tick_count // 12) % 2 == 0
        bottom = [
            self.status_line(now),
            *self.input_box(width, blink),
            *self.suggestion_lines(width),
            self.status_bar(width),
        ]
        room = rows - len(top) - len(bottom) - 1
        body = list(self.transcript)
        live = self.live_block(width, room)
        if live:
            keep = max(0, room - len(live) - 1)
            body = (body[-keep:] if keep else []) + ["", *live]
        picker_at = len(body)
        picker = self.picker_lines(width)
        body += picker
        self.scroll = max(0, min(self.scroll, max(0, len(body) - room)))
        end = len(body) - self.scroll
        start = max(0, end - room)
        body = body[start:end] if room > 0 else []
        body += [""] * (room - len(body))
        self.hits = self._hit_rows(len(top), room, picker_at, start, end)
        return [pad(line, width) for line in top + body + bottom]

    def _hit_rows(
        self, top: int, room: int, picker_at: int, start: int, end: int
    ) -> dict[int, tuple[str, str]]:
        """Which screen row (1-based) a click would choose: a picker option
        or a command suggestion. Only what is on screen right now counts."""
        hits: dict[int, tuple[str, str]] = {}
        if self.picker is not None:
            # picker_lines(): a blank, the title, then one row per option
            for index, (value, _) in enumerate(self.picker.options):
                body_index = picker_at + 2 + index
                if start <= body_index < end:
                    hits[top + (body_index - start) + 1] = ("pick", value)
        found = self.suggestions()
        if found:
            after_input = top + room + 1 + 3  # status line, then the boxed input
            for index, (usage, _) in enumerate(found[:6]):
                hits[after_input + index + 1] = ("suggest", usage.split()[0])
        return hits

    def draw(self, now: float, *, force: bool = False) -> None:
        size = self.size()
        if size != self.last_size:
            # The console reflowed what was on screen; nothing we drew can be
            # trusted any more, so start from a clean window.
            self.last_size, self.drawn, force = size, [], True
            out = self.write
            assert out is not None
            out(CLEAR_SCREEN)
        lines = self.screen(now)
        out = self.write
        assert out is not None
        if force or len(lines) != len(self.drawn):
            out(HOME + "\n".join(ERASE_LINE + line for line in lines))
        else:
            for row, (before, line) in enumerate(zip(self.drawn, lines, strict=True)):
                if before != line:
                    out(cursor_to(row + 1) + ERASE_LINE + line)
        # No newline ends a partial redraw, and a console stdout is line
        # buffered: without this the typed characters sit in Python's buffer
        # and the screen looks frozen.
        self.flush()
        self.drawn = lines
        self.dirty = False

    # --- playing -----------------------------------------------------------

    def play(self, record: TraceRecord, after: Sequence[str] = ()) -> None:
        """Queue a record; it plays when the current one has finished."""
        self.queue.append((record, tuple(after)))
        self.dirty = True

    def _advance(self, now: float) -> None:
        play = self.playing
        if play is None:
            if self.queue:
                record, after = self.queue.popleft()
                self.playing = Play(
                    record=record, started=now, phase_start=now, after=after
                )
                self.dirty = True
            return
        ms = (now - play.phase_start) * 1000
        delay, hold = self.config.hop_delay_ms, self.config.hold_ms
        if play.caught:
            if ms >= CAUGHT_MS:
                play.caught, play.phase_start = False, now
                if delay == 0:
                    play.final(now)
        elif play.stopped:
            if ms >= STOP_HOLD_MS:
                self._commit(play)
        elif ms < delay:
            play.t = ms / delay
        elif ms < delay + hold:
            play.t = 1.0
        elif play.hop < play.record.hops_used:
            play.hop, play.t, play.phase_start = play.hop + 1, 0.0, now
        else:
            play.stopped, play.t, play.phase_start = True, 1.0, now
        self.dirty = True

    def _commit(self, play: Play) -> None:
        # Marked done first: an interrupt half-way through the writing below
        # must not leave the record playing, to be committed a second time.
        self.playing = None
        self.played.append(play.record)
        width, rows = self.size()
        map_rows = self.map_rows_for(play.record, width, rows - 12)
        self.log(
            "",
            *query_block(
                play.record,
                play.record.hops_used,
                1.0,
                self.style,
                width=width,
                with_map=map_rows > 0,
                final=True,
                map_rows=map_rows,
            ),
        )
        if play.after:
            self.reply(*play.after)
        for tone, text in honesty_lines(play.record):
            self.reply(self.paint(text, tone), tone=tone)

    def passage_lines(
        self, record: TraceRecord, texts: Mapping[str, str] | None = None
    ) -> list[str]:
        """The strongest passages under a played query, numbered for `/show`.

        Text comes from the record; `texts` fills in for a record traced
        without it (`TraceConfig.text_chars` can drop it)."""
        cfg = self.config
        texts = texts or {}
        ranked = ranked_passages(record)[: cfg.after_passages]
        if not ranked:
            return []
        rows = [
            self.paint("top passages", "bold")
            + self.paint("   /show <n> opens one", "dim")
        ]
        for number, node in enumerate(ranked, 1):
            text = node.text or texts.get(node.id, "")
            rows.append(
                self.paint(f"{number:>2} ", "muted")
                + self.paint(f"{node.energy:5.2f} ", "accent")
                + self.paint(node.id, "muted")
                + "  "
                + preview(text, cfg.preview_chars, self.glyphs.ellipsis)
            )
        return rows

    def current_record(self) -> TraceRecord | None:
        """The query on screen: the one playing, else the last one played."""
        if self.playing is not None:
            return self.playing.record
        return self.played[-1] if self.played else None

    # --- input -------------------------------------------------------------

    def ask(
        self, question: str, on_answer: Callable[[str], None], *, default: str = ""
    ) -> None:
        self.prompt = Prompt(question=question, on_answer=on_answer, default=default)
        self.buffer = ""
        self.dirty = True

    def pick(
        self,
        title: str,
        options: Sequence[tuple[str, str]],
        on_choose: Callable[[str], None],
        *,
        cursor: int = 0,
        sticky: bool = False,
    ) -> None:
        self.picker = Picker(
            title=title,
            options=list(options),
            on_choose=on_choose,
            cursor=cursor,
            sticky=sticky,
        )
        self.dirty = True

    def handle_key(self, key: str, now: float) -> None:
        self.dirty = True
        if self.debug:
            self.debug(f"key {key!r}")
        if key in (FOCUS_IN, FOCUS_OUT):
            self.focused = key == FOCUS_IN
            return
        if is_mouse(key):
            self._mouse(key)
            return
        if key != TAB:
            self.tab_seed = None
        if key == CTRL_C:
            if now - self.last_interrupt < DOUBLE_INTERRUPT_S:
                self.running = False
                return
            self.last_interrupt = now
            self.buffer, self.prompt, self.picker = "", None, None
            return
        if self.picker is not None:
            self._picker_key(key)
            return
        play = self.playing
        if (
            play is not None
            and not play.stopped
            and not self.buffer
            and key not in NAMED
            and key != "/"
        ):
            play.final(now)
            return
        if key == ENTER:
            self._submit()
        elif key == BACKSPACE:
            self.delete_before()
        elif key == DELETE:
            self.delete_under()
        elif key == LEFT:
            self.move(-1)
        elif key == RIGHT:
            self.move(1)
        elif key == HOME_KEY or key == "\x01":  # ctrl-a
            self.cursor = 0
        elif key == END_KEY or key == "\x05":  # ctrl-e
            self.cursor = len(self.buffer)
        elif key == PAGE_UP:
            self.scroll += PAGE_ROWS
        elif key == PAGE_DOWN:
            self.scroll = max(0, self.scroll - PAGE_ROWS)
        elif key == TAB:
            self._complete()
        elif key == "\x15":  # ctrl-u: the line, gone
            self.buffer = ""
        elif key == "\x0c":  # ctrl-l: the transcript, gone
            self.transcript.clear()
        elif key == "?" and not self.buffer and self.prompt is None:
            self.reply(*self.shortcuts())
        elif key == ESCAPE:
            if self.buffer:
                self.buffer = ""
            elif self.prompt is not None:
                self.prompt = None
                self.reply(self.paint("cancelled", "muted"), tone="muted")
        elif key == UP and not self.prompt:
            if self.history and self.history_at > 0:
                self.history_at -= 1
                self.buffer = self.history[self.history_at]
        elif key == DOWN and not self.prompt:
            if self.history_at < len(self.history):
                self.history_at += 1
                self.buffer = (
                    self.history[self.history_at]
                    if self.history_at < len(self.history)
                    else ""
                )
        elif key and key not in NAMED and key.isprintable():
            self.insert(key)

    def handle(self, key: str, now: float) -> None:
        """`handle_key`, but a failing command never takes the screen down:
        every command, picker and prompt callback runs under it, so one
        guard here covers them all. Drawing and ticking stay unguarded -
        an error there is a bug in the monitor, not in what was asked."""
        try:
            self.handle_key(key, now)
        except KeyboardInterrupt:
            raise
        except (Exception, SystemExit) as failure:
            self.failed(failure)

    def failed(self, failure: BaseException) -> None:
        """Say what went wrong under the prompt; the traceback goes to the
        debug log, never over the screen."""
        if self.debug:
            import traceback

            self.debug(
                "command failed\n"
                + "".join(traceback.format_exception(failure)).rstrip()
            )
        self.prompt, self.picker = None, None
        message = str(failure) or type(failure).__name__
        self.reply(
            self.paint(f"{type(failure).__name__}: {message}", "warn"), tone="warn"
        )

    def _mouse(self, key: str) -> None:
        """Wheel scrolls the transcript; a left click picks what it lands on."""
        button, _column, row, pressed = parse_mouse(key)
        if button in (64, 65):
            self.scroll += MOUSE_SCROLL_ROWS if button == 64 else -MOUSE_SCROLL_ROWS
            self.scroll = max(0, self.scroll)
            return
        if button != 0 or not pressed:
            return
        hit = self.hits.get(row)
        if hit is None:
            return
        kind, value = hit
        if kind == "pick" and self.picker is not None:
            picker = self.picker
            picker.cursor = next(
                (i for i, (v, _) in enumerate(picker.options) if v == value), 0
            )
            if not picker.sticky:
                self.picker = None
            picker.on_choose(value)
        elif kind == "suggest":
            self.buffer = value + " "

    def _picker_key(self, key: str) -> None:
        picker = self.picker
        assert picker is not None
        if key == UP:
            picker.cursor = (picker.cursor - 1) % len(picker.options)
        elif key == DOWN:
            picker.cursor = (picker.cursor + 1) % len(picker.options)
        elif key in (ENTER, " "):
            value = picker.options[picker.cursor][0]
            if not picker.sticky:
                self.picker = None
            picker.on_choose(value)
        elif key == ESCAPE or key == "q":
            self.picker = None
        elif key.isdigit() and 1 <= int(key) <= len(picker.options):
            value = picker.options[int(key) - 1][0]
            if not picker.sticky:
                self.picker = None
            picker.on_choose(value)

    def suggestions(self) -> list[tuple[str, str]]:
        """Commands the half-typed buffer could become: `/qu` -> `/query`."""
        if self.prompt is not None or not self.buffer.startswith("/"):
            return []
        if " " in self.buffer.strip():
            return []
        from spiyweb.commands import COMMANDS

        typed = self.buffer[1:].lower()
        return [
            (command.usage, command.summary)
            for command in COMMANDS
            if command.name.startswith(typed) and command.name != "quit"
        ]

    def _complete(self) -> None:
        """Tab: the one match is typed out; several, the next one in turn.
        The prefix typed before the first tab is what keeps cycling. After a
        command and a space, what is completed is a path."""
        from spiyweb.commands import COMMANDS

        if self.buffer.startswith("/") and " " in self.buffer:
            self._complete_path()
            return

        def matching(seed: str) -> list[str]:
            if not seed.startswith("/") or " " in seed:
                return []
            typed = seed[1:].lower()
            return [
                "/" + c.name
                for c in COMMANDS
                if c.name.startswith(typed) and c.name != "quit"
            ]

        current = self.buffer.strip()
        seed = self.tab_seed if self.tab_seed is not None else current
        matches = matching(seed)
        if current not in matches:  # the line was edited since the last tab
            seed, matches = current, matching(current)
        if not matches:
            return
        if current in matches and len(matches) > 1:
            nxt = matches[(matches.index(current) + 1) % len(matches)]
        else:
            nxt = matches[0]
        self.tab_seed = seed
        self.buffer = nxt + (" " if len(matches) == 1 else "")

    def _complete_path(self) -> None:
        head, token = split_last_token(self.buffer)
        seed = self.tab_seed if self.tab_seed is not None else token
        matches = complete_path(seed, self.cwd)
        if token not in matches:  # the line was edited since the last tab
            seed, matches = token, complete_path(token, self.cwd)
        if not matches:
            return
        if token in matches and len(matches) > 1:
            nxt = matches[(matches.index(token) + 1) % len(matches)]
        else:
            nxt = matches[0]
        # One match is a step taken: the next tab goes on from it, into the
        # folder. Several are a choice: the next tab offers the next one.
        self.tab_seed = seed if len(matches) > 1 else None
        self.buffer = head + nxt

    def shortcuts(self) -> list[str]:
        rows = [
            ("/", "commands - keep typing to narrow, tab completes"),
            ("tab", "complete the command or a folder (again: the next match)"),
            ("up / down", "earlier commands, remembered across sessions"),
            ("left / right", "move inside the line; home / end, ctrl-a / ctrl-e"),
            ("esc", "clear the line, or close a list"),
            ("ctrl-u", "clear the line"),
            ("ctrl-l", "clear the transcript"),
            ("any key", "skip a playing query to its last frame"),
            ("pgup / pgdn", "scroll the transcript"),
            ("mouse", "off by default so selecting and copying text works;"),
            ("", "/config turns it on: wheel scrolls, click picks, shift+drag copies"),
            ("ctrl-c twice", "leave (or just close the terminal)"),
        ]
        return [self.paint("shortcuts", "bold")] + [
            self.paint(key.ljust(14), "accent") + self.paint(what, "muted")
            for key, what in rows
        ]

    def suggestion_lines(self, width: int) -> list[str]:
        found = self.suggestions()
        if not found:
            return []
        lines = []
        for usage, summary in found[:6]:
            lines.append(
                "   "
                + self.paint(usage.ljust(28), "accent")
                + self.paint(summary, "muted")
            )
        return [pad(line, width) for line in lines]

    def _submit(self) -> None:
        self.scroll = 0
        line, self.buffer = self.buffer.strip(), ""
        if self.prompt is not None:
            self.hint_dismissed = True
            prompt, self.prompt = self.prompt, None
            prompt.on_answer(line or prompt.default)
            return
        if not line:
            return
        self.hint_dismissed = True
        self.history.append(line)
        self.history_at = len(self.history)
        if not line.startswith("!"):
            # `!` lines are never written down: `! KEY=... python app.py`
            # would leave a credential in a file.
            append_history(self.directory / HISTORY_FILENAME, line, self.config)
        if line.startswith("!"):
            self.start_job(line[1:].strip())
            return
        from spiyweb.commands import dispatch

        dispatch(self, line)

    def start_job(self, command: str) -> None:
        """`! python app.py`: run it here, in the background, watching."""
        self.echo("! " + command)
        if not command:
            self.reply(self.paint("nothing to run after the !", "warn"), tone="warn")
            return
        try:
            job = Job.start(len(self.jobs) + 1, command, self.cwd)
        except OSError as failure:
            self.reply(self.paint(str(failure), "warn"), tone="warn")
            return
        self.jobs.append(job)
        self.reply(
            self.paint(f"job {job.number} started", "good")
            + self.paint(f"  pid {job.process.pid} - /jobs lists, /kill stops", "dim"),
            self.paint(same_python(command), "dim"),
        )

    def pump_jobs(self, now: float = 0.0) -> None:
        from spiyweb.commands import problem_hint

        for job in list(self.jobs):
            if job.done:
                continue
            lines, ended = job.drain(JOB_LINES_PER_TICK)
            for line in lines:
                if job.progress is not None and job.progress.feed(line):
                    continue  # a progress bar: it lives in the status line
                if job.progress is not None:
                    if not line.strip():
                        continue
                    line = problem_hint(line)
                self.log(
                    self.paint(f"  {self.glyphs.v} ", "dim")
                    + self.paint(f"{job.number} ", "muted")
                    + line
                )
            if ended:
                self._ended(job, now)
            if lines or ended:
                self.dirty = True

    def _ended(self, job: Job, now: float) -> None:
        tone = "good" if job.exit_code == 0 else "warn"
        self.reply(
            self.paint(f"job {job.number} ended", tone)
            + self.paint(f"  exit {job.exit_code}", "dim"),
            tone=tone,
        )
        bell = self.config.bell_after_s
        if job.progress is not None and bell and now - job.started >= bell:
            # Long enough that the person has looked away: the terminal's
            # bell (a sound, or a flashing tab) says it is done.
            assert self.write is not None
            self.write(BELL)
        if job.on_done is not None:
            try:
                job.on_done(job)
            except (Exception, SystemExit) as failure:
                self.failed(failure)

    def stop_jobs(self) -> None:
        for job in self.jobs:
            job.stop()

    def spawn(
        self,
        argv: Sequence[str],
        *,
        shown: str,
        progress: JobProgress | None = None,
        on_done: Callable[[Job], None] | None = None,
    ) -> Job | None:
        """Start a child the monitor needs (a verb, pip) as a background job;
        `None`, with the reason said, when it cannot start at all."""
        try:
            job = Job.spawn(
                len(self.jobs) + 1,
                argv,
                self.cwd,
                shown=shown,
                started=self.clock(),
                progress=progress,
                on_done=on_done,
            )
        except OSError as failure:
            self.reply(self.paint(str(failure), "warn"), tone="warn")
            return None
        if progress is not None:
            progress.started = job.started
        self.jobs.append(job)
        self.reply(
            self.paint(f"job {job.number}: {shown}", "muted")
            + self.paint(f"  - in the background, /kill {job.number} stops it", "dim")
        )
        return job

    def spawn_spiyweb(
        self,
        args: Sequence[str],
        *,
        progress: JobProgress | None = None,
        on_done: Callable[[Job], None] | None = None,
    ) -> Job | None:
        """`spiyweb <args>` in a child, with THIS interpreter - never whichever
        `spiyweb` the PATH finds first. The child crashing cannot take the
        screen with it, and `/kill` can stop it."""
        return self.spawn(
            [sys.executable, "-m", "spiyweb", *args],
            shown="spiyweb " + " ".join(_quoted(arg) for arg in args),
            progress=progress or JobProgress(label=f"spiyweb {args[0]}", started=0),
            on_done=on_done,
        )

    def busy(self) -> Job | None:
        """The running job with a progress status, newest first."""
        for job in reversed(self.jobs):
            if not job.done and not job.stopped and job.progress is not None:
                return job
        return None

    # --- the loop ----------------------------------------------------------

    def tick(self, now: float) -> None:
        self.tick_count += 1
        if self.marker.beat():
            pass
        for record in self.tail.poll():
            self.last_record_at = now
            self.play(record, after=self.passage_lines(record))
        self.pump_jobs(now)
        self._advance(now)
        if self.playing is not None or self.busy() or self.tick_count % 12 == 0:
            self.dirty = True

    def run(self) -> int:
        out = self.write
        assert out is not None and self.poll is not None
        out(HIDE_CURSOR + CLEAR_SCREEN + ENABLE_FOCUS)
        previous_input = enable_windows_vt_input()
        self.mouse_on = False
        self.set_mouse(self.settings.mouse)
        self.flush()
        self.marker.start()
        if self.debug:
            self.debug(
                f"start size={self.size()} color={self.color} unicode={self.unicode}"
            )
        self.log(
            " "
            + self.paint(self.glyphs.tip, "accent")
            + " "
            + self.paint("Tip:", "bold")
            + " "
            + self.paint(
                "run your project in another terminal - "
                "every query it makes spreads here",
                "muted",
            )
        )
        try:
            while self.running:
                try:
                    now = self.clock()
                    self.tick(now)
                    if self.dirty:
                        self.draw(now)
                    key = self.poll(self.config.poll_ms / 1000)
                    if key is not None:
                        self.handle(key, self.clock())
                        if self.dirty:
                            self.draw(self.clock())
                except KeyboardInterrupt:
                    # A console turns ctrl-c into a signal, raised wherever
                    # the loop happened to be - a draw may be half written.
                    # It means what the key means: once clears, twice leaves.
                    self.drawn = []
                    self.handle_key(CTRL_C, self.clock())
        except Exception as failure:
            if self.debug:
                import traceback

                self.debug("crash\n" + traceback.format_exc())
            out(DISABLE_MOUSE + DISABLE_FOCUS + SHOW_CURSOR + "\n")
            raise failure
        finally:
            # The terminal first: stopping jobs can take seconds, and another
            # ctrl-c in them must not leave a hidden cursor and mouse capture.
            out(DISABLE_MOUSE + DISABLE_FOCUS + SHOW_CURSOR + "\n")
            self.flush()
            restore_windows_input(previous_input)
            self.marker.stop()
            try:
                self.stop_jobs()
            except KeyboardInterrupt:
                pass
        return 0


def input_hint(*, dismissed: bool, modal: bool, busy: bool, has_index: bool) -> str:
    """The one line on the input box's top border, until the first message.

    Someone who has never used the monitor sees what to type first; once
    they have typed anything it is gone for the session. Silent while a
    list or a question is open (that has its own words) or a job runs.
    """
    if dismissed or modal or busy:
        return ""
    if has_index:
        return "type a question, Enter"
    return "start: /demo or /index <folder>"


def load_history(path: Path, limit: int) -> list[str]:
    """The last `limit` lines typed in earlier sessions here."""
    if limit <= 0:
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [line for line in lines if line.strip()][-limit:]


def append_history(path: Path, line: str, config: WatchConfig) -> None:
    """One more line; the file is cut back to `history_max` once it is twice
    that long, so it never grows without bound."""
    if config.history_max <= 0:
        return
    try:
        ensure_private_dir(path.parent)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line.replace("\n", " ") + "\n")
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) > 2 * config.history_max:
            path.write_text(
                "\n".join(lines[-config.history_max :]) + "\n", encoding="utf-8"
            )
    except OSError:
        pass


def _quoted(arg: str) -> str:
    """An argument as it would have to be typed: quoted when it has a space."""
    return f'"{arg}"' if " " in arg else arg


def _debug_log(directory: Path) -> Callable[[str], None] | None:
    """`SPIYWEB_DEBUG=1`: every key and every crash into `monitor.log`, for
    the terminal that behaves differently from every other terminal."""
    if not os.environ.get("SPIYWEB_DEBUG"):
        return None
    path = directory / "monitor.log"

    def write(message: str) -> None:
        try:
            ensure_private_dir(directory)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(f"{time.strftime('%H:%M:%S')} {message}\n")
        except OSError:
            pass

    return write


def interactive(config: WatchConfig | None = None) -> int:
    """What a bare `spiyweb` runs: the monitor in a terminal, usage in a pipe,
    the numbered menu where single keys or a full screen cannot be had."""
    from spiyweb import wizard

    if not wizard.is_interactive():
        return wizard.interactive()
    if not wizard.supports_raw_input() or not supports_screen():
        return wizard.run_wizard()
    return run_monitor(config or WatchConfig())


def run_monitor(config: WatchConfig, *, directory: Path | None = None) -> int:
    from spiyweb.cli import _writable_stdout

    _writable_stdout()
    return Monitor(config=config, directory=directory).run()
