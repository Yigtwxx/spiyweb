"""Bare `spiyweb`: the terminal is the interface.

Open a terminal in the project folder, type `spiyweb`, and the window is
taken over - a welcome box with the spider, a transcript, a status line, a
boxed prompt for `/` commands, a status bar. Nothing is asked. In another
terminal the application runs; every query it makes lands here within a
poll and is played hop by hop.

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

import io
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
from spiyweb.config import WatchConfig
from spiyweb.keys import (
    BACKSPACE,
    DELETE,
    DISABLE_FOCUS,
    DOWN,
    ENABLE_FOCUS,
    END_KEY,
    ENTER,
    ESCAPE,
    FOCUS_IN,
    FOCUS_OUT,
    HOME_KEY,
    LEFT,
    NAMED,
    RIGHT,
    TAB,
    UP,
    poll_raw,
)
from spiyweb.pet import pet_lines, pet_width
from spiyweb.terminal import (
    CLEAR_SCREEN,
    ERASE_LINE,
    HIDE_CURSOR,
    HOME,
    SHOW_CURSOR,
    cursor_to,
    pad,
    printed_width,
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
    from collections.abc import Callable, Sequence

__all__ = ["Job", "Marker", "Monitor", "TraceTail", "interactive", "run_monitor"]

SETTINGS_FILENAME = "monitor.json"
"""Where `/config` choices persist, next to the marker."""

STOP_HOLD_MS = 1200
"""How long the finished picture stays live before it joins the transcript."""

CAUGHT_MS = 500
"""The "query caught" beat before the first hop starts."""

MAP_MIN_ROWS = 7
"""A map shorter than this is a smudge; below it the ranking stands alone."""

DOUBLE_INTERRUPT_S = 1.0
"""Two ctrl-c inside this window leave; one clears the prompt."""

JOB_LINES_PER_TICK = 20
"""How much of a job's output one tick may log; the rest waits its turn."""


PROFILES = ("explore", "precise", "compare")
HOP_DELAYS = (0, 300, 650, 1000)


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


@dataclass
class Job:
    """A shell command started with `!`, running beside the monitor.

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

    @classmethod
    def start(cls, number: int, command: str, cwd: Path) -> Job:
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        job = cls(number=number, command=command, process=process)

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
                self.done = True
                self.exit_code = self.process.wait()
                break
            out.append(item)
        return out, self.done

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()


@dataclass
class Settings:
    """What `/config` can change; persisted as JSON next to the marker."""

    profile: str = "explore"
    map_enabled: bool = True
    pet_enabled: bool = True
    hop_delay_ms: int = 650
    ascii: bool = False
    color: bool = True

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
            hop_delay_ms=self.config.hop_delay_ms,
        )
        self.settings = Settings.load(self.directory / SETTINGS_FILENAME, base=base)
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
        self.history: list[str] = []
        self.history_at = 0
        self.tab_seed: str | None = None
        self.queue: deque[tuple[TraceRecord, tuple[str, ...]]] = deque()
        self.played: list[TraceRecord] = []
        self.playing: Play | None = None
        self.picker: Picker | None = None
        self.prompt: Prompt | None = None
        self.last_record_at: float | None = None
        self.last_interrupt = -1e9
        self.suggested_index: str | None = None
        self.indexes: dict[str, object] = {}
        self.jobs: list[Job] = []
        self.running = True
        self.dirty = True
        self.drawn: list[str] = []
        self.last_size = (0, 0)
        self.focused = True
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

    # --- settings ----------------------------------------------------------

    def apply_settings(self) -> None:
        """Rebuild the style after `/config` changed something."""
        self.config = replace(
            self.config,
            map_enabled=self.settings.map_enabled,
            pet_enabled=self.settings.pet_enabled,
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

    # --- painting helpers --------------------------------------------------

    def paint(self, text: str, *styles: str) -> str:
        return self.style.paint(text, *styles)

    def log(self, *lines: str) -> None:
        """Append to the transcript, the way a reply appears under a prompt."""
        self.transcript.extend(lines)
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

    def boxed(self, rows: list[str], width: int) -> list[str]:
        g = self.glyphs
        inner = width - 4
        edge = self.paint(g.v, "dim")
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
                "type a command - / lists them, ! runs a shell command", "dim"
            )
        return self.boxed([self.paint(g.prompt, "accent", "bold") + " " + body], width)

    def status_bar(self, width: int) -> str:
        g = self.glyphs
        left = self.paint("  ? for shortcuts", "dim") + self.paint(
            f"  {g.dot}  /help", "dim"
        )
        live = self.last_record_at is not None
        dot = self.paint(g.live, "good") if live else self.paint(g.off, "muted")
        right = (
            dot
            + " "
            + self.paint("attached" if live else "listening", "muted")
            + " "
            + self.paint(g.dot, "dim")
            + " "
            + self.paint(self._shown_dir(), "muted")
            + " "
            + self.paint(g.dot, "dim")
            + " "
            + self.paint(f"{len(self.played)} played", "muted")
            + (
                " "
                + self.paint(g.dot, "dim")
                + " "
                + self.paint(f"{self.tail.skipped} unreadable", "warn")
                if self.tail.skipped
                else ""
            )
            + "  "
        )
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
        body += self.picker_lines(width)
        body = body[-room:] if room > 0 else []
        body += [""] * (room - len(body))
        return [pad(line, width) for line in top + body + bottom]

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
        self.played.append(play.record)
        self.playing = None

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
        if key != TAB:
            self.tab_seed = None
        if key == "\x03":
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
        The prefix typed before the first tab is what keeps cycling."""
        from spiyweb.commands import COMMANDS

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

    def shortcuts(self) -> list[str]:
        rows = [
            ("/", "commands - keep typing to narrow, tab completes"),
            ("tab", "complete the command (again: the next match)"),
            ("up / down", "earlier commands"),
            ("left / right", "move inside the line; home / end, ctrl-a / ctrl-e"),
            ("esc", "clear the line, or close a list"),
            ("ctrl-u", "clear the line"),
            ("ctrl-l", "clear the transcript"),
            ("any key", "skip a playing query to its last frame"),
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
        line, self.buffer = self.buffer.strip(), ""
        if self.prompt is not None:
            prompt, self.prompt = self.prompt, None
            prompt.on_answer(line or prompt.default)
            return
        if not line:
            return
        self.history.append(line)
        self.history_at = len(self.history)
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
            + self.paint(f"  pid {job.process.pid} - /jobs lists, /kill stops", "dim")
        )

    def pump_jobs(self) -> None:
        for job in self.jobs:
            if job.done:
                continue
            lines, ended = job.drain(JOB_LINES_PER_TICK)
            for line in lines:
                self.log(
                    self.paint(f"  {self.glyphs.v} ", "dim")
                    + self.paint(f"{job.number} ", "muted")
                    + line
                )
            if ended:
                tone = "good" if job.exit_code == 0 else "warn"
                self.reply(
                    self.paint(f"job {job.number} ended", tone)
                    + self.paint(f"  exit {job.exit_code}", "dim"),
                    tone=tone,
                )
            if lines or ended:
                self.dirty = True

    def stop_jobs(self) -> None:
        for job in self.jobs:
            job.stop()

    # --- the loop ----------------------------------------------------------

    def tick(self, now: float) -> None:
        self.tick_count += 1
        if self.marker.beat():
            pass
        for record in self.tail.poll():
            self.last_record_at = now
            self.play(record)
        self.pump_jobs()
        self._advance(now)
        if self.playing is not None or self.tick_count % 12 == 0:
            self.dirty = True

    def run(self) -> int:
        out = self.write
        assert out is not None and self.poll is not None
        out(HIDE_CURSOR + CLEAR_SCREEN + ENABLE_FOCUS)
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
                now = self.clock()
                self.tick(now)
                if self.dirty:
                    self.draw(now)
                key = self.poll(self.config.poll_ms / 1000)
                if key is not None:
                    self.handle_key(key, self.clock())
                    if self.dirty:
                        self.draw(self.clock())
        except KeyboardInterrupt:
            pass
        except Exception as failure:
            if self.debug:
                import traceback

                self.debug("crash\n" + traceback.format_exc())
            out(DISABLE_FOCUS + SHOW_CURSOR + "\n")
            raise failure
        finally:
            self.stop_jobs()
            self.marker.stop()
            out(DISABLE_FOCUS + SHOW_CURSOR + "\n")
            self.flush()
        return 0

    # --- for commands ------------------------------------------------------

    def run_captured(self, argv: list[str]) -> int:
        """Run a CLI verb in-process, its output landing in the transcript."""
        from contextlib import redirect_stderr, redirect_stdout

        from spiyweb.cli import Problem, main

        self.echo("spiyweb " + " ".join(argv))
        buffer = io.StringIO()
        code = 0
        try:
            with redirect_stdout(buffer), redirect_stderr(buffer):
                code = main(argv)
        except Problem as problem:
            from spiyweb.commands import problem_hint

            self.reply(self.paint(problem_hint(str(problem)), "warn"), tone="warn")
            return 1
        except KeyboardInterrupt:
            self.reply(self.paint("interrupted", "warn"), tone="warn")
            return 130
        lines = [line.rstrip() for line in buffer.getvalue().splitlines()]
        while lines and not lines[-1]:
            lines.pop()
        if lines:
            self.reply(*lines)
        return int(code)

    def suspended(self, action: Callable[[], int]) -> int:
        """Give the real terminal to `action` (the old wizard), then take it back."""
        out = self.write
        assert out is not None
        out(SHOW_CURSOR + CLEAR_SCREEN)
        try:
            return action()
        finally:
            out(HIDE_CURSOR + CLEAR_SCREEN)
            self.drawn = []
            self.dirty = True


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
