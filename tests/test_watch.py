"""The attach handshake and the monitor loop, driven without a terminal."""

from __future__ import annotations

import io
import json
import os
import sys
import time
from typing import TYPE_CHECKING

import pytest

from conftest import canonical_record
from spiyweb.config import TraceConfig, WatchConfig
from spiyweb.terminal import SHOW_CURSOR
from spiyweb.trace import (
    TRACE_FILENAME,
    WATCH_MARKER,
    TraceStore,
    attached_trace_path,
    load_traces,
)
from spiyweb.watch import Marker, Monitor, TraceTail

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from spiyweb.session import SpiywebIndex


# --- the library side: TraceStore obeys a fresh marker ------------------------


@pytest.fixture(autouse=True)
def _attach_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPIYWEB_NO_ATTACH", raising=False)


def _fresh_marker(directory: Path) -> Path:
    directory.mkdir(exist_ok=True)
    marker = directory / WATCH_MARKER
    marker.write_text("{}", encoding="utf-8")
    return marker


def test_attached_trace_path_reads_the_marker_age_from_the_clock(
    tmp_path: Path,
) -> None:
    marker = _fresh_marker(tmp_path)
    stamp = marker.stat().st_mtime
    assert attached_trace_path(tmp_path, stale_s=10.0, now=stamp + 5) == (
        tmp_path / TRACE_FILENAME
    )
    assert attached_trace_path(tmp_path, stale_s=10.0, now=stamp + 11) is None
    assert attached_trace_path(tmp_path / "absent", stale_s=10.0) is None


def test_a_fresh_marker_makes_the_store_append_and_ignore_its_folder(
    tmp_path: Path,
) -> None:
    attach = tmp_path / ".spiyweb"
    _fresh_marker(attach)
    store = TraceStore(TraceConfig(attach_dir=attach))
    store.append(canonical_record())
    written = attach / TRACE_FILENAME
    assert written.read_text(encoding="utf-8").count("\n") == 1
    assert (attach / ".gitignore").read_text(encoding="utf-8") == "*\n"
    assert load_traces(attach)[0].query == canonical_record().query
    assert store.path is None, "the configured path stays untouched"


def test_a_stale_marker_and_a_missing_marker_write_nothing(tmp_path: Path) -> None:
    attach = tmp_path / ".spiyweb"
    marker = _fresh_marker(attach)
    old = time.time() - 60
    os.utime(marker, (old, old))
    TraceStore(TraceConfig(attach_dir=attach)).append(canonical_record())
    assert not (attach / TRACE_FILENAME).exists()
    TraceStore(TraceConfig(attach_dir=tmp_path / "nowhere")).append(canonical_record())
    assert not (tmp_path / "nowhere").exists()


def test_attach_can_be_switched_off_and_a_directory_wins(tmp_path: Path) -> None:
    attach = tmp_path / ".spiyweb"
    _fresh_marker(attach)
    TraceStore(TraceConfig(attach_dir=None)).append(canonical_record())
    assert not (attach / TRACE_FILENAME).exists()
    explicit = tmp_path / "explicit"
    TraceStore(TraceConfig(directory=explicit, attach_dir=attach)).append(
        canonical_record()
    )
    assert (explicit / TRACE_FILENAME).exists()
    assert not (attach / TRACE_FILENAME).exists()


def test_an_unwritable_attach_dir_never_reaches_the_app(tmp_path: Path) -> None:
    attach = tmp_path / ".spiyweb"
    _fresh_marker(attach)
    (attach / TRACE_FILENAME).mkdir()  # a directory where the file should be
    store = TraceStore(TraceConfig(attach_dir=attach))
    assert store.append(canonical_record()) is not None


def test_a_real_query_lands_in_the_attach_dir(
    tmp_path: Path, open_tiny: Callable[..., SpiywebIndex]
) -> None:
    attach = tmp_path / ".spiyweb"
    _fresh_marker(attach)
    index = open_tiny(TraceConfig(attach_dir=attach))
    index.retrieve("who built the tower")
    records = load_traces(attach)
    assert len(records) == 1 and records[0].query == "who built the tower"


# --- the monitor side: Marker and TraceTail -----------------------------------


def test_marker_starts_beats_and_stops(tmp_path: Path) -> None:
    now = [100.0]
    marker = Marker(tmp_path / ".spiyweb", heartbeat_s=2.0, clock=lambda: now[0])
    marker.start()
    payload = json.loads(marker.path.read_text(encoding="utf-8"))
    assert payload["pid"] == os.getpid()
    assert (tmp_path / ".spiyweb" / ".gitignore").exists()
    assert marker.beat() is False, "too soon for a heartbeat"
    now[0] += 2.5
    assert marker.beat() is True
    marker.stop()
    assert not marker.path.exists()
    marker.stop()  # idempotent


def _append(path: Path, *records: object) -> None:
    with path.open("ab") as handle:
        for record in records:
            handle.write((json.dumps(record.to_dict()) + "\n").encode("utf-8"))  # type: ignore[attr-defined]


def test_tail_sees_only_what_arrives_after_it_started(tmp_path: Path) -> None:
    path = tmp_path / TRACE_FILENAME
    _append(path, canonical_record(sequence=0), canonical_record(sequence=1))
    tail = TraceTail(path)
    assert tail.poll() == []
    _append(path, canonical_record(sequence=2))
    got = tail.poll()
    assert [r.sequence for r in got] == [2]


def test_tail_waits_for_a_partial_line_to_complete(tmp_path: Path) -> None:
    path = tmp_path / TRACE_FILENAME
    tail = TraceTail(path)
    line = json.dumps(canonical_record().to_dict())
    with path.open("ab") as handle:
        handle.write(line[:40].encode("utf-8"))
    assert tail.poll() == []
    with path.open("ab") as handle:
        handle.write((line[40:] + "\n").encode("utf-8"))
    assert len(tail.poll()) == 1


def test_tail_restarts_when_the_file_shrinks_and_counts_broken_lines(
    tmp_path: Path,
) -> None:
    path = tmp_path / TRACE_FILENAME
    _append(path, canonical_record(sequence=0), canonical_record(sequence=1))
    tail = TraceTail(path)
    path.write_bytes(b"not json\n")
    assert tail.poll() == []
    assert tail.skipped == 1
    _append(path, canonical_record(sequence=5))
    assert [r.sequence for r in tail.poll()] == [5]


def test_tail_survives_a_file_that_does_not_exist_yet(tmp_path: Path) -> None:
    path = tmp_path / TRACE_FILENAME
    tail = TraceTail(path)
    assert tail.poll() == []
    _append(path, canonical_record())
    assert len(tail.poll()) == 1


# --- the monitor loop -----------------------------------------------------------


class Driver:
    """A scripted keyboard and a fake clock: each poll is one tick."""

    def __init__(self, keys: list[str | None], *, tick_s: float = 0.05) -> None:
        self.keys = list(keys)
        self.now = 0.0
        self.tick_s = tick_s
        self.monitor: Monitor | None = None
        self.on_tick: dict[int, Callable[[], None]] = {}
        self.polls = 0

    def clock(self) -> float:
        return self.now

    def poll(self, _timeout: float) -> str | None:
        self.polls += 1
        self.now += self.tick_s
        if self.polls in self.on_tick:
            self.on_tick[self.polls]()
        if not self.keys:
            assert self.monitor is not None
            self.monitor.running = False
            return None
        key = self.keys.pop(0)
        if key == INTERRUPT:
            raise KeyboardInterrupt  # what a console's ctrl-c signal does
        return key


INTERRUPT = "<sigint>"
"""A scripted key that arrives the way a real console delivers ctrl-c."""


def make_monitor(
    tmp_path: Path,
    driver: Driver,
    *,
    size: tuple[int, int] = (120, 30),
    **knobs: object,
) -> tuple[Monitor, io.StringIO]:
    out = io.StringIO()
    monitor = Monitor(
        config=WatchConfig(**{"hop_delay_ms": 100, "hold_ms": 50, **knobs}),  # type: ignore[arg-type]
        directory=tmp_path / ".spiyweb",
        write=out.write,
        poll=driver.poll,
        clock=driver.clock,
        size=lambda: size,
        color=False,
        unicode=True,
        cwd=tmp_path,
    )
    driver.monitor = monitor
    return monitor, out


def test_the_monitor_drops_a_marker_and_removes_it_when_it_ends(
    tmp_path: Path,
) -> None:
    driver = Driver([None, None])
    monitor, out = make_monitor(tmp_path, driver)
    seen: list[bool] = []
    driver.on_tick[1] = lambda: seen.append(monitor.marker.path.exists())
    assert monitor.run() == 0
    assert seen == [True]
    assert not monitor.marker.path.exists()
    assert out.getvalue().endswith(SHOW_CURSOR + "\n")


def test_a_record_appended_while_watching_is_played_then_committed(
    tmp_path: Path,
) -> None:
    driver = Driver([None] * 60)
    monitor, _out = make_monitor(tmp_path, driver)
    path = tmp_path / ".spiyweb" / TRACE_FILENAME

    def land() -> None:
        _append(path, canonical_record())

    driver.on_tick[2] = land
    monitor.run()
    assert len(monitor.played) == 1
    assert monitor.playing is None
    joined = "\n".join(monitor.transcript)
    assert "stopped: threshold" in joined and "converging" in joined
    assert "1 played" in "\n".join(monitor.screen(driver.now))


def test_typing_edits_the_buffer_and_enter_runs_the_command(tmp_path: Path) -> None:
    driver = Driver([*"/helx", "backspace", "p", "enter", None])
    monitor, _ = make_monitor(tmp_path, driver)
    monitor.run()
    joined = "\n".join(monitor.transcript)
    assert "/help" in joined and "/find" in joined and "/config" in joined


def test_an_unknown_command_is_answered_not_crashed(tmp_path: Path) -> None:
    driver = Driver([*"/nope", "enter", None])
    monitor, _ = make_monitor(tmp_path, driver)
    monitor.run()
    assert "no such command /nope" in "\n".join(monitor.transcript)


def test_escape_and_quit_do_not_leave_only_the_terminal_does(tmp_path: Path) -> None:
    driver = Driver(["escape", *"/quit", "enter", "escape", None])
    monitor, _ = make_monitor(tmp_path, driver)
    monitor.run()
    assert monitor.running is False, "the driver stopped it, nothing else did"
    assert "close the tab" in "\n".join(monitor.transcript)


def test_ctrl_c_twice_inside_a_second_is_the_emergency_exit(tmp_path: Path) -> None:
    driver = Driver(["\x03", "\x03", "x", "x", "x"])
    monitor, _ = make_monitor(tmp_path, driver)
    monitor.run()
    assert driver.keys == ["x", "x", "x"], "left on the second ctrl-c"


def test_a_lone_ctrl_c_only_clears_the_prompt(tmp_path: Path) -> None:
    driver = Driver([*"/fi", "\x03", None])
    monitor, _ = make_monitor(tmp_path, driver)
    monitor.run()
    assert monitor.buffer == ""
    assert monitor.running is False


def test_a_signalled_ctrl_c_clears_the_line_and_keeps_running(tmp_path: Path) -> None:
    driver = Driver([*"/fi", INTERRUPT, *"ab"])
    monitor, _ = make_monitor(tmp_path, driver)
    monitor.run()
    assert monitor.buffer == "ab", "the monitor survived and kept taking keys"


def test_two_signalled_ctrl_c_inside_a_second_leave(tmp_path: Path) -> None:
    driver = Driver([INTERRUPT, INTERRUPT, "x", "x"])
    monitor, _ = make_monitor(tmp_path, driver)
    monitor.run()
    assert driver.keys == ["x", "x"], "left on the second signal"


def test_two_signalled_ctrl_c_far_apart_do_not_leave(tmp_path: Path) -> None:
    driver = Driver([INTERRUPT, *([None] * 30), INTERRUPT, "x"], tick_s=0.05)
    monitor, _ = make_monitor(tmp_path, driver)
    monitor.run()
    assert monitor.buffer == "x", "1.5 s apart: two separate single presses"


def test_a_failing_command_is_reported_and_the_monitor_keeps_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from spiyweb import commands

    def explode(_monitor: Monitor, _invocation: object) -> None:
        raise OSError("disk on fire")

    broken = commands.Command("boom", "/boom", "fails", explode)
    monkeypatch.setattr(commands, "COMMANDS", (*commands.COMMANDS, broken))
    driver = Driver([*"/boom", "enter", *"ok"])
    monitor, _ = make_monitor(tmp_path, driver)
    monitor.run()
    assert "OSError: disk on fire" in "\n".join(monitor.transcript)
    assert monitor.buffer == "ok", "still typing after the failure"


def test_an_exit_raised_inside_a_command_does_not_end_the_monitor(
    tmp_path: Path,
) -> None:
    driver = Driver([None])
    monitor, _ = make_monitor(tmp_path, driver)
    monitor.ask("name?", lambda _answer: sys.exit(3))
    monitor.handle("enter", 0.0)
    assert monitor.prompt is None
    assert "SystemExit: 3" in "\n".join(monitor.transcript)


def test_a_job_gets_its_own_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    seen: dict[str, object] = {}

    class FakePopen:
        pid = 1
        stdout = io.StringIO("")

        def __init__(self, command: object, **options: object) -> None:
            seen.update(options)

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    from spiyweb.watch import Job

    Job.start(1, "echo hi", tmp_path)
    if sys.platform == "win32":
        assert seen["creationflags"] == subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert seen["start_new_session"] is True


def test_a_spawned_verb_runs_without_a_shell_and_unbuffered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    seen: list[tuple[object, dict[str, object]]] = []

    class FakePopen:
        pid = 1
        stdout = io.StringIO("")

        def __init__(self, command: object, **options: object) -> None:
            seen.append((command, options))

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    driver = Driver([None])
    monitor, _ = make_monitor(tmp_path, driver)
    monitor.spawn_spiyweb(["index", "my docs", "out"])
    command, options = seen[0]
    assert command == [sys.executable, "-m", "spiyweb", "index", "my docs", "out"]
    assert options["shell"] is False, "a path with spaces stays one argument"
    environment = options["env"]
    assert isinstance(environment, dict)
    assert environment["PYTHONUNBUFFERED"] == "1"
    assert 'spiyweb index "my docs" out' in "\n".join(monitor.transcript)


def test_python_dash_m_spiyweb_is_the_same_command(tmp_path: Path) -> None:
    import subprocess

    from spiyweb import __version__

    done = subprocess.run(
        [sys.executable, "-m", "spiyweb", "version", "--json"],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["version"] == __version__


class _EndedProcess:
    pid = 1

    def __init__(self, code: int = 0) -> None:
        self.code = code

    def wait(self, timeout: float | None = None) -> int:
        return self.code

    def poll(self) -> int:
        return self.code


def _fed_job(lines: list[str], **options: object) -> object:
    from spiyweb.watch import Job

    job = Job(number=1, command="spiyweb index docs out", process=_EndedProcess())  # type: ignore[arg-type]
    for name, value in options.items():
        setattr(job, name, value)
    for line in lines:
        job.lines.put(line)
    return job


def test_a_jobs_progress_shows_in_the_status_line_and_bars_stay_out(
    tmp_path: Path,
) -> None:
    from spiyweb.progress import IndexProgress

    driver = Driver([None])
    monitor, _ = make_monitor(tmp_path, driver)
    progress = IndexProgress(label="indexing docs", started=0.0)
    job = _fed_job(
        [
            "3 document(s), 12 unit(s) -> out",
            "embedding 12 passages ...",
            "Batches:  45%|####5     | 9/20",
        ],
        progress=progress,
    )
    monitor.jobs.append(job)  # type: ignore[arg-type]
    monitor.pump_jobs(5.0)
    shown = "\n".join(monitor.transcript)
    assert "embedding 12 passages" in shown
    assert "Batches" not in shown, "a progress bar never reaches the transcript"
    status = monitor.status_line(5.0)
    assert "indexing docs" in status and "stage 2/4 embed" in status
    assert "0:05" in status and "45%" in status and "/kill 1" in status


def test_a_long_job_rings_the_bell_when_it_ends_a_short_one_does_not(
    tmp_path: Path,
) -> None:
    from spiyweb.progress import JobProgress
    from spiyweb.watch import BELL

    for elapsed, rings in ((3.0, False), (30.0, True)):
        driver = Driver([None])
        monitor, out = make_monitor(tmp_path, driver)
        finished: list[object] = []
        job = _fed_job(
            [],
            progress=JobProgress(label="x", started=0.0),
            on_done=finished.append,
        )
        job.lines.put(None)  # type: ignore[attr-defined]
        monitor.jobs.append(job)  # type: ignore[arg-type]
        monitor.pump_jobs(elapsed)
        assert (BELL in out.getvalue()) is rings, elapsed
        assert finished == [job], "on_done ran once the process ended"


def test_a_failing_on_done_is_reported_not_raised(tmp_path: Path) -> None:
    from spiyweb.progress import JobProgress

    def broken(_job: object) -> None:
        raise RuntimeError("could not read the new index")

    driver = Driver([None])
    monitor, _ = make_monitor(tmp_path, driver)
    job = _fed_job([], progress=JobProgress(label="x", started=0.0), on_done=broken)
    job.lines.put(None)  # type: ignore[attr-defined]
    monitor.jobs.append(job)  # type: ignore[arg-type]
    monitor.pump_jobs(1.0)
    assert "RuntimeError: could not read" in "\n".join(monitor.transcript)


def test_a_cli_install_hint_in_a_jobs_output_becomes_a_slash_command(
    tmp_path: Path,
) -> None:
    from spiyweb.progress import JobProgress

    driver = Driver([None])
    monitor, _ = make_monitor(tmp_path, driver)
    job = _fed_job(
        ['spiyweb: No module named faiss - pip install "spiyweb[index]"'],
        progress=JobProgress(label="x", started=0.0),
    )
    monitor.jobs.append(job)  # type: ignore[arg-type]
    monitor.pump_jobs(1.0)
    assert "/install index" in "\n".join(monitor.transcript)


def test_a_key_during_the_animation_skips_it_but_a_slash_starts_a_command(
    tmp_path: Path,
) -> None:
    driver = Driver([None, None, None, "x", None, None, None])
    monitor, _ = make_monitor(tmp_path, driver, hop_delay_ms=10_000)
    driver.on_tick[1] = lambda: _append(
        tmp_path / ".spiyweb" / TRACE_FILENAME, canonical_record()
    )
    monitor.run()
    play = monitor.playing
    assert play is not None and play.stopped and play.hop == 2
    assert monitor.buffer == ""

    driver2 = Driver([None, None, None, "/", None])
    monitor2, _ = make_monitor(tmp_path, driver2, hop_delay_ms=10_000)
    driver2.on_tick[1] = lambda: _append(
        tmp_path / ".spiyweb" / TRACE_FILENAME, canonical_record()
    )
    monitor2.run()
    assert monitor2.buffer == "/"
    play2 = monitor2.playing
    assert play2 is not None and not play2.stopped


def test_the_screen_fills_the_terminal_exactly(tmp_path: Path) -> None:
    driver = Driver([None])
    for size in ((80, 24), (120, 30), (160, 50)):
        monitor, _ = make_monitor(tmp_path, driver, size=size)
        lines = monitor.screen(0.0)
        assert len(lines) == size[1] - 1
        from spiyweb.terminal import printed_width

        assert {printed_width(line) for line in lines} == {size[0]}


def test_a_resize_forces_a_clean_redraw(tmp_path: Path) -> None:
    current = [(120, 30)]
    driver = Driver([None, None, None])
    out = io.StringIO()
    monitor = Monitor(
        config=WatchConfig(),
        directory=tmp_path / ".spiyweb",
        write=out.write,
        poll=driver.poll,
        clock=driver.clock,
        size=lambda: current[0],
        color=False,
        unicode=True,
        cwd=tmp_path,
    )
    driver.monitor = monitor
    monitor.draw(0.0)
    before = out.getvalue().count("\x1b[3J")
    monitor.draw(0.1)
    assert out.getvalue().count("\x1b[3J") == before, "same size: no wipe"
    current[0] = (100, 28)
    monitor.draw(0.2)
    assert out.getvalue().count("\x1b[3J") == before + 1, "new size: wiped"


def test_no_terminal_means_usage_and_exit_two(monkeypatch: pytest.MonkeyPatch) -> None:
    from spiyweb.watch import interactive

    monkeypatch.setattr("spiyweb.wizard.is_interactive", lambda: False)
    assert interactive() == 2


def test_without_raw_input_the_numbered_menu_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spiyweb.watch import interactive

    monkeypatch.setattr("spiyweb.wizard.is_interactive", lambda: True)
    monkeypatch.setattr("spiyweb.wizard.supports_raw_input", lambda: False)
    monkeypatch.setattr("spiyweb.wizard.run_wizard", lambda: 7)
    assert interactive() == 7


def test_a_bare_invocation_opens_the_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    from spiyweb.cli import main

    monkeypatch.setattr("spiyweb.wizard.is_interactive", lambda: True)
    monkeypatch.setattr("spiyweb.wizard.supports_raw_input", lambda: True)
    monkeypatch.setattr("spiyweb.watch.supports_screen", lambda: True)
    monkeypatch.setattr("spiyweb.watch.run_monitor", lambda config: 9)
    assert main([]) == 9
    assert main(["watch"]) == 9


def test_tab_completes_a_command_and_cycles_through_matches(tmp_path: Path) -> None:
    driver = Driver([None])
    monitor, _ = make_monitor(tmp_path, driver)
    monitor.buffer = "/fi"
    monitor.handle_key("tab", 0.0)
    assert monitor.buffer == "/find "
    monitor.buffer = "/c"
    assert [u for u, _ in monitor.suggestions()] == [
        "/compare <question>",
        "/config",
        "/clear",
    ]
    monitor.handle_key("tab", 0.0)
    assert monitor.buffer == "/compare"
    monitor.handle_key("tab", 0.0)
    assert monitor.buffer == "/config"
    monitor.handle_key("tab", 0.0)
    assert monitor.buffer == "/clear"
    monitor.buffer = "/zz"
    monitor.handle_key("tab", 0.0)
    assert monitor.buffer == "/zz"


def test_suggestions_show_under_the_prompt_while_a_command_is_typed(
    tmp_path: Path,
) -> None:
    driver = Driver([None])
    monitor, _ = make_monitor(tmp_path, driver)
    monitor.buffer = "/re"
    assert any("/replay" in line for line in monitor.screen(0.0))
    monitor.buffer = "/replay 2"
    assert monitor.suggestions() == []


def test_question_mark_ctrl_u_and_ctrl_l_are_shortcuts(tmp_path: Path) -> None:
    driver = Driver([None])
    monitor, _ = make_monitor(tmp_path, driver)
    monitor.handle_key("?", 0.0)
    assert "shortcuts" in "\n".join(monitor.transcript)
    monitor.buffer = "/half"
    monitor.handle_key("\x15", 0.0)
    assert monitor.buffer == ""
    monitor.handle_key("\x0c", 0.0)
    assert monitor.transcript == []


def test_the_caret_follows_the_terminal_focus(tmp_path: Path) -> None:
    driver = Driver([None])
    monitor, _ = make_monitor(tmp_path, driver)
    assert any("▏" in line for line in monitor.input_box(80, True))
    monitor.handle_key("focus_out", 0.0)
    assert not any("▏" in line for line in monitor.input_box(80, True))
    monitor.handle_key("focus_in", 0.0)
    assert any("▏" in line for line in monitor.input_box(80, True))
    assert monitor.buffer == "", "focus reports never type"


def test_a_bang_runs_a_shell_command_and_streams_its_output(tmp_path: Path) -> None:
    import sys

    command = f'{sys.executable} -c "print(1 + 1); print(3)"'
    driver = Driver([*("!" + command), "enter", *([None] * 40)], tick_s=0.05)
    monitor, _ = make_monitor(tmp_path, driver)
    real_poll = driver.poll

    def slow_poll(timeout: float) -> str | None:
        import time as _time

        _time.sleep(0.05)
        return real_poll(timeout)

    monitor.poll = slow_poll
    monitor.run()
    joined = "\n".join(monitor.transcript)
    assert "job 1 started" in joined
    assert " 2" in joined and " 3" in joined, joined
    assert "job 1 ended" in joined and "exit 0" in joined
    assert monitor.jobs[0].done


def test_jobs_and_kill_manage_what_bang_started(tmp_path: Path) -> None:
    import sys

    driver = Driver([None])
    monitor, _ = make_monitor(tmp_path, driver)
    monitor.start_job(f'{sys.executable} -c "import time; time.sleep(30)"')
    from spiyweb.commands import dispatch

    dispatch(monitor, "/jobs")
    assert "running" in "\n".join(monitor.transcript)
    dispatch(monitor, "/kill 1")
    assert monitor.jobs[0].process.poll() is not None
    assert "job 1 stopped" in "\n".join(monitor.transcript)
    dispatch(monitor, "/kill")
    assert "nothing is running" in "\n".join(monitor.transcript)


def test_an_empty_bang_is_refused(tmp_path: Path) -> None:
    driver = Driver([None])
    monitor, _ = make_monitor(tmp_path, driver)
    monitor.start_job("")
    assert "nothing to run" in "\n".join(monitor.transcript)
    assert monitor.jobs == []


def test_the_cursor_moves_inside_the_line_and_edits_there(tmp_path: Path) -> None:
    driver = Driver([None])
    monitor, _ = make_monitor(tmp_path, driver)
    for key in "/qery":
        monitor.handle_key(key, 0.0)
    for _ in range(3):
        monitor.handle_key("left", 0.0)
    monitor.handle_key("u", 0.0)
    assert monitor.buffer == "/query" and monitor.cursor == 3
    monitor.handle_key("home", 0.0)
    assert monitor.cursor == 0
    monitor.handle_key("delete", 0.0)
    assert monitor.buffer == "query"
    monitor.handle_key("end", 0.0)
    monitor.handle_key("backspace", 0.0)
    assert monitor.buffer == "quer" and monitor.cursor == 4
    monitor.handle_key("left", 0.0)
    monitor.handle_key("left", 0.0)
    monitor.handle_key("\x01", 0.0)
    assert monitor.cursor == 0
    monitor.handle_key("\x05", 0.0)
    assert monitor.cursor == 4
    monitor.buffer = "reset"
    assert monitor.cursor == 5, "setting the line moves the cursor to its end"


def test_the_caret_is_drawn_where_the_cursor_is(tmp_path: Path) -> None:
    driver = Driver([None])
    monitor, _ = make_monitor(tmp_path, driver)
    monitor.color = True
    monitor.settings.color = True
    monitor.apply_settings()
    monitor.buffer = "abc"
    monitor.handle_key("left", 0.0)
    line = monitor.input_box(60, True)[1]
    assert "\x1b[7mc\x1b[0m" in line, "the character under the cursor is reversed"
    monitor.handle_key("end", 0.0)
    line = monitor.input_box(60, True)[1]
    assert "abc" in line and "▏" in line


def test_a_bang_python_is_the_monitors_python() -> None:
    import sys

    from spiyweb.watch import same_python

    assert same_python("python app.py") == f'"{sys.executable}" app.py'
    assert same_python("PY -m mod") == f'"{sys.executable}" -m mod'
    assert same_python("python") == f'"{sys.executable}"'
    assert same_python("npm start") == "npm start"


def test_the_wheel_scrolls_the_transcript_and_typing_snaps_back(tmp_path: Path) -> None:
    driver = Driver([None])
    monitor, _ = make_monitor(tmp_path, driver, size=(100, 24))
    monitor.transcript = [f"line {n}" for n in range(60)]
    bottom = monitor.screen(0.0)
    assert any("line 59" in row for row in bottom)
    monitor.handle_key("mouse:64:5:5:press", 0.0)
    monitor.handle_key("mouse:64:5:5:press", 0.0)
    scrolled = monitor.screen(0.0)
    assert not any("line 59" in row for row in scrolled)
    assert any("line 53" in row for row in scrolled)
    monitor.handle_key("mouse:65:5:5:press", 0.0)
    assert monitor.scroll == 3
    monitor.handle_key("/", 0.0)
    monitor.handle_key("enter", 0.0)
    assert monitor.scroll == 0


def test_a_click_on_a_picker_option_chooses_it(tmp_path: Path) -> None:
    driver = Driver([None])
    monitor, _ = make_monitor(tmp_path, driver, size=(100, 30))
    chosen: list[str] = []
    monitor.pick(
        "which?", [("a", "first"), ("b", "second"), ("c", "third")], chosen.append
    )
    lines = monitor.screen(0.0)
    row_of_second = next(i for i, line in enumerate(lines) if "second" in line) + 1
    assert monitor.hits[row_of_second] == ("pick", "b")
    monitor.handle_key(f"mouse:0:10:{row_of_second}:press", 0.0)
    assert chosen == ["b"] and monitor.picker is None


def test_a_click_on_a_suggestion_completes_the_command(tmp_path: Path) -> None:
    driver = Driver([None])
    monitor, _ = make_monitor(tmp_path, driver, size=(100, 30))
    monitor.buffer = "/re"
    lines = monitor.screen(0.0)
    row = (
        next(
            i
            for i, line in enumerate(lines)
            if "/replay" in line and "play the last" in line
        )
        + 1
    )
    assert monitor.hits[row] == ("suggest", "/replay")
    monitor.handle_key(f"mouse:0:4:{row}:press", 0.0)
    assert monitor.buffer == "/replay "
    monitor.handle_key("mouse:0:4:2:press", 0.0)  # a click on nothing changes nothing
    assert monitor.buffer == "/replay "


# --- the first-run hint, the status bar, history, path completion ------------


def _border(monitor: Monitor, width: int = 120) -> str:
    return monitor.input_box(width, True)[0]


def test_the_input_box_says_how_to_start_until_the_first_message(
    tmp_path: Path,
) -> None:
    from spiyweb.terminal import printed_width

    driver = Driver([None])
    monitor, _ = make_monitor(tmp_path, driver)
    border = _border(monitor)
    assert "start: /demo or /index <folder>" in border
    assert printed_width(border) == 120, "the label does not widen the box"
    monitor.buffer = "/help"
    monitor.handle_key("enter", 0.0)
    assert "start:" not in _border(monitor), "gone after the first message"
    assert "start:" not in _border(monitor), "and stays gone"
    fresh, _ = make_monitor(tmp_path, Driver([None]))
    assert "start:" in _border(fresh), "a new session shows it again"


def test_the_hint_asks_for_a_question_once_an_index_is_near(tmp_path: Path) -> None:
    (tmp_path / "idx").mkdir()
    (tmp_path / "idx" / "nodes.json").write_text("[]", encoding="utf-8")
    monitor, _ = make_monitor(tmp_path, Driver([None]))
    assert "type a question, Enter" in _border(monitor)


def test_the_hint_keeps_quiet_while_a_list_is_open(tmp_path: Path) -> None:
    monitor, _ = make_monitor(tmp_path, Driver([None]))
    monitor.pick("which?", [("a", "first")], lambda _v: None)
    assert "start:" not in _border(monitor)


def test_the_hint_is_clipped_on_a_narrow_box_and_ascii_in_ascii(
    tmp_path: Path,
) -> None:
    from spiyweb.terminal import printed_width

    monitor, _ = make_monitor(tmp_path, Driver([None]))
    narrow = _border(monitor, 30)
    assert printed_width(narrow) == 30 and "…" in narrow
    assert "start" not in _border(monitor, 16), "no room: no label"
    monitor.unicode = False
    monitor.apply_settings()
    assert all(ord(char) < 128 for char in _border(monitor, 30))


def test_the_status_bar_names_the_active_index_and_the_profile(
    tmp_path: Path,
) -> None:
    from spiyweb.terminal import printed_width

    (tmp_path / "my-index").mkdir()
    (tmp_path / "my-index" / "nodes.json").write_text("[]", encoding="utf-8")
    monitor, _ = make_monitor(tmp_path, Driver([None]))
    assert "no index" in monitor.status_bar(120)
    monitor.set_active(tmp_path / "my-index")
    wide = monitor.status_bar(120)
    assert "my-index" in wide and "explore" in wide and "0 played" in wide
    narrow = monitor.status_bar(70)
    assert printed_width(narrow) <= 70
    assert "my-index" in narrow and "played" not in narrow, "least useful goes"
    again, _ = make_monitor(tmp_path, Driver([None]))
    assert again.active_index == str(tmp_path / "my-index"), "remembered"


def test_a_vanished_active_index_is_forgotten_on_start(tmp_path: Path) -> None:
    monitor, _ = make_monitor(tmp_path, Driver([None]))
    monitor.set_active(tmp_path / "gone")
    again, _ = make_monitor(tmp_path, Driver([None]))
    assert again.active_index is None


def test_history_survives_a_restart_but_bang_lines_are_never_written(
    tmp_path: Path,
) -> None:
    monitor, _ = make_monitor(tmp_path, Driver([None]))
    for line in ("/help", "! API_KEY=secret python app.py", "/jobs"):
        monitor.buffer = line
        monitor._submit()
    written = (tmp_path / ".spiyweb" / "history").read_text(encoding="utf-8")
    assert "secret" not in written
    again, _ = make_monitor(tmp_path, Driver([None]))
    assert again.history == ["/help", "/jobs"]
    again.handle_key("up", 0.0)
    assert again.buffer == "/jobs"


def test_history_max_zero_writes_nothing(tmp_path: Path) -> None:
    monitor, _ = make_monitor(tmp_path, Driver([None]), history_max=0)
    monitor.buffer = "/help"
    monitor._submit()
    assert not (tmp_path / ".spiyweb" / "history").exists()


def test_tab_after_a_command_completes_a_folder(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "my notes").mkdir()
    monitor, _ = make_monitor(tmp_path, Driver([None]))
    monitor.buffer = "/index do"
    monitor.handle_key("tab", 0.0)
    assert monitor.buffer == "/index docs/"
    monitor.buffer = "/index d"
    monitor.handle_key("tab", 0.0)
    assert monitor.buffer == "/index data/"
    monitor.handle_key("tab", 0.0)
    assert monitor.buffer == "/index docs/", "again: the next match"
    monitor.buffer = "/index my"
    monitor.handle_key("tab", 0.0)
    assert monitor.buffer == '/index "my notes/"'


def test_an_apps_query_gets_numbered_passages_and_its_findings(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    from spiyweb.trace import TraceEvent

    record = canonical_record(
        events=(TraceEvent(kind="conflict", node="A", other="C", amount=0.2, hop=1),)
    )
    record = replace(
        record, nodes=tuple(replace(n, text=f"text of {n.id}") for n in record.nodes)
    )
    driver = Driver([None] * 60)
    monitor, _ = make_monitor(tmp_path, driver)
    driver.on_tick[2] = lambda: _append(tmp_path / ".spiyweb" / TRACE_FILENAME, record)
    monitor.run()
    shown = "\n".join(monitor.transcript)
    assert "top passages" in shown and "/show <n> opens one" in shown
    assert " 1  5.62 A  text of A" in shown
    assert "A and C contradict each other" in shown


def test_tab_after_a_single_match_goes_on_into_the_folder(tmp_path: Path) -> None:
    (tmp_path / "only" / "inner").mkdir(parents=True)
    monitor, _ = make_monitor(tmp_path, Driver([None]))
    monitor.buffer = "/index o"
    monitor.handle_key("tab", 0.0)
    assert monitor.buffer == "/index only/"
    monitor.handle_key("tab", 0.0)
    assert monitor.buffer == "/index only/inner/"


def test_a_job_whose_stream_ends_mid_interrupt_still_finishes(tmp_path: Path) -> None:
    from spiyweb.progress import JobProgress

    class SlowWait(_EndedProcess):
        calls = 0

        def wait(self, timeout: float | None = None) -> int:
            SlowWait.calls += 1
            if SlowWait.calls == 1:
                raise KeyboardInterrupt
            return 0

    from spiyweb.watch import Job

    finished: list[object] = []
    job = Job(number=1, command="x", process=SlowWait())  # type: ignore[arg-type]
    job.progress, job.on_done = JobProgress(label="x", started=0.0), finished.append
    job.lines.put(None)
    monitor, _ = make_monitor(tmp_path, Driver([None]))
    monitor.jobs.append(job)
    with pytest.raises(KeyboardInterrupt):
        monitor.pump_jobs(1.0)
    monitor.pump_jobs(1.0)
    assert job.done and job.exit_code == 0 and finished == [job]


def test_kill_stops_a_job_whose_leader_left_but_whose_output_is_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from spiyweb import watch
    from spiyweb.commands import dispatch

    stopped: list[object] = []
    monkeypatch.setattr(watch, "stop_group", lambda process: stopped.append(process))
    job = _fed_job([])  # leader exited (poll() is 0), stream still open
    monitor, _ = make_monitor(tmp_path, Driver([None]))
    monitor.jobs.append(job)  # type: ignore[arg-type]
    dispatch(monitor, "/kill")
    assert stopped and "job 1 stopped" in "\n".join(monitor.transcript)
