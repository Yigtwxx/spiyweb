"""The `/` commands, run against a monitor that never touches a terminal."""

from __future__ import annotations

import io
import sys
from typing import TYPE_CHECKING

import pytest

from conftest import canonical_record
from spiyweb.commands import (
    COMMANDS,
    dispatch,
    find_usages,
    parse,
    pip_command,
    problem_hint,
)
from spiyweb.config import WatchConfig
from spiyweb.watch import Monitor

if TYPE_CHECKING:
    from pathlib import Path


def quiet_monitor(tmp_path: Path) -> Monitor:
    return Monitor(
        config=WatchConfig(),
        directory=tmp_path / ".spiyweb",
        write=io.StringIO().write,
        poll=lambda _t: None,
        clock=lambda: 0.0,
        size=lambda: (120, 30),
        color=False,
        unicode=True,
        cwd=tmp_path,
    )


def transcript(monitor: Monitor) -> str:
    return "\n".join(monitor.transcript)


# --- parsing --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "name", "args"),
    [
        ("/query idx what now", "query", ("idx", "what", "now")),
        ("  /HELP  ", "help", ()),
        ("/", "help", ()),
        ("/replay 3", "replay", ("3",)),
    ],
)
def test_parse_splits_name_and_arguments(
    line: str, name: str, args: tuple[str, ...]
) -> None:
    invocation = parse(line)
    assert invocation is not None
    assert (invocation.name, invocation.args) == (name, args)


def test_text_without_a_slash_is_a_question_not_a_command() -> None:
    assert parse("who built the tower") is None


def test_every_command_appears_in_help(tmp_path: Path) -> None:
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/help")
    shown = transcript(monitor)
    for command in COMMANDS:
        assert command.usage in shown, command.usage


def test_the_old_menu_items_are_all_commands() -> None:
    names = {command.name for command in COMMANDS}
    assert {"query", "lint", "index", "install", "config", "menu"} <= names


# --- /find ------------------------------------------------------------------------


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_find_lists_imports_and_suggests_the_opened_index(tmp_path: Path) -> None:
    _write(
        tmp_path / "app" / "main.py",
        "import os\nfrom spiyweb import open_index\n\n"
        "index = open_index(\"my-index\")\nother = SpiywebIndex.open('elsewhere')\n",
    )
    _write(tmp_path / ".venv" / "lib" / "x.py", "import spiyweb\n")
    _write(tmp_path / "node_modules" / "y.py", "import spiyweb\n")
    _write(tmp_path / "a/b/c/d/e/f/g/deep.py", "import spiyweb\n")
    _write(
        tmp_path / "vendor" / "spiyweb" / "__init__.py", "from spiyweb.core import x\n"
    )
    _write(tmp_path / "vendor" / "spiyweb" / "core" / "propagate.py", "")
    (tmp_path / "my-index").mkdir()
    (tmp_path / "my-index" / "nodes.json").write_text("[]", encoding="utf-8")

    report = find_usages(tmp_path, max_depth=6, max_files=100, max_file_bytes=10_000)
    assert [(u.path.name, u.line) for u in report.usages] == [("main.py", 2)]
    assert report.index_hints == ("my-index", "elsewhere")

    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/find")
    shown = transcript(monitor)
    assert "main.py" in shown and "1 import, first at line 2" in shown
    assert "main.py:2" not in shown, "one row per file, not per line"
    assert monitor.suggested_index is not None
    assert monitor.suggested_index.endswith("my-index")
    assert "elsewhere" in shown and "not found" in shown


def test_find_stops_at_the_file_cap_and_says_so(tmp_path: Path) -> None:
    for number in range(5):
        _write(tmp_path / f"f{number}.py", "import spiyweb\n")
    report = find_usages(tmp_path, max_depth=2, max_files=3, max_file_bytes=10_000)
    assert report.truncated and report.scanned == 3


def test_find_with_nothing_to_find_is_a_warning(tmp_path: Path) -> None:
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/find")
    assert "nothing imports spiyweb" in transcript(monitor)


# --- /install ---------------------------------------------------------------------


def test_pip_command_prefers_pip_then_uv_then_gives_up() -> None:
    assert pip_command("index", python="py", has_pip=True) == [
        "py", "-m", "pip", "install", "spiyweb[index]"
    ]  # fmt: skip
    assert pip_command("view", python="py", has_pip=False, which=lambda _: "/uv") == [
        "/uv", "pip", "install", "--python", "py", "spiyweb[view]"
    ]  # fmt: skip
    assert pip_command("view", python="py", has_pip=False, which=lambda _: None) is None


def test_install_without_an_argument_lists_the_extras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("spiyweb.cli._installed", lambda modules: False)
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/install")
    shown = transcript(monitor)
    assert "[store]" in shown and "not installed" in shown


def test_install_refuses_an_unknown_extra_without_running_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran: list[list[str]] = []
    monkeypatch.setattr("spiyweb.commands._run_install", lambda cmd: ran.append(cmd))
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/install evil;rm")
    assert "no extra called" in transcript(monitor)
    assert ran == []


def test_install_asks_first_and_runs_only_on_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from spiyweb.commands import InstallResult

    ran: list[list[str]] = []

    def fake_run(command: list[str]) -> InstallResult:
        ran.append(command)
        return InstallResult(code=0, output="Successfully installed spiyweb\n")

    monkeypatch.setattr("spiyweb.commands._run_install", fake_run)
    monkeypatch.setattr("spiyweb.commands.pip_command", lambda extra: ["pip", extra])
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/install view")
    assert monitor.prompt is not None and "(y/n)" in monitor.prompt.question
    monitor.buffer = "n"
    monitor._submit()
    assert ran == [] and "not installed" in transcript(monitor)

    dispatch(monitor, "/install view")
    monitor.buffer = "y"
    monitor._submit()
    assert ran == [["pip", "view"]]
    assert "Successfully installed" in transcript(monitor)


def test_a_cli_problem_about_an_extra_becomes_an_install_hint() -> None:
    message = (
        'No module named faiss\nopening an index needs: pip install "spiyweb[index]"'
    )
    assert problem_hint(message).endswith("/install index")
    assert problem_hint("plain trouble") == "plain trouble"


# --- /query, /lint, /replay -------------------------------------------------------


def test_query_with_no_index_known_offers_a_choice_or_asks(tmp_path: Path) -> None:
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/query what")
    assert monitor.prompt is not None, "no index nearby: a path is asked for"
    assert "path to an index" in monitor.prompt.question


def test_a_plain_question_needs_an_index_first(tmp_path: Path) -> None:
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "who built the tower")
    assert "no index chosen yet" in transcript(monitor)


def test_query_plays_its_own_record_instead_of_writing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakePassage:
        node_id, energy, text = "A", 5.6, "Tesla built Wardenclyffe on Long Island."

    class FakeAnswer:
        trace = canonical_record()
        passages = (FakePassage(),)

    class FakeIndex:
        def retrieve(self, question: str, *, profile: str) -> FakeAnswer:
            assert profile == "explore"
            return FakeAnswer()

    opened: list[object] = []

    def fake_open(path: str, **options: object) -> FakeIndex:
        opened.append(options["trace"])
        return FakeIndex()

    monkeypatch.setattr("spiyweb.cli._open", fake_open)
    monkeypatch.setattr("spiyweb.cli._is_index", lambda path: str(path) == "idx")
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/query idx who built the tower")
    assert len(monitor.queue) == 1, "queued to play, not written to the file"
    assert opened[0].attach_dir is None, "never through the marker, or it replays"  # type: ignore[attr-defined]
    record, after = monitor.queue[0]
    assert record.query == "which tower did tesla build"
    assert any("Wardenclyffe" in line for line in after)
    assert monitor.suggested_index == "idx"
    assert not (tmp_path / ".spiyweb" / "traces.jsonl").exists()


def test_replay_plays_the_last_n_recorded(tmp_path: Path) -> None:
    from spiyweb.config import TraceConfig
    from spiyweb.trace import TraceStore

    store = TraceStore(TraceConfig(directory=tmp_path / ".spiyweb"))
    for number in range(3):
        store.append(canonical_record(sequence=number))
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/replay 2")
    assert [r.sequence for r, _ in monitor.queue] == [1, 2]


def test_lint_hands_the_verb_to_the_cli_and_captures_its_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_main(argv: list[str]) -> int:
        sys.stdout.write("islands: none\n")
        return 0

    monkeypatch.setattr("spiyweb.cli.main", fake_main)
    monkeypatch.setattr("spiyweb.cli._is_index", lambda path: str(path) == "idx")
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/lint idx")
    shown = transcript(monitor)
    assert "spiyweb lint idx" in shown and "islands: none" in shown


# --- /config ------------------------------------------------------------------------


def test_config_is_an_arrow_key_list_that_cycles_and_persists(tmp_path: Path) -> None:
    monitor = quiet_monitor(tmp_path)
    dispatch(monitor, "/config")
    picker = monitor.picker
    assert picker is not None and picker.sticky
    assert picker.options[0][0] == "profile"
    monitor.handle_key("enter", 0.0)  # profile: explore -> precise
    assert monitor.settings.profile == "precise"
    monitor.handle_key("down", 0.0)
    monitor.handle_key(" ", 0.0)  # map: on -> off
    assert monitor.settings.map_enabled is False
    assert monitor.config.map_enabled is False, "the live config follows"
    monitor.handle_key("escape", 0.0)
    assert monitor.picker is None
    reloaded = quiet_monitor(tmp_path)
    assert reloaded.settings.profile == "precise"
    assert reloaded.settings.map_enabled is False


def test_a_picker_takes_the_arrows_and_the_prompt_takes_the_text(
    tmp_path: Path,
) -> None:
    monitor = quiet_monitor(tmp_path)
    chosen: list[str] = []
    monitor.pick("which?", [("a", "first"), ("b", "second")], chosen.append)
    for key in ("down", "enter"):
        monitor.handle_key(key, 0.0)
    assert chosen == ["b"] and monitor.picker is None
    answers: list[str] = []
    monitor.ask("name?", answers.append, default="anon")
    monitor.handle_key("enter", 0.0)
    assert answers == ["anon"]
