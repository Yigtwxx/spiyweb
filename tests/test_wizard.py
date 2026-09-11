"""The guided menu: one word to type, then questions.

The rule that matters most here is the one about NOT asking. A prompt with
nobody at it is a hung build, so a bare `spiyweb` in a pipe, a script or a CI
job must print usage and leave - and that is asserted before anything about
the menu itself.

Everything else is a translation test: each answer becomes an argv that the
ordinary parser accepts. The wizard runs nothing of its own, so if the argv
is right the behaviour is whatever the subcommand already does, and that has
its own tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spiyweb.wizard import (
    ask_text,
    choose,
    discover,
    interactive,
    is_interactive,
    run_wizard,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

OPTIONS = [("a", "first thing"), ("b", "second thing")]


@pytest.fixture
def answers(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Queue the keystrokes a person would type."""
    queued: list[str] = []

    def fake_input(prompt: str = "") -> str:
        if not queued:
            raise EOFError
        return queued.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)
    yield queued


# --- the rule that is not negotiable ---------------------------------------


def test_a_pipe_gets_usage_and_a_non_zero_exit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A prompt with nobody at it is a hung build, not a friendly one."""
    monkeypatch.setattr("spiyweb.wizard.is_interactive", lambda: False)
    assert interactive() == 2
    captured = capsys.readouterr()
    assert "usage:" in captured.out
    assert "only runs in a terminal" in captured.err


def test_bare_invocation_never_blocks_without_a_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The end-to-end version: `spiyweb` with no arguments, in a script."""
    from spiyweb.cli import main

    monkeypatch.setattr("spiyweb.wizard.is_interactive", lambda: False)
    assert main([]) == 2


def test_both_ends_must_be_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    class Stream:
        def __init__(self, tty: bool) -> None:
            self._tty = tty

        def isatty(self) -> bool:
            return self._tty

    monkeypatch.setattr("sys.stdin", Stream(True))
    monkeypatch.setattr("sys.stdout", Stream(False))
    assert is_interactive() is False

    monkeypatch.setattr("sys.stdout", Stream(True))
    assert is_interactive() is True


def test_a_stream_that_cannot_answer_is_not_a_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Hostile:
        def isatty(self) -> bool:
            raise ValueError("closed")

    monkeypatch.setattr("sys.stdin", Hostile())
    assert is_interactive() is False


# --- asking -----------------------------------------------------------------


def test_a_number_picks_that_option(answers: list[str]) -> None:
    answers.append("2")
    assert choose("pick", OPTIONS) == "b"


def test_enter_takes_the_first_option(answers: list[str]) -> None:
    """The common case should cost one keystroke."""
    answers.append("")
    assert choose("pick", OPTIONS) == "a"


def test_q_quits(answers: list[str]) -> None:
    answers.append("q")
    assert choose("pick", OPTIONS) is None


def test_end_of_input_quits_rather_than_raising(answers: list[str]) -> None:
    """Ctrl-D should leave the terminal tidy, not print a traceback."""
    assert choose("pick", OPTIONS) is None


def test_a_bad_answer_asks_again(
    answers: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    answers.extend(["9", "banana", "1"])
    assert choose("pick", OPTIONS) == "a"
    assert "pick 1-2" in capsys.readouterr().out


def test_free_text_falls_back_to_its_default(answers: list[str]) -> None:
    answers.append("")
    assert ask_text("path?", default="docs") == "docs"


def test_free_text_can_be_quit(answers: list[str]) -> None:
    answers.append("quit")
    assert ask_text("path?") is None


# --- discovery --------------------------------------------------------------


def test_an_index_nearby_is_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Typing a path is the part people get wrong, so it is not required.

    Built here rather than pointed at the shared fixture: every module's
    `tmp_path_factory` directory is a sibling of every other one, so a scan
    from that parent finds a dozen unrelated indexes and the cap hides the
    one under test. A test whose result depends on what OTHER tests created
    is not a test.
    """
    import json

    index = tmp_path / "my-index"
    index.mkdir()
    (index / "nodes.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    indexes = discover()
    assert [found.path.name for found in indexes] == ["my-index"]
    assert indexes[0].detail == "3 atoms"


def test_a_directory_that_is_not_an_index_is_not_offered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "not-an-index").mkdir()
    monkeypatch.chdir(tmp_path)
    indexes = discover()
    assert indexes == []


# --- answers become a subcommand -------------------------------------------


def test_the_menu_builds_a_real_command(
    answers: list[str], monkeypatch: pytest.MonkeyPatch, tiny_index_root: Path
) -> None:
    """Every answer ends as argv for the parser everything else goes through."""
    seen: list[list[str]] = []
    monkeypatch.setattr("spiyweb.cli.main", lambda argv: seen.append(argv) or 0)
    monkeypatch.setattr(
        "spiyweb.wizard._pick_index", lambda question: str(tiny_index_root)
    )
    answers.extend(["1", "who raised the tower", "2"])  # query, text, precise
    assert run_wizard() == 0
    assert seen == [
        ["query", str(tiny_index_root), "who raised the tower", "--profile", "precise"]
    ]


def test_the_first_profile_option_adds_no_flag(
    answers: list[str], monkeypatch: pytest.MonkeyPatch, tiny_index_root: Path
) -> None:
    seen: list[list[str]] = []
    monkeypatch.setattr("spiyweb.cli.main", lambda argv: seen.append(argv) or 0)
    monkeypatch.setattr(
        "spiyweb.wizard._pick_index", lambda question: str(tiny_index_root)
    )
    answers.extend(["1", "a question", ""])
    assert run_wizard() == 0
    # No flag: `query` supplies `explore` itself, because the bare library
    # default cannot spread past the seed.
    assert seen == [["query", str(tiny_index_root), "a question"]]


def test_the_menu_shows_what_it_ran(
    answers: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """So the second time you type it yourself and the menu is unnecessary."""
    monkeypatch.setattr("spiyweb.cli.main", lambda argv: 0)
    answers.append("4")  # version
    assert run_wizard() == 0
    assert "running: spiyweb version" in capsys.readouterr().out


def test_quitting_the_first_question_runs_nothing(
    answers: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    ran: list[list[str]] = []
    monkeypatch.setattr("spiyweb.cli.main", lambda argv: ran.append(argv) or 0)
    answers.append("q")
    assert run_wizard() == 0
    assert ran == []


def test_an_empty_question_runs_nothing(
    answers: list[str], monkeypatch: pytest.MonkeyPatch, tiny_index_root: Path
) -> None:
    """An accidental Enter must not fire a query with no text in it."""
    ran: list[list[str]] = []
    monkeypatch.setattr("spiyweb.cli.main", lambda argv: ran.append(argv) or 0)
    monkeypatch.setattr(
        "spiyweb.wizard._pick_index", lambda question: str(tiny_index_root)
    )
    answers.extend(["1", ""])
    assert run_wizard() == 0
    assert ran == []


# --- the arrow path ---------------------------------------------------------


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Queue keypresses and force the arrow menu on."""
    queued: list[str] = []
    monkeypatch.setattr("spiyweb.wizard.supports_raw_input", lambda: True)
    monkeypatch.setattr(
        "spiyweb.wizard.read_key", lambda: queued.pop(0) if queued else "quit"
    )
    yield queued


def test_arrows_move_the_cursor_and_enter_chooses(keys: list[str]) -> None:
    keys.extend(["down", "enter"])
    assert choose("pick", OPTIONS) == "b"


def test_the_cursor_wraps_at_both_ends(keys: list[str]) -> None:
    """Up from the first option lands on the last, not on nothing."""
    keys.extend(["up", "enter"])
    assert choose("pick", OPTIONS) == "b"

    keys.extend(["down", "down", "enter"])
    assert choose("pick", OPTIONS) == "a"


def test_q_still_quits_the_arrow_menu(keys: list[str]) -> None:
    keys.append("quit")
    assert choose("pick", OPTIONS) is None


def test_numbers_still_work_in_the_arrow_menu(keys: list[str]) -> None:
    """The muscle memory of every other menu should not be punished."""
    keys.append("2")
    assert choose("pick", OPTIONS) == "b"


def test_an_unknown_key_is_ignored_rather_than_chosen(keys: list[str]) -> None:
    keys.extend(["z", "enter"])
    assert choose("pick", OPTIONS) == "a"


def test_ctrl_c_leaves_without_choosing(monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupt() -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("spiyweb.wizard.supports_raw_input", lambda: True)
    monkeypatch.setattr("spiyweb.wizard.read_key", interrupt)
    assert choose("pick", OPTIONS) is None


def test_the_cursor_is_always_restored(
    keys: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Exiting with the cursor hidden leaves the next shell looking broken."""
    from spiyweb.wizard import HIDE_CURSOR, SHOW_CURSOR

    keys.append("enter")
    choose("pick", OPTIONS)
    printed = capsys.readouterr().out
    assert printed.count(HIDE_CURSOR) == printed.count(SHOW_CURSOR) == 1
    assert printed.rindex(SHOW_CURSOR) > printed.rindex(HIDE_CURSOR)


def test_the_marker_changes_shape_not_only_colour() -> None:
    """On a terminal that refused colour a chromatic cursor is no cursor."""
    from spiyweb.wizard import FILLED, HOLLOW, _render

    lines = _render(OPTIONS, cursor=1, color=False)
    assert FILLED in lines[1] and HOLLOW in lines[0]
    assert FILLED not in lines[0]


def test_without_raw_input_the_numbered_menu_runs(
    answers: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pipe, a CI log and an editor pane all land here."""
    monkeypatch.setattr("spiyweb.wizard.supports_raw_input", lambda: False)
    answers.append("2")
    assert choose("pick", OPTIONS) == "b"


def test_raw_input_is_refused_without_a_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spiyweb.wizard import supports_raw_input

    monkeypatch.setattr("spiyweb.wizard.is_interactive", lambda: False)
    assert supports_raw_input() is False


@pytest.mark.parametrize(
    ("char", "expected"),
    [
        ("\r", "enter"),
        ("\n", "enter"),
        ("\x1b", "quit"),
        ("\x04", "quit"),
        ("q", "quit"),
        ("k", "up"),
        ("j", "down"),
        ("w", "up"),
        ("s", "down"),
        ("7", "7"),
    ],
)
def test_a_character_becomes_an_intent(char: str, expected: str) -> None:
    from spiyweb.wizard import _classify

    assert _classify(char) == expected


def test_ctrl_c_is_raised_rather_than_classified() -> None:
    from spiyweb.wizard import _classify

    with pytest.raises(KeyboardInterrupt):
        _classify("\x03")


@pytest.mark.parametrize(
    ("bytes_after_escape", "expected"),
    [
        ("", "quit"),  # a bare ESC: nothing follows, and nothing may block
        ("[A", "up"),
        ("[B", "down"),
        ("OA", "up"),  # application-mode arrows
        ("OB", "down"),
        ("[C", ""),  # right arrow: not an intent here
        ("x", ""),  # alt+x, or noise
    ],
)
def test_an_escape_sequence_is_decoded_without_blocking(
    bytes_after_escape: str, expected: str
) -> None:
    """The POSIX reader once did `read(2)` after ESC and froze on a bare ESC."""
    from spiyweb.wizard import _decode_escape

    queue = list(bytes_after_escape)

    def read(n: int) -> str:
        assert queue, "read past the end - this is the blocking read"
        return "".join(queue.pop(0) for _ in range(n))

    assert _decode_escape(pending=lambda: bool(queue), read=read) == expected
    assert not queue, "everything the terminal sent was consumed"
