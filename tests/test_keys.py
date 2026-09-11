"""Raw key reading with a timeout - the monitor's only input primitive."""

from __future__ import annotations

import pytest

from spiyweb.keys import (
    BACKSPACE,
    DOWN,
    ENTER,
    ESCAPE,
    LEFT,
    RIGHT,
    UP,
    _poll_windows,
    decode_escape,
    name_windows_key,
)


@pytest.mark.parametrize(
    ("burst", "expected"),
    [
        ("", ESCAPE),
        ("[A", UP),
        ("[B", DOWN),
        ("[C", RIGHT),
        ("[D", LEFT),
        ("OA", UP),
        ("x", ""),
        ("[Z", ""),
    ],
)
def test_decode_escape_names_every_arrow_and_a_bare_escape(
    burst: str, expected: str
) -> None:
    queue = list(burst)

    def read(count: int) -> str:
        assert count == 1 and queue, "read past what was pending"
        return queue.pop(0)

    assert decode_escape(pending=lambda: bool(queue), read=read) == expected


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ("\x00", "H", UP),
        ("\xe0", "P", DOWN),
        ("\xe0", "K", LEFT),
        ("\xe0", "M", RIGHT),
        ("\xe0", "Q", ""),
        ("\r", None, ENTER),
        ("\x08", None, BACKSPACE),
        ("\x1b", None, ESCAPE),
        ("a", None, "a"),
        ("/", None, "/"),
        ("q", None, "q"),
    ],
)
def test_windows_keys_are_named_and_letters_stay_letters(
    first: str, second: str | None, expected: str
) -> None:
    reads: list[None] = []

    def next_half() -> str:
        reads.append(None)
        assert second is not None, "read a second half nobody sent"
        return second

    assert name_windows_key(first, next_half) == expected
    assert len(reads) == (1 if second is not None else 0)


def test_ctrl_c_is_raised_not_returned() -> None:
    with pytest.raises(KeyboardInterrupt):
        name_windows_key("\x03", lambda: "")


def test_poll_returns_none_after_the_timeout_without_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([0.0, 0.01, 0.02, 0.03, 0.05])
    monkeypatch.setattr("spiyweb.keys.time.monotonic", lambda: next(ticks))
    slept: list[float] = []
    result = _poll_windows(
        0.04, kbhit=lambda: False, getwch=lambda: "?", sleep=slept.append
    )
    assert result is None
    assert slept, "waited by sleeping in slices, not by spinning"


def test_poll_returns_the_key_as_soon_as_one_is_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("spiyweb.keys.time.monotonic", lambda: 0.0)
    hits = iter([False, True])
    result = _poll_windows(
        1.0, kbhit=lambda: next(hits), getwch=lambda: "k", sleep=lambda _: None
    )
    assert result == "k"


def test_home_end_and_delete_are_named_on_both_platforms() -> None:
    from spiyweb.keys import DELETE, END_KEY, HOME_KEY

    assert name_windows_key("\xe0", lambda: "G") == HOME_KEY
    assert name_windows_key("\xe0", lambda: "O") == END_KEY
    assert name_windows_key("\xe0", lambda: "S") == DELETE
    burst = list("[3~")
    assert (
        decode_escape(pending=lambda: bool(burst), read=lambda _: burst.pop(0))
        == DELETE
    )
    burst = list("[H")
    assert (
        decode_escape(pending=lambda: bool(burst), read=lambda _: burst.pop(0))
        == HOME_KEY
    )


def test_a_mouse_report_is_decoded_and_parsed() -> None:
    from spiyweb.keys import is_mouse, parse_mouse

    burst = list("[<0;12;7M")
    key = decode_escape(pending=lambda: bool(burst), read=lambda _: burst.pop(0))
    assert is_mouse(key)
    assert parse_mouse(key) == (0, 12, 7, True)
    burst = list("[<64;1;1m")
    key = decode_escape(pending=lambda: bool(burst), read=lambda _: burst.pop(0))
    assert parse_mouse(key) == (64, 1, 1, False)
    burst = list("[<x;y;zM")
    assert decode_escape(pending=lambda: bool(burst), read=lambda _: burst.pop(0)) == ""
