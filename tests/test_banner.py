"""The SPIYWEB wordmark: its shape, its colours, and its place in the welcome box."""

from __future__ import annotations

import io
from pathlib import Path

from spiyweb.banner import (
    BANDS,
    SHADOW,
    WORDMARK,
    wordmark_lines,
    wordmark_width,
)
from spiyweb.config import WatchConfig
from spiyweb.pet import FULL
from spiyweb.terminal import printed_width
from spiyweb.watch import Monitor


def test_the_mark_is_six_rows_of_letters_and_shadow() -> None:
    # Six rows: the height of the owner's other tools' wordmarks.
    assert len(WORDMARK) == 6
    assert len({len(line) for line in WORDMARK}) == 1
    assert set("".join(WORDMARK)) <= set("█╗╔═╝║╚ ")
    assert len(set(BANDS)) == len(BANDS) and SHADOW not in BANDS


def test_every_line_is_the_mark_width_with_colour_or_without() -> None:
    for color in (False, True):
        lines = wordmark_lines(color=color)
        assert len(lines) == len(WORDMARK)
        assert {printed_width(line) for line in lines} == {wordmark_width()}


def test_colour_paints_the_gradient_left_to_right_and_off_means_plain() -> None:
    assert wordmark_lines(color=False) == list(WORDMARK)
    first = wordmark_lines(color=True)[0]
    positions = [first.index(f"\x1b[{code}m") for code in BANDS]
    assert positions == sorted(positions)
    assert f"\x1b[{SHADOW}m" in first


def _monitor(tmp_path: Path, size: tuple[int, int], **knobs: object) -> Monitor:
    return Monitor(
        config=WatchConfig(**knobs),  # type: ignore[arg-type]
        directory=tmp_path / ".spiyweb",
        write=io.StringIO().write,
        poll=lambda _: None,
        size=lambda: size,
        color=False,
        unicode=True,
        cwd=tmp_path,
    )


def _box(lines: list[str]) -> list[str]:
    end = next(i for i, line in enumerate(lines) if line.startswith("╰"))
    return lines[: end + 1]


def test_the_mark_sits_left_of_the_spider_on_its_last_row(tmp_path: Path) -> None:
    box = _box(_monitor(tmp_path, (120, 30)).screen(0.0))
    row = next(i for i, line in enumerate(box) if WORDMARK[0] in line)
    assert row + len(WORDMARK) == len(FULL) + 1, "mark ends on the spider's row"
    line = box[row]
    assert line.index(WORDMARK[0]) < line.index(FULL[row - 1])


def test_the_text_goes_beside_when_it_fits_and_under_otherwise(tmp_path: Path) -> None:
    wide = _box(_monitor(tmp_path, (140, 32)).screen(0.0))
    assert len(wide) == len(FULL) + 2
    assert any("███" in line and "Welcome to spiyweb" in line for line in wide)
    narrow = _box(_monitor(tmp_path, (100, 32)).screen(0.0))
    assert len(narrow) > len(FULL) + 2
    welcome = next(line for line in narrow if "Welcome to spiyweb" in line)
    assert "███" not in welcome


def test_the_plain_box_stands_when_the_mark_would_not_fit(tmp_path: Path) -> None:
    def marked(monitor: Monitor) -> bool:
        return any(WORDMARK[0] in line for line in monitor.screen(0.0))

    assert marked(_monitor(tmp_path, (120, 30)))
    assert not marked(_monitor(tmp_path, (120, 30), banner_enabled=False))
    assert not marked(_monitor(tmp_path, (72, 30))), "too narrow"
    assert not marked(_monitor(tmp_path, (120, 24))), "the map keeps its room"
    ascii_monitor = _monitor(tmp_path, (120, 30))
    ascii_monitor.settings.ascii = True
    ascii_monitor.apply_settings()
    assert not marked(ascii_monitor)


def test_without_the_spider_the_mark_stands_alone(tmp_path: Path) -> None:
    box = _box(_monitor(tmp_path, (120, 30), pet_enabled=False).screen(0.0))
    assert any(WORDMARK[0] in line for line in box)
    assert not any(FULL[3] in line for line in box)
