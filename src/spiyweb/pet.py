"""The spider in the monitor's welcome box.

A silhouette, not a cartoon: a small head, a round abdomen, eight legs, on
braille dots so it survives any monospace font that has them. The left half
was rasterised from a reference drawing and mirrored, so the two sides match
to the dot. It does not move - the owner's call: a pet that fidgets steals
attention from the web, which is the thing worth watching.

Two sizes, chosen by the terminal's height, and an ASCII fallback for a
console without braille. Every frame set is a rectangle - equal line count,
equal width - so the box around it never has to guess.
"""

from __future__ import annotations

__all__ = ["ASCII", "FULL", "MINI", "pet_lines", "pet_width"]

FULL: tuple[str, ...] = (
    "⡆   ⡠          ⢄   ⢰",
    "⢷⡀ ⠰⡇          ⢸⠆ ⢀⡾",
    "⠈⠓⠦⢤⣝⣓⣦⣤⡀⣤⣤⢀⣤⣴⣚⣫⡤⠴⠚⠁",
    "   ⣀⡤⢴⣿⢿⣼⣿⣿⣧⡿⣿⡦⢤⣀   ",
    " ⣴⠋⣡⠖⠋⢰⣿⣿⣿⣿⣿⣿⡆⠙⠲⣌⠙⣦ ",
    " ⢻⡀⠳⣄ ⠘⢿⣿⣿⣿⣿⡿⠃ ⣠⠞⢀⡟ ",
    " ⠈⠳ ⠈⠙⠦⣄⡀⠉⠉⢀⣠⠴⠋⠁ ⠞⠁ ",
)
"""Twenty columns by seven rows, for terminals with room."""

MINI: tuple[str, ...] = (
    "⡇  ⢠⠆        ⠰⡄  ⢸",
    "⠙⢦⣀⣘⣦⣄⣀⡀⣀⣀⢀⣀⣠⣴⣃⣀⡴⠋",
    "  ⣀⡨⢭⣿⢿⣥⣿⣿⣬⡿⣿⡭⢅⣀  ",
    " ⣿⢡⡞⠉⠰⣿⣿⣿⣿⣿⣿⠆⠉⢳⡌⣿ ",
    " ⠘⠆⠉⠓⠦⣈⠉⠛⠛⠉⣁⠴⠚⠉⠰⠃ ",
)
"""Eighteen by five, for a terminal under `MINI_BELOW_ROWS` rows."""

ASCII: tuple[str, ...] = (
    "   \\  .-+-.  /   ",
    " ---\\-(o o)-/--- ",
    " ---/-(---)-\\--- ",
    "   /  '---'  \\   ",
)
"""The line-art fallback: legs, a head with eyes, a belly. Seventeen by four."""

MINI_BELOW_ROWS = 44
"""Terminals shorter than this get the small spider so the web keeps its room."""


def pet_lines(*, rows: int, unicode: bool) -> tuple[str, ...]:
    """The frame set for a terminal `rows` high."""
    if not unicode:
        return ASCII
    return MINI if rows < MINI_BELOW_ROWS else FULL


def pet_width(frames: tuple[str, ...]) -> int:
    return max(len(line) for line in frames)
