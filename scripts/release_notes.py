"""Print the CHANGELOG section of one version, for a GitHub Release's notes.

``python scripts/release_notes.py 0.2.4`` writes the body under
``## [0.2.4] - ...`` up to the next ``## [`` heading, without the heading
itself. Exit 1 when the version has no section, or a heading with nothing
under it, so a release never goes out with empty notes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"


def section(version: str, text: str) -> str | None:
    """The body of ``## [version]``, or ``None`` when there is no such heading."""
    pattern = re.compile(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    return match.group(1).strip() + "\n" if match else None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: release_notes.py VERSION", file=sys.stderr)
        return 2
    body = section(argv[1].lstrip("v"), CHANGELOG.read_text(encoding="utf-8"))
    if body is None or not body.strip():
        # A heading with an empty body is as missing as no heading.
        print(f"CHANGELOG.md has no section for {argv[1]}", file=sys.stderr)
        return 1
    sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
