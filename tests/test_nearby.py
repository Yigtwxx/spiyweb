"""The folders around the monitor: indexes, corpora, and typed paths."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from spiyweb.nearby import (
    complete_path,
    describe,
    find_indexes,
    index_stamp,
    is_index,
    read_meta,
    split_last_token,
    text_folders,
)

if TYPE_CHECKING:
    from pathlib import Path

SUFFIXES = (".txt", ".md")


def _index(path: Path, **meta: object) -> Path:
    path.mkdir(parents=True)
    (path / "nodes.json").write_text("[]", encoding="utf-8")
    if meta:
        (path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def _text(path: Path, text: str = "a passage") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_an_index_is_a_folder_with_nodes_json(tmp_path: Path) -> None:
    assert is_index(_index(tmp_path / "idx"))
    (tmp_path / "plain").mkdir()
    assert not is_index(tmp_path / "plain")
    assert not is_index(tmp_path / "missing")


def test_read_meta_survives_a_missing_or_broken_receipt(tmp_path: Path) -> None:
    idx = _index(tmp_path / "idx")
    assert read_meta(idx) == {}
    (idx / "meta.json").write_text("{not json", encoding="utf-8")
    assert read_meta(idx) == {}
    assert index_stamp(idx) > 0, "the receipt's time names the build"


def test_describe_reads_the_receipt_only(tmp_path: Path) -> None:
    idx = _index(
        tmp_path / "idx",
        nodes=42,
        propositions=None,
        embedding_model="hash-bow-64",
        edges={"semantic": 10, "entity": 0, "structural": 3},
    )
    info = describe(idx)
    assert (info.nodes, info.propositions, info.model) == (42, None, "hash-bow-64")
    assert info.layers == ("semantic", "structural"), "empty layers are not listed"


def test_find_indexes_offers_the_given_ones_first_then_nearby(tmp_path: Path) -> None:
    _index(tmp_path / "beta")
    _index(tmp_path / "data" / "alpha")
    elsewhere = _index(tmp_path / "far" / "away" / "gamma")
    (tmp_path / "notes").mkdir()
    found = [info.path.name for info in find_indexes(tmp_path, [elsewhere])]
    assert found == ["gamma", "beta", "alpha"]
    again = find_indexes(tmp_path, [tmp_path / "beta", "beta"])
    assert [i.path.name for i in again].count("beta") == 1, "one index, one row"


def test_text_folders_count_like_the_index_verb_reads(tmp_path: Path) -> None:
    _text(tmp_path / "docs" / "a.md")
    _text(tmp_path / "docs" / "deep" / "b.TXT")
    _text(tmp_path / "docs" / "deep" / "c.py")
    _text(tmp_path / "top.md")
    _text(tmp_path / ".hidden" / "x.md")
    _text(tmp_path / ".venv" / "y.md")
    _text(tmp_path / "node_modules" / "z.md")
    _index(tmp_path / "an-index")
    _text(tmp_path / "an-index" / "notes.md")
    found = {
        folder.path.relative_to(tmp_path.resolve()).as_posix(): folder.files
        for folder in text_folders(tmp_path, SUFFIXES, max_depth=6, max_files=100)
    }
    assert found == {".": 1, "docs": 2, "docs/deep": 1}


def test_split_last_token_keeps_a_quoted_path_whole() -> None:
    assert split_last_token("/index my") == ("/index ", "my")
    assert split_last_token('/index "my do') == ("/index ", '"my do')
    assert split_last_token("/index ") == ("/index ", "")


def test_complete_path_lists_folders_the_way_they_are_typed(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "Data Sets").mkdir()
    (tmp_path / "dataset.md").write_text("", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / "docs" / "inner").mkdir()
    assert complete_path("do", tmp_path) == ["docs/"]
    assert complete_path("da", tmp_path) == ['"Data Sets/"'], "files and case aside"
    assert complete_path("docs/", tmp_path) == ["docs/inner/"]
    assert complete_path("docs\\in", tmp_path) == ["docs/inner/"]
    assert ".git/" not in complete_path("", tmp_path)
    assert complete_path(".g", tmp_path) == [".git/"]
    assert complete_path("nowhere/x", tmp_path) == []


def test_complete_path_keeps_an_absolute_root(tmp_path: Path) -> None:
    (tmp_path / "Users").mkdir()
    (tmp_path / "docs").mkdir()
    root = tmp_path.as_posix() + "/"
    assert complete_path(root + "Us", tmp_path / "docs") == [root + "Users/"]
    assert root + "docs/" in complete_path(root, tmp_path / "docs")


def test_complete_path_at_the_filesystem_root_stays_at_the_root(
    tmp_path: Path,
) -> None:
    from pathlib import Path as RealPath

    anchor = RealPath(tmp_path.anchor).as_posix()  # "C:/" or "/"
    found = complete_path(anchor, tmp_path)
    assert found, "the root has folders"
    assert all(entry.strip('"').startswith(anchor) for entry in found), found
