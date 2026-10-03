"""Framework adapters: same web, same order, framework-shaped results.

Skipped without the extras - CI's base install does not carry LangChain or
LlamaIndex, and the zero-dependency promise means it never has to.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from spiyweb.session import SpiywebIndex

QUERY = "who raised the tower"


def test_importing_spiyweb_never_pulls_a_framework_in() -> None:
    code = (
        "import sys, spiyweb, spiyweb.integrations; "
        "bad = [m for m in ('langchain_core', 'llama_index') if m in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.call([sys.executable, "-c", code]) == 0, (
        "the adapters are extras - `import spiyweb` must stay framework-free"
    )


def test_langchain_documents_follow_the_web(
    open_tiny: Callable[..., SpiywebIndex],
) -> None:
    pytest.importorskip("langchain_core")
    from spiyweb.integrations.langchain import SpiywebRetriever

    with open_tiny() as index:
        expected = [p for p in index.retrieve(QUERY).passages if p.text]
        docs = SpiywebRetriever(index=index).invoke(QUERY)

    assert [doc.page_content for doc in docs] == [p.text for p in expected], (
        "the adapter keeps the web's energy order and adds no ranking of its own"
    )
    first = docs[0].metadata
    assert first["node_id"] == expected[0].node_id
    assert first["energy"] == expected[0].energy
    assert {"votes", "hop", "source_id", "layer"} <= set(first)


def test_langchain_max_documents_caps_after_the_ranking(
    open_tiny: Callable[..., SpiywebIndex],
) -> None:
    pytest.importorskip("langchain_core")
    from spiyweb.integrations.langchain import SpiywebRetriever

    with open_tiny() as index:
        full = SpiywebRetriever(index=index).invoke(QUERY)
        capped = SpiywebRetriever(index=index, max_documents=1).invoke(QUERY)

    assert len(full) > 1
    assert [d.id for d in capped] == [full[0].id]


def test_llamaindex_nodes_are_scored_by_energy(
    open_tiny: Callable[..., SpiywebIndex],
) -> None:
    pytest.importorskip("llama_index.core")
    from spiyweb.integrations.llamaindex import SpiywebLlamaRetriever

    with open_tiny() as index:
        expected = [p for p in index.retrieve(QUERY).passages if p.text]
        nodes = SpiywebLlamaRetriever(index).retrieve(QUERY)
        capped = SpiywebLlamaRetriever(index, max_nodes=2).retrieve(QUERY)

    assert [n.node.get_content() for n in nodes] == [p.text for p in expected]
    assert [n.score for n in nodes] == [p.energy for p in expected], (
        "the score is the web's own ranking key, not a cosine"
    )
    assert nodes[0].node.metadata["source_id"] == expected[0].source_id
    assert len(capped) == 2
