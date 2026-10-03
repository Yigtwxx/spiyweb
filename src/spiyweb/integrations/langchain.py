"""LangChain retriever over a Spiyweb index (`pip install spiyweb[langchain]`).

    from spiyweb import open_index
    from spiyweb.integrations.langchain import SpiywebRetriever

    retriever = SpiywebRetriever(index=open_index("my_index"))
    docs = retriever.invoke("who funded Wardenclyffe?")

Documents come back in the web's order - accumulated energy, strongest
first - and there is no `k` by default: the web stops where its energy dies,
which is the point of it. `max_documents` is a hard cap for a context window,
applied after the ranking, never instead of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError as error:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "the LangChain adapter needs langchain-core; install it with "
        "`pip install spiyweb[langchain]`"
    ) from error

from pydantic import ConfigDict

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun

    from spiyweb.session import Passage


def passage_metadata(passage: Passage, query: str) -> dict[str, Any]:
    """What a passage carries beyond its text, under stable key names."""
    return {
        "node_id": passage.node_id,
        "source_id": passage.source_id,
        "layer": passage.layer,
        "energy": passage.energy,
        "votes": passage.votes,
        "hop": passage.hop,
        "query": query,
    }


class SpiywebRetriever(BaseRetriever):
    """A `BaseRetriever` whose documents are a Spiyweb web's activated passages.

    Attributes:
        index: An open `spiyweb.SpiywebIndex`.
        profile: `precise` / `explore` / `compare`, or `None` for the
            library default (`explore`).
        max_documents: Optional cap on returned documents; `None` returns
            every activated passage that carries text.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: Any
    profile: str | None = None
    max_documents: int | None = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        answer = self.index.retrieve(query, profile=self.profile)
        passages = [passage for passage in answer.passages if passage.text]
        if self.max_documents is not None:
            passages = passages[: self.max_documents]
        return [
            Document(
                page_content=passage.text,
                metadata=passage_metadata(passage, query),
                id=passage.node_id,
            )
            for passage in passages
        ]
