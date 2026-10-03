"""LlamaIndex retriever over a Spiyweb index (`pip install spiyweb[llamaindex]`).

    from spiyweb import open_index
    from spiyweb.integrations.llamaindex import SpiywebLlamaRetriever

    retriever = SpiywebLlamaRetriever(open_index("my_index"))
    nodes = retriever.retrieve("who funded Wardenclyffe?")

Each result is a `NodeWithScore` whose score is the passage's accumulated
energy - the web's own ranking key, not a cosine - so a postprocessor that
sorts by score keeps the web's order. As in the LangChain adapter there is
no `k` by default; `max_nodes` caps the list after the ranking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
except ImportError as error:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "the LlamaIndex adapter needs llama-index-core; install it with "
        "`pip install spiyweb[llamaindex]`"
    ) from error

if TYPE_CHECKING:
    from spiyweb.session import SpiywebIndex


class SpiywebLlamaRetriever(BaseRetriever):
    """A LlamaIndex `BaseRetriever` backed by a Spiyweb web.

    Args:
        index: An open `spiyweb.SpiywebIndex`.
        profile: `precise` / `explore` / `compare`, or `None` for the
            library default (`explore`).
        max_nodes: Optional cap on returned nodes; `None` returns every
            activated passage that carries text.
    """

    def __init__(
        self,
        index: SpiywebIndex,
        *,
        profile: str | None = None,
        max_nodes: int | None = None,
    ) -> None:
        super().__init__()
        self._index = index
        self._profile = profile
        self._max_nodes = max_nodes

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query = query_bundle.query_str
        answer = self._index.retrieve(query, profile=self._profile)
        passages = [passage for passage in answer.passages if passage.text]
        if self._max_nodes is not None:
            passages = passages[: self._max_nodes]
        return [
            NodeWithScore(
                node=TextNode(
                    id_=passage.node_id,
                    text=passage.text,
                    metadata={
                        "node_id": passage.node_id,
                        "source_id": passage.source_id,
                        "layer": passage.layer,
                        "votes": passage.votes,
                        "hop": passage.hop,
                    },
                ),
                score=passage.energy,
            )
            for passage in passages
        ]
