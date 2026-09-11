"""A small application that asks spiyweb things - to watch from the monitor.

    spiyweb                      # in this folder, in one terminal
    ! python examples/hello_web.py   # typed into the monitor's prompt

or, in a second terminal, plainly `python examples/hello_web.py`. Either way
every `retrieve()` below spreads on the monitor as it happens.

No model is downloaded: the embedder is a bag-of-words hash (sentences that
share words land near each other) and the entity finder is a capitalised-word
rule. That is enough for a five-passage corpus about a tower, and it keeps
this file runnable on a bare `pip install spiyweb` plus the `store` extra
(numpy + faiss), which building any index needs.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from pathlib import Path

from spiyweb import open_index
from spiyweb.config import EntityEdgeConfig, EntityExtractionConfig
from spiyweb.indexing import DocumentInput, TextUnit, build_index

INDEX = Path("hello-index")
DIMENSIONS = 64

CORPUS = (
    DocumentInput(
        source_id="tesla",
        units=(
            TextUnit(text="Nikola Tesla built the Wardenclyffe Tower on Long Island."),
            TextUnit(text="The Wardenclyffe Tower was meant for wireless power."),
        ),
    ),
    DocumentInput(
        source_id="morgan",
        units=(
            TextUnit(text="J. P. Morgan financed the Wardenclyffe project at first."),
            TextUnit(text="Morgan later withdrew his money from Tesla."),
        ),
    ),
    DocumentInput(
        source_id="shoreham",
        units=(
            TextUnit(
                text="Shoreham is the village on Long Island where the tower stood."
            ),
        ),
    ),
)

QUESTIONS = (
    "who built the tower on long island",
    "who paid for wardenclyffe",
    "where did the tower stand",
)


class HashEmbedder:
    """Bag of words into a fixed vector: shared words, nearby vectors."""

    model_name = "hash-bow-64"

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * DIMENSIONS
        for word in re.findall(r"[a-z]+", text.lower()):
            digest = hashlib.blake2b(word.encode(), digest_size=4).digest()
            vector[int.from_bytes(digest, "big") % DIMENSIONS] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]


class CapitalisedEntities:
    """Every capitalised word that is not the first of its sentence."""

    def pipe(self, texts: list[str]) -> list[object]:
        class Span:
            def __init__(self, text: str) -> None:
                self.text, self.label_ = text, "ENT"

        class Doc:
            def __init__(self, ents: tuple[Span, ...]) -> None:
                self.ents = ents

        docs = []
        for text in texts:
            words = re.findall(r"[A-Za-z][a-z]+", text)
            ents = tuple(Span(w) for w in words[1:] if w[0].isupper())
            docs.append(Doc(ents))
        return docs


def main() -> None:
    embedder = HashEmbedder()
    if not (INDEX / "nodes.json").exists():
        print(f"building {INDEX} (once)")
        build_index(
            CORPUS,
            INDEX,
            embedder=embedder,
            entity_pipeline=CapitalisedEntities(),
            embedding_model=embedder.model_name,
            extraction_config=EntityExtractionConfig(min_entities=0),
            entity_config=EntityEdgeConfig(max_df_ratio=1.0),
            log=lambda message: None,
        )
    index = open_index(INDEX, embedder=embedder)
    for question in QUESTIONS:
        answer = index.retrieve(question)
        top = answer.passages[0] if answer.passages else None
        print(f"asked: {question}")
        if top is not None:
            print(f"   -> {top.text}  ({top.energy:.2f})")
        time.sleep(2)
    print("done - /replay plays them again")


if __name__ == "__main__":
    main()
