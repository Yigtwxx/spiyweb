"""Benchmark adapter over `spiyweb.indexing` - MuSiQue in, artifacts out.

The pipeline itself moved to `spiyweb.indexing` on 2026-08-25, and what is
left here is the part that is genuinely about the benchmark: the dataset
file names, the title-plus-text composition MuSiQue needs, and the sampled
question count that belongs in the receipt. `IndexPaths` extends
`IndexLayout` rather than replacing it, so every existing caller - the
harness, the inspector, the tests - keeps the names it already imports.

The split is not cosmetic. Every sealed Phase 1 number went through the code
this module now delegates to, so a user's own corpus and the measurement runs
share one implementation instead of two that drift; `tests/test_index_golden.py`
pins the artifacts across the move.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from spiyweb.indexing import (
    IndexLayout,
    IndexManifest,
    load_entities,
    load_graph,
    load_nli_edges,
    load_propositions,
    load_similarity,
    load_store,
    load_texts,
    read_manifest,
)
from spiyweb.indexing import build_index as _build_index

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from spiyweb.config import (
        EntityEdgeConfig,
        EntityExtractionConfig,
        NLICandidateConfig,
        NLIEdgeConfig,
        PropositionConfig,
        SemanticEdgeConfig,
        StructuralEdgeConfig,
    )
    from spiyweb.edges import NLIModel
    from spiyweb.embedding import Embedder
    from spiyweb.entities import EntityPipeline
    from spiyweb.evaluation.datasets import MusiqueDataset
    from spiyweb.llm import LLMClient

__all__ = [
    "IndexLayout",
    "IndexManifest",
    "IndexPaths",
    "build_index",
    "composed_text",
    "load_entities",
    "load_graph",
    "load_nli_edges",
    "load_propositions",
    "load_similarity",
    "load_store",
    "load_texts",
    "read_manifest",
]


@dataclass(frozen=True)
class IndexPaths(IndexLayout):
    """`IndexLayout` plus the files only a benchmark run has.

    The dataset downloads and the evaluation output live here; the artifacts
    the library reads live on the base class. Being a subclass is what keeps
    `load_graph(paths)` and friends working unchanged.
    """

    @property
    def dataset_jsonl(self) -> Path:
        return self.root / "musique_ans_v1.0_dev.jsonl"

    @property
    def twowiki_dev_json(self) -> Path:
        return self.root / "2wiki_dev.json"

    @property
    def hotpot_dev_json(self) -> Path:
        return self.root / "hotpot_dev_distractor.json"

    @property
    def results_json(self) -> Path:
        return self.root / "results.json"

    @property
    def per_query_jsonl(self) -> Path:
        return self.root / "per_query.jsonl"

    @property
    def answers_jsonl(self) -> Path:
        return self.root / "answers.jsonl"

    @property
    def answer_results_json(self) -> Path:
        return self.root / "answer_results.json"


def composed_text(title: str, text: str) -> str:
    """The one string both the embedder and the extractor see for a passage.

    MuSiQue titles are entity-dense - they name the article's subject - so
    the passage is indexed with its title attached. A corpus without titles
    passes nothing and gets the chunk's own text; the choice is the caller's,
    which is exactly why it is here and not in the pipeline.
    """
    return f"{title}\n{text}"


def build_index(
    dataset: MusiqueDataset,
    paths: IndexPaths,
    *,
    embedder: Embedder,
    entity_pipeline: EntityPipeline,
    llm: LLMClient | None = None,
    embedding_model: str | None = None,
    extraction_config: EntityExtractionConfig | None = None,
    semantic_config: SemanticEdgeConfig | None = None,
    structural_config: StructuralEdgeConfig | None = None,
    entity_config: EntityEdgeConfig | None = None,
    llm_model: str | None = None,
    entity_llm: bool = True,
    propositions: bool = False,
    proposition_config: PropositionConfig | None = None,
    nli_model: NLIModel | None = None,
    nli_model_name: str | None = None,
    nli_config: NLIEdgeConfig | None = None,
    nli_candidates: NLICandidateConfig | None = None,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> IndexManifest:
    """Index a `MusiqueDataset` - the corpus-agnostic build, benchmark-shaped.

    Two things are added on top of `spiyweb.indexing.build_index`: passages
    are composed as `title + newline + text`, and the sampled question count
    joins the receipt. Everything else passes straight through.
    """
    texts = {
        chunk_id: composed_text(dataset.titles[chunk_id], dataset.texts[chunk_id])
        for chunk_id in dataset.texts
    }
    return _build_index(
        list(dataset.documents),
        paths,
        embedder=embedder,
        entity_pipeline=entity_pipeline,
        texts=texts,
        llm=llm,
        embedding_model=embedding_model,
        extraction_config=extraction_config,
        semantic_config=semantic_config,
        structural_config=structural_config,
        entity_config=entity_config,
        llm_model=llm_model,
        entity_llm=entity_llm,
        propositions=propositions,
        proposition_config=proposition_config,
        nli_model=nli_model,
        nli_model_name=nli_model_name,
        nli_config=nli_config,
        nli_candidates=nli_candidates,
        # The library has no business knowing what a "question" is; the
        # harness records its own sample size and nothing else changes.
        extra_meta={"questions": len(dataset.questions)},
        force=force,
        log=log,
    )
