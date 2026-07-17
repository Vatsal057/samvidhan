"""
The public entry point: ``answer(query)`` runs retrieve → generate and returns a
single ``RAGResult`` with the answer, its sources, and a confidence proxy.

Generation is best-effort. If it fails (no token, model unavailable, network),
the result is still useful: ``answer`` carries the reason and the retrieved
sources are returned so the UI can show what matched.
"""

from __future__ import annotations

from dataclasses import dataclass

from .generate import GenerationError, generate
from .retrieve import Retrieved, retrieve


@dataclass
class RAGResult:
    answer: str
    sources: list[Retrieved]
    retrieval_score: float  # mean similarity of the retrieved chunks
    generated: bool  # True if an LLM produced the answer; False = retrieval-only


def _mean_score(hits: list[Retrieved]) -> float:
    return sum(h.score for h in hits) / len(hits) if hits else 0.0


def answer(query: str) -> RAGResult:
    hits = retrieve(query)
    score = _mean_score(hits)

    if not hits:
        return RAGResult(
            answer="The knowledge base is empty. Run `python -m samvidhan.ingest` first.",
            sources=[],
            retrieval_score=0.0,
            generated=False,
        )

    try:
        text = generate(query, hits)
        return RAGResult(answer=text, sources=hits, retrieval_score=score, generated=True)
    except GenerationError as exc:
        note = (
            "Answer generation is unavailable, so here are the most relevant "
            f"sections instead.\n\n_({exc})_"
        )
        return RAGResult(answer=note, sources=hits, retrieval_score=score, generated=False)
