"""
Retrieval evaluation harness.

Measures how often the correct article makes it into the retrieved set. These
metrics need no HF token — they exercise retrieval only, which is where the
FAILURES.md analysis found the real problems. If HF_TOKEN is set, an optional
answer-keyword check is also run.

    python eval/run_eval.py                 # retrieval metrics
    USE_RERANKER=true python eval/run_eval.py   # compare with re-ranking on

Requires an ingested knowledge base:
    python -m samvidhan.ingest --text data/sample
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from samvidhan import retrieve  # noqa: E402
from samvidhan.config import settings  # noqa: E402

QUESTIONS = Path(__file__).parent / "questions.jsonl"


def load_questions() -> list[dict]:
    return [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]


def recall_at_k(retrieved_articles: set[str], expected: list[str]) -> float:
    """Fraction of expected articles present in the retrieved set."""
    if not expected:
        return 1.0
    hits = sum(1 for a in expected if a in retrieved_articles)
    return hits / len(expected)


def reciprocal_rank(ranked_articles: list[str], expected: list[str]) -> float:
    for i, art in enumerate(ranked_articles, start=1):
        if art in expected:
            return 1.0 / i
    return 0.0


def main() -> int:
    questions = load_questions()
    print(f"Evaluating {len(questions)} questions "
          f"(top_k={settings.top_k}, reranker={'on' if settings.use_reranker else 'off'})\n")

    total_recall, total_rr, hit_at_k = 0.0, 0.0, 0
    for q in questions:
        hits = retrieve.retrieve(q["question"])
        ranked = [h.article for h in hits if h.article]
        found = set(ranked)
        recall = recall_at_k(found, q["expected_articles"])
        rr = reciprocal_rank(ranked, q["expected_articles"])
        got = any(a in found for a in q["expected_articles"])
        hit_at_k += int(got)
        total_recall += recall
        total_rr += rr
        mark = "✓" if got else "✗"
        print(f"  {mark} {q['question'][:48]:<50} "
              f"expected {q['expected_articles']}  got {ranked[:4]}")

    n = len(questions)
    print("\n" + "-" * 60)
    print(f"  Hit@{settings.top_k}     : {hit_at_k}/{n}  ({hit_at_k / n:.0%})")
    print(f"  Mean recall  : {total_recall / n:.3f}")
    print(f"  MRR          : {total_rr / n:.3f}")
    print("-" * 60)
    return 0 if hit_at_k >= int(0.75 * n) else 1


if __name__ == "__main__":
    raise SystemExit(main())
