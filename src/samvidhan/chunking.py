"""
Article-aware chunking for the Constitution of India.

Fixed-size character chunking splits articles mid-sentence and destroys the
document's natural structure — the single biggest source of retrieval errors in
the first version of this project (see FAILURES.md). The Constitution has a very
regular shape: an article number, a title, then clauses. This module exploits
that.

Strategy
--------
1. Flatten the per-page text into one stream while remembering which page each
   character came from (so citations stay accurate).
2. Split the stream at article headers (e.g. ``21. Protection of life ...``),
   keeping the article number as metadata.
3. Emit one chunk per article. Only articles longer than ``chunk_size`` are
   sub-split, with overlap, and every sub-chunk keeps its article label.

If a document has no recognisable article headers (arbitrary text, or a badly
scanned PDF), the whole thing falls back to plain sliding-window chunking so the
pipeline never breaks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# An article header: a 1–3 digit number (optionally suffixed like "21A"),
# a period, whitespace, then a capitalised word starting the title.
_ARTICLE_HEADER = re.compile(r"(?:(?<=\s)|^)(\d{1,3}[A-Z]{0,2})\.\s+(?=[A-Z])")


@dataclass
class Chunk:
    id: str
    text: str
    page: int
    article: str | None  # e.g. "21", "21A", or None for unnumbered content


def _flatten(pages: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    """Join pages into one string; return (text, [(char_offset, page_number)])."""
    buf: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for p in pages:
        text = p["text"].strip()
        if not text:
            continue
        offsets.append((cursor, int(p["page"])))
        buf.append(text)
        cursor += len(text) + 1  # +1 for the single-space join
    return " ".join(buf), offsets


def _page_at(offsets: list[tuple[int, int]], pos: int) -> int:
    """Return the page number that contains character position ``pos``."""
    page = offsets[0][1] if offsets else 1
    for start, pg in offsets:
        if pos >= start:
            page = pg
        else:
            break
    return page


def _split_articles(text: str) -> list[tuple[str | None, int, str]]:
    """Return [(article_number, start_offset, body)] split at article headers."""
    marks = [(m.start(), m.group(1)) for m in _ARTICLE_HEADER.finditer(text)]
    if not marks:
        return [(None, 0, text)]

    spans: list[tuple[str | None, int, str]] = []
    # Any preamble before the first article header is kept as unnumbered content.
    if marks[0][0] > 0:
        head = text[: marks[0][0]].strip()
        if head:
            spans.append((None, 0, head))

    for i, (pos, num) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[pos:end].strip()
        if body:
            spans.append((num, pos, body))
    return spans


def _window(body: str, size: int, overlap: int) -> list[tuple[int, str]]:
    """Sliding-window split of an over-long body. Yields (local_offset, text)."""
    if len(body) <= size:
        return [(0, body)]
    step = max(1, size - overlap)
    out: list[tuple[int, str]] = []
    start = 0
    while start < len(body):
        piece = body[start : start + size].strip()
        if piece:
            out.append((start, piece))
        start += step
    return out


def chunk_pages(
    pages: list[dict], chunk_size: int = 900, chunk_overlap: int = 120
) -> list[Chunk]:
    """Turn extracted pages into article-aware chunks with page + article metadata."""
    text, offsets = _flatten(pages)
    if not text:
        return []

    chunks: list[Chunk] = []
    for article, abs_start, body in _split_articles(text):
        for local_off, piece in _window(body, chunk_size, chunk_overlap):
            if len(piece) < 40:  # drop trivial fragments
                continue
            pos = abs_start + local_off
            chunks.append(
                Chunk(
                    id=f"c{len(chunks):05d}",
                    text=piece,
                    page=_page_at(offsets, pos),
                    article=article,
                )
            )
    return chunks


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    demo_pages = [
        {
            "page": 3,
            "text": "19. Protection of certain rights regarding freedom of speech. "
            "All citizens shall have the right to freedom of speech and expression. "
            "20. Protection in respect of conviction for offences. No person shall "
            "be convicted of any offence except for violation of a law in force.",
        },
        {
            "page": 4,
            "text": "21. Protection of life and personal liberty. No person shall be "
            "deprived of his life or personal liberty except according to procedure "
            "established by law.",
        },
    ]
    result = chunk_pages(demo_pages)
    assert {c.article for c in result} == {"19", "20", "21"}, result
    assert next(c for c in result if c.article == "21").page == 4
    print(f"OK — {len(result)} chunks, articles: {sorted({c.article for c in result})}")
