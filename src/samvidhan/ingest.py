"""
Ingestion: load a source document, chunk it article-by-article, embed each
chunk, and store everything in ChromaDB.

Two sources are supported:

* ``--pdf constitution_of_india.pdf`` — the full official PDF (see README).
* ``--text data/sample/`` — a folder of ``.txt`` files, used for the offline
  demo and CI. Each blank-line-separated block is treated as one "page".

Run:
    python -m samvidhan.ingest --pdf constitution_of_india.pdf
    python -m samvidhan.ingest --text data/sample
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from .chunking import Chunk, chunk_pages
from .config import settings


def _pages_from_pdf(pdf_path: str) -> list[dict]:
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    print(f"  extracted {len(pages)} pages from {pdf_path}")
    return pages


def _pages_from_text(path: str) -> list[dict]:
    """Read .txt file(s); each double-newline block becomes a numbered 'page'."""
    p = Path(path)
    files = sorted(p.glob("*.txt")) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"No .txt files found at {path}")

    pages, page_no = [], 0
    for f in files:
        for block in re.split(r"\n\s*\n", f.read_text(encoding="utf-8")):
            block = re.sub(r"\s+", " ", block).strip()
            if block:
                page_no += 1
                pages.append({"page": page_no, "text": block})
    print(f"  read {len(pages)} text blocks from {path}")
    return pages


def _store(chunks: list[Chunk]) -> int:
    import chromadb
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embed_model)
    client = chromadb.PersistentClient(path=settings.chroma_dir)
    try:
        client.delete_collection(settings.collection)  # idempotent re-ingest
    except Exception:  # noqa: BLE001
        pass
    collection = client.create_collection(
        name=settings.collection, metadata={"hnsw:space": "cosine"}
    )

    batch = 64
    for i in range(0, len(chunks), batch):
        group = chunks[i : i + batch]
        collection.add(
            ids=[c.id for c in group],
            documents=[c.text for c in group],
            embeddings=model.encode([c.text for c in group]).tolist(),
            metadatas=[{"page": c.page, "article": c.article or ""} for c in group],
        )
    return collection.count()


def ingest(pdf: str | None = None, text: str | None = None) -> int:
    if pdf:
        pages = _pages_from_pdf(pdf)
    elif text:
        pages = _pages_from_text(text)
    else:
        raise ValueError("Provide either --pdf or --text")

    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    numbered = sum(1 for c in chunks if c.article)
    print(f"  created {len(chunks)} chunks ({numbered} article-labelled)")
    stored = _store(chunks)
    print(f"  stored {stored} chunks in ChromaDB -> {settings.chroma_dir}/")
    return stored


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest a document into ChromaDB.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", help="Path to the Constitution of India PDF")
    src.add_argument("--text", help="Path to a .txt file or folder of .txt files")
    args = ap.parse_args()
    ingest(pdf=args.pdf, text=args.text)
    print("\nDone. Launch the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
