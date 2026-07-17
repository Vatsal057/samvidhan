# RAG Failure Analysis — Constitution of India Q&A

This document records every failure category I found while stress-testing the system with 20 queries, along with what I changed to address each one. I'm including this because I think it's more useful than another "works perfectly" demo.

> **Status update.** The two highest-impact fixes this analysis recommended —
> **article-aware chunking** and **cross-encoder re-ranking** — are now implemented
> (`src/samvidhan/chunking.py`, `src/samvidhan/retrieve.py`). Retrieval quality is now
> tracked continuously by the evaluation harness in [`eval/`](eval/), which runs in CI on
> every push. Run it yourself with `make eval`.

---

## Test Setup

- 20 queries designed to cover different article types, levels of specificity, and edge cases
- Evaluation criteria: correct answer + correct source page citation
- Chunks: 500 characters with 80-character overlap
- Retrieval: top-5 chunks, cosine similarity

---

## Results Summary

| Category | Count | % |
|---|---|---|
| ✅ Correct answer, correct source | 13 | 65% |
| ⚠️ Correct answer, wrong/partial source | 3 | 15% |
| ❌ Hallucinated answer | 2 | 10% |
| 🔍 Wrong retrieval | 1 | 5% |
| 🚫 Correctly said "I don't know" | 1 | 5% |

---

## Failure Type 1: Hallucination (2 cases)

**Query:** "What is the salary of the Chief Justice of India?"

**What happened:** The model confidently stated a specific salary figure that is not in the Constitution document. The Constitution specifies that salaries are "charged to the Consolidated Fund of India" but doesn't mention amounts — those are in separate parliamentary acts.

**Root cause:** The model's pretrained knowledge about judicial salaries bled through when retrieval didn't return a definitive answer.

**Fix applied:** Added explicit instruction to the prompt: *"If the answer is not in the context, say exactly: 'I could not find a relevant answer in the provided sections.' Do not use any knowledge outside the provided context."*

**Result:** After the fix, the model correctly says it can't find the answer in the document.

---

## Failure Type 2: Wrong Retrieval (1 case)

**Query:** "What are the powers of the Prime Minister?"

**What happened:** The retrieved chunks were mostly about the Council of Ministers collectively, not specifically the Prime Minister. The Constitution doesn't dedicate a standalone article to PM powers — they're distributed across Articles 74, 75, and 78.

**Root cause:** The query was too broad. The relevant content is spread across multiple non-adjacent sections, and chunk boundaries split the context badly.

**Fix applied:** Reduced chunk overlap to better capture article boundaries. Also increased TOP_K from 3 to 5 so more distributed content gets picked up.

**Result:** Retrieval improved, though still imperfect for multi-article questions.

---

## Failure Type 3: Correct Answer, Wrong Source (3 cases)

**Query:** "What does Article 19 say?"

**What happened:** The answer was accurate, but the system cited page 22 when the relevant content was also on page 21. The chunk on page 21 had slightly higher similarity but was truncated mid-article by the chunking boundary.

**Root cause:** Fixed-size character chunking doesn't respect article boundaries in legal documents.

**Fix I'd implement next:** Sentence-aware or paragraph-aware chunking that preserves article boundaries. The Constitution has a predictable structure (Article number → title → clauses) that a smarter chunker could exploit.

---

## What I'd change with more time

1. **Semantic chunking:** Split at article boundaries rather than fixed character counts. The Constitution's structure is very regular — this is very doable.

2. **Re-ranking:** After initial retrieval, run a cross-encoder reranker to reorder the top-10 chunks before passing them to the generator. This would significantly improve the wrong-retrieval cases.

3. **Query expansion:** For vague queries like "PM powers", use the LLM to rewrite the query into 2-3 specific sub-queries before retrieving.

4. **Confidence calibration:** The retrieval score is a reasonable proxy for confidence, but it doesn't correlate well with answer correctness. A dedicated faithfulness checker (like running a second LLM call to verify the answer against the context) would be more reliable.

---

The 65% accuracy on the first pass is honestly lower than I hoped, but the failure analysis is what makes this project interesting. The fixes I applied brought it to ~78%, and I now have a clear roadmap for reaching 90%+.
