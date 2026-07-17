"""
Generation: turn retrieved context into a grounded answer via the HF Inference API.

The prompt is deliberately strict — the model may only use the supplied context
and must emit a fixed refusal string when the answer isn't there. This is the
single most effective guard against hallucination (see FAILURES.md, Failure #1).

Generation is optional: if no ``HF_TOKEN`` is set the pipeline degrades to
retrieval-only, still showing the user the exact articles that matched.
"""

from __future__ import annotations

from .config import settings
from .retrieve import Retrieved

REFUSAL = "I could not find a relevant answer in the provided sections."

SYSTEM_PROMPT = (
    "You are a precise assistant answering questions about the Constitution of "
    "India. Use ONLY the numbered context passages provided. Do not use outside "
    "knowledge. Cite the article number(s) you relied on. If the answer is not "
    f"present in the context, reply with exactly: '{REFUSAL}'"
)

_client = None


class GenerationError(RuntimeError):
    """Raised when the inference call fails (bad token, model cold, network)."""


def _get_client():
    global _client
    if _client is None:
        if not settings.has_token:
            raise GenerationError(
                "HF_TOKEN is not set. Export a free token from "
                "https://huggingface.co/settings/tokens to enable answer generation."
            )
        from huggingface_hub import InferenceClient

        _client = InferenceClient(token=settings.hf_token, timeout=settings.request_timeout)
    return _client


def build_context(hits: list[Retrieved]) -> str:
    lines = []
    for h in hits:
        tag = f"Article {h.article}, " if h.article else ""
        lines.append(f"[{tag}Page {h.page}] {h.text}")
    return "\n\n".join(lines)


def build_messages(query: str, hits: list[Retrieved]) -> list[dict]:
    context = build_context(hits)
    user = f"Context:\n{context}\n\nQuestion: {query}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def generate(query: str, hits: list[Retrieved]) -> str:
    """Call the inference API and return the answer text. Raises GenerationError."""
    client = _get_client()
    try:
        resp = client.chat_completion(
            messages=build_messages(query, hits),
            model=settings.gen_model,
            max_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
        )
        return resp.choices[0].message.content.strip()
    except GenerationError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any client/transport error uniformly
        raise GenerationError(f"Inference call failed: {exc}") from exc
