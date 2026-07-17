import pytest

from samvidhan import generate
from samvidhan.generate import REFUSAL, GenerationError, build_context, build_messages
from samvidhan.retrieve import Retrieved

HITS = [
    Retrieved(text="No person shall be deprived of his life or personal liberty.", page=4,
              article="21", score=0.82),
    Retrieved(text="All citizens shall have the right to freedom of speech.", page=3,
              article="19", score=0.55),
]


def test_context_includes_article_and_page_tags():
    ctx = build_context(HITS)
    assert "Article 21" in ctx and "Page 4" in ctx
    assert "Article 19" in ctx and "Page 3" in ctx


def test_messages_are_system_then_user_and_carry_the_guard():
    msgs = build_messages("What does Article 21 say?", HITS)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert REFUSAL in msgs[0]["content"]
    assert "What does Article 21 say?" in msgs[1]["content"]


def test_generate_without_token_raises_generation_error(monkeypatch):
    # settings is a frozen dataclass, so swap the whole reference for a tokenless stub.
    class _NoToken:
        hf_token = ""
        has_token = False
        request_timeout = 60

    monkeypatch.setattr(generate, "settings", _NoToken())
    generate._client = None
    with pytest.raises(GenerationError):
        generate.generate("q", HITS)
