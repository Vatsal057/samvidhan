from samvidhan import pipeline
from samvidhan.generate import GenerationError
from samvidhan.retrieve import Retrieved

HITS = [
    Retrieved(text="…life or personal liberty…", page=4, article="21", score=0.80),
    Retrieved(text="…freedom of speech…", page=3, article="19", score=0.60),
]


def test_answer_generates_when_llm_available(monkeypatch):
    monkeypatch.setattr(pipeline, "retrieve", lambda q: HITS)
    monkeypatch.setattr(pipeline, "generate", lambda q, h: "Article 21 protects life and liberty.")
    r = pipeline.answer("What does Article 21 say?")
    assert r.generated is True
    assert r.answer.startswith("Article 21")
    assert abs(r.retrieval_score - 0.70) < 1e-9  # mean of 0.80 and 0.60


def test_answer_degrades_to_retrieval_only_on_generation_error(monkeypatch):
    monkeypatch.setattr(pipeline, "retrieve", lambda q: HITS)

    def boom(q, h):
        raise GenerationError("model cold")

    monkeypatch.setattr(pipeline, "generate", boom)
    r = pipeline.answer("q")
    assert r.generated is False
    assert r.sources == HITS
    assert "model cold" in r.answer


def test_answer_handles_empty_knowledge_base(monkeypatch):
    monkeypatch.setattr(pipeline, "retrieve", lambda q: [])
    r = pipeline.answer("q")
    assert r.generated is False
    assert r.retrieval_score == 0.0
    assert r.sources == []
