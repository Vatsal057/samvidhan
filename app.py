"""
Streamlit UI for Samvidhan — Constitution of India RAG.

    streamlit run app.py

Set HF_TOKEN (env var or the sidebar box) to enable grounded answer generation.
Without it, the app runs in retrieval-only mode and still shows the exact
articles that match your question.
"""

import os
import sys
from pathlib import Path

# Make the src/ package importable without an editable install (keeps HF Spaces simple).
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st  # noqa: E402

from samvidhan import pipeline, retrieve  # noqa: E402
from samvidhan.config import settings  # noqa: E402

st.set_page_config(
    page_title="Samvidhan · Constitution Q&A",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;1,400&family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family:'Inter',sans-serif; background:#faf9f7; color:#1c1917; }
.title { font-family:'EB Garamond',serif; font-size:2.6rem; font-weight:600; line-height:1.1; }
.subtitle { color:#78716c; font-size:.9rem; margin-top:2px; margin-bottom:1rem; }
.answer-box { background:#fff; color:#1c1917; border:1px solid #e7e5e4; border-left:4px solid #1c4ed8;
  border-radius:6px; padding:1.2rem 1.4rem; font-size:.95rem; line-height:1.7; white-space:pre-wrap; }
.source-card, .source-card div { color:#1c1917; }
.answer-box.retrieval-only { border-left-color:#d97706; }
.source-card { background:#f5f5f4; border:1px solid #e7e5e4; border-radius:6px;
  padding:.8rem 1rem; margin-bottom:.5rem; font-size:.82rem; }
.page-badge { display:inline-block; background:#1c4ed8; color:#fff; border-radius:3px;
  padding:1px 7px; font-size:.7rem; font-family:'DM Mono',monospace; margin-right:6px; }
.article-badge { display:inline-block; background:#0f766e; color:#fff; border-radius:3px;
  padding:1px 7px; font-size:.7rem; font-family:'DM Mono',monospace; margin-right:6px; }
.score-pill { display:inline-block; background:#f0fdf4; color:#166534; border:1px solid #bbf7d0;
  border-radius:20px; padding:1px 8px; font-size:.68rem; font-family:'DM Mono',monospace; }
.label { font-size:.72rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase;
  color:#a8a29e; margin-bottom:4px; }
.user-bubble { background:#1c4ed8; color:#fff; border-radius:14px 14px 4px 14px; padding:.7rem 1rem;
  margin-bottom:.5rem; font-size:.9rem; max-width:80%; margin-left:auto; }
div[data-testid="stButton"]>button { background:#1c4ed8; color:#fff; border:none; border-radius:6px;
  font-size:.85rem; font-weight:600; padding:.5rem 1.4rem; transition:background .15s; }
div[data-testid="stButton"]>button:hover { background:#1d4ed8cc; }
hr { border-color:#e7e5e4; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading knowledge base…")
def boot():
    try:
        retrieve.get_embedder()
        try:
            retrieve.get_collection()
        except Exception:  # noqa: BLE001 - first boot: build the index from the sample
            from samvidhan.ingest import ingest

            ingest(text="data/sample")
            retrieve._collection = None  # force a fresh handle after ingest
            retrieve.get_collection()
        return True
    except Exception as e:  # noqa: BLE001
        return str(e)


status = boot()

with st.sidebar:
    st.markdown("### ⚖️ About")
    st.markdown(
        "Ask anything about the **Constitution of India**. The system retrieves the "
        "most relevant articles, then generates a grounded answer that cites its sources."
    )
    st.divider()
    st.markdown("### 🔑 HuggingFace token")
    token = st.text_input(
        "HF_TOKEN", type="password", value=os.getenv("HF_TOKEN", ""), placeholder="hf_…"
    )
    if token:
        os.environ["HF_TOKEN"] = token
    if not settings.has_token and not token:
        st.caption("No token → retrieval-only mode (sources shown, no generated answer).")
    st.divider()
    st.markdown("### 💡 Sample questions")
    examples = [
        "What does Article 21 guarantee?",
        "How is the President of India elected?",
        "What is the procedure for amending the Constitution?",
        "What rights does Article 19 protect?",
        "How many members are in the Rajya Sabha?",
        "Is untouchability legal in India?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state["prefill"] = ex

st.markdown('<div class="title">⚖️ Samvidhan</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Retrieval-Augmented Q&A over the Constitution of India · '
    "article-aware retrieval · grounded, cited answers</div>",
    unsafe_allow_html=True,
)
st.divider()

if isinstance(status, str):
    st.error(
        "**Knowledge base not found.** Build it first:\n\n"
        "```bash\npython -m samvidhan.ingest --text data/sample\n```\n\n"
        f"Details: {status}"
    )
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

prefill = st.session_state.pop("prefill", "")
col_q, col_b = st.columns([5, 1], vertical_alignment="bottom")
with col_q:
    query = st.text_input(
        "question", value=prefill, placeholder="Ask anything about the Indian Constitution…",
        label_visibility="collapsed",
    )
with col_b:
    ask = st.button("Ask →", use_container_width=True)

if ask and query.strip():
    with st.spinner("Retrieving and generating…"):
        st.session_state.history.append({"q": query.strip(), "r": pipeline.answer(query.strip())})

for item in reversed(st.session_state.history):
    q, r = item["q"], item["r"]
    st.markdown(f'<div class="user-bubble">{q}</div>', unsafe_allow_html=True)
    col_ans, col_src = st.columns([3, 2])
    with col_ans:
        st.markdown('<div class="label">Answer</div>', unsafe_allow_html=True)
        cls = "answer-box" if r.generated else "answer-box retrieval-only"
        st.markdown(f'<div class="{cls}">{r.answer}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="margin-top:6px"><span class="score-pill">'
            f"avg retrieval score: {r.retrieval_score:.3f}</span></div>",
            unsafe_allow_html=True,
        )
    with col_src:
        st.markdown('<div class="label">Sources retrieved</div>', unsafe_allow_html=True)
        for s in r.sources:
            snippet = s.text[:220].replace("\n", " ") + "…"
            art = f'<span class="article-badge">Art {s.article}</span>' if s.article else ""
            st.markdown(
                f'<div class="source-card">{art}'
                f'<span class="page-badge">Page {s.page}</span>'
                f'<span class="score-pill">{s.score:.3f}</span>'
                f'<div style="margin-top:6px;color:#57534e">{snippet}</div></div>',
                unsafe_allow_html=True,
            )
    st.divider()

if not st.session_state.history:
    st.markdown(
        '<div style="color:#a8a29e;text-align:center;padding:3rem 0">'
        "Ask a question to get started — try one of the examples in the sidebar.</div>",
        unsafe_allow_html=True,
    )
