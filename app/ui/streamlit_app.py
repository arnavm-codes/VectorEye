"""Streamlit demo UI: type a query, get back the matching 10s clips.

Talks directly to the retrieval code (no FastAPI process needed for the
demo) -- embedding-based Qdrant similarity search only (backend model
swappable via EMBEDDING_BACKEND in app/config.py), with an optional Groq
"chat mode" for query cleanup + a conversational summary. Groq never
touches clip content, only the text query and already-retrieved metadata.

Run: uv run streamlit run app/ui/streamlit_app.py
Prereq: Qdrant running (docker compose up -d) and clips already indexed
(uv run python scripts/run_pipeline.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from app.config import ENABLE_AUDIO_SEARCH, GROQ_API_KEY, QDRANT_COLLECTION

st.set_page_config(page_title="Video Library RAG (POC)", page_icon="🎥", layout="wide")


@st.cache_resource(show_spinner="Loading models (first run only, then cached for this session)...")
def _warm_models():
    """Loads all search-time models up front instead of lazily on first
    query, so the first search a user runs isn't the one that eats a ~15-17s
    model-load stall (the sentence-transformer used for transcript search is
    the slow one -- see vault note's performance measurements, 2026-09-03).
    Cached via st.cache_resource, so this only actually runs once per
    Streamlit server process, however many browser sessions connect to it.
    """
    from app.pipeline.embedder import _load_model as load_visual_model

    load_visual_model()

    if ENABLE_AUDIO_SEARCH:
        from app.pipeline.text_embedder import _load_model as load_text_embed_model

        load_text_embed_model()
        # Whisper is indexing-only (never called during search), so it isn't
        # warmed here -- pre-loading it wouldn't speed up anything a user of
        # this UI actually does.

    return True


_warm_models()


def _qdrant_status() -> tuple[bool, str]:
    try:
        from app.pipeline.indexer import get_client
        client = get_client()
        if not client.collection_exists(QDRANT_COLLECTION):
            return False, f"Qdrant is up, but collection '{QDRANT_COLLECTION}' doesn't exist yet. Run the indexing pipeline first."
        count = client.count(QDRANT_COLLECTION).count
        if count == 0:
            return False, "Collection exists but has 0 clips indexed. Run scripts/run_pipeline.py first."
        return True, f"{count} clips indexed."
    except Exception as exc:
        return False, f"Can't reach Qdrant ({exc}). Run `docker compose up -d` first."


def _list_camera_ids() -> list[str]:
    try:
        from app.pipeline.indexer import get_client
        client = get_client()
        points, _ = client.scroll(QDRANT_COLLECTION, limit=1000, with_payload=True)
        return sorted({p.payload.get("camera_id") for p in points if p.payload.get("camera_id")})
    except Exception:
        return []


st.title("🎥 Video Library RAG — POC")
st.caption(
    "Retrieval-only: embedding-based similarity search via Qdrant. "
    "No LLM ever watches the footage."
)

ready, status_msg = _qdrant_status()
(st.success if ready else st.warning)(status_msg)

with st.sidebar:
    st.header("Options")
    top_k = st.slider("Results to show", min_value=1, max_value=10, value=5)

    camera_ids = _list_camera_ids() if ready else []
    camera_filter = st.selectbox("Camera filter", options=["All cameras"] + camera_ids)
    camera_filter = None if camera_filter == "All cameras" else camera_filter

    use_chat_mode = st.toggle(
        "Groq chat mode (query cleanup + summary)",
        value=False,
        disabled=not GROQ_API_KEY,
        help="Requires GROQ_API_KEY in .env. Groq only sees the text query "
        "and retrieved metadata, never the video itself.",
    )
    if not GROQ_API_KEY:
        st.caption("Set GROQ_API_KEY in .env to enable chat mode.")

    use_transcript_fusion = st.toggle(
        "Fuse in speech content (experimental)",
        value=False,
        disabled=not ENABLE_AUDIO_SEARCH,
        help="Combines the visual CLIP ranking with a ranking over clips' "
        "transcribed speech (Reciprocal Rank Fusion), for queries about "
        "what was said rather than what's visible. Only affects clips "
        "where speech was detected during indexing.",
    )
    if not ENABLE_AUDIO_SEARCH:
        st.caption("Set ENABLE_AUDIO_SEARCH=true and re-index to enable this.")

query = st.text_input(
    "Search query",
    placeholder='e.g. "a blue car waiting at the gate"',
)
run_search = st.button("Search", type="primary", disabled=not ready)

if run_search and query.strip():
    from app.api.search import search_clips

    search_query = query
    summary = None

    if use_chat_mode:
        from app.chat.groq_wrapper import clean_query, explain_results

        with st.spinner("Cleaning up query with Groq..."):
            search_query = clean_query(query)
        st.caption(f"Searched as: _{search_query}_")

    with st.spinner("Searching..."):
        results = search_clips(
            search_query,
            top_k=top_k,
            camera_id=camera_filter,
            use_transcript_fusion=use_transcript_fusion,
        )

    if use_chat_mode:
        with st.spinner("Summarizing with Groq..."):
            summary = explain_results(query, results)
        st.info(summary)

    if not results:
        st.warning("No matching clips found.")
    else:
        for r in results:
            with st.container(border=True):
                cols = st.columns([2, 1])
                with cols[0]:
                    st.video(r["clip_path"])
                with cols[1]:
                    st.metric("Similarity score", f"{r['score']:.3f}")
                    st.write(f"**Camera:** {r['camera_id']}")
                    st.write(f"**Time range:** {r['start_ts']}s - {r['end_ts']}s")
                    if r.get("has_speech"):
                        st.write(f"**Transcript:** _{r['transcript']}_")
                    st.caption(r["clip_path"])
