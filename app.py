"""SPARC-LTM — Real System UI.

Run:
    streamlit run app.py

Every message you type is processed by the full Phase 2 pipeline:
  - LLM fact extraction (meta-llama/llama-3.1-8b-instruct via OpenRouter)
  - Salience scoring
  - Contradiction/supersession detection
  - Persistent SQLite storage
  - Automatic lifecycle (compress/forget at capacity)

Every question retrieves from real persistent memory and generates
a grounded answer via OpenRouter.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from collections import deque

import streamlit as st

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import security utilities
from locomo_memory.security.validators import (
    InputValidator,
    APIKeyValidator,
    ValidationError,
)
from locomo_memory.security.sanitizers import InputSanitizer, LogSanitizer
from locomo_memory.security.rate_limiter import RateLimiter, RateLimitConfig, RateLimitExceeded

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SPARC-LTM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
div[data-testid="stSidebarContent"] .stButton button {
    width: 100%; text-align: left; background: transparent;
    border: none; padding: 0.4rem 0.6rem; border-radius: 6px; font-size: 0.9rem;
}
div[data-testid="stSidebarContent"] .stButton button:hover { background: rgba(255,255,255,0.08); }
.hit-card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px; padding: 0.6rem 0.8rem; margin-bottom: 0.4rem; font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
from locomo_memory.system.engine import SystemEngine, ProcessResult, AskResult  # noqa: E402
from locomo_memory.phase2.schemas import MemoryStatus  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TIER = {
    MemoryStatus.ACTIVE:     ("🟢", "Active"),
    MemoryStatus.COMPRESSED: ("🟡", "Compressed"),
    MemoryStatus.FORGOTTEN:  ("⚫", "Forgotten"),
    MemoryStatus.ARCHIVED:   ("🔷", "Archived"),
    MemoryStatus.DELETED:    ("🔴", "Deleted"),
}
_DB_PATH = os.environ.get("SPARC_DB_PATH", "data/system/memory.db")
_MAX_MESSAGES_PER_CHAT = 100  # Prevent unbounded memory growth
_MAX_INPUT_LENGTH = 2000  # Prevent DoS via large inputs

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init() -> None:
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "engine_error" not in st.session_state:
        st.session_state.engine_error = None
    # chats: {cid: {"name": str, "messages": deque, "session_id": str}}
    if "chats" not in st.session_state:
        cid = _cid()
        st.session_state.chats = {cid: {"name": "Chat 1", "messages": deque(maxlen=_MAX_MESSAGES_PER_CHAT), "session_id": "s1"}}
        st.session_state.active_chat = cid
    if "active_chat" not in st.session_state:
        chat_keys = list(st.session_state.chats.keys())
        st.session_state.active_chat = chat_keys[0] if chat_keys else _cid()
    if "view" not in st.session_state:
        st.session_state.view = "chat"
    # Initialize rate limiter
    if "rate_limiter" not in st.session_state:
        st.session_state.rate_limiter = RateLimiter(
            RateLimitConfig(max_requests=100, window_seconds=60.0)
        )


def _cid() -> str:
    return f"c_{uuid.uuid4().hex[:6]}"


def _engine() -> SystemEngine | None:
    if st.session_state.engine is not None:
        return st.session_state.engine
    if st.session_state.engine_error:
        return None

    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        st.session_state.engine_error = "OPENROUTER_API_KEY not set in .env"
        return None
    
    # Validate API key format
    try:
        key = APIKeyValidator.validate_openrouter_key(key)
    except ValidationError as e:
        st.session_state.engine_error = f"Invalid API key: {e}"
        return None

    with st.spinner("Starting SPARC-LTM… (loading embedding model)"):
        try:
            e = SystemEngine(db_path=_DB_PATH, api_key=key)
            st.session_state.engine = e
        except Exception as exc:
            # Sanitize error message before displaying
            error_msg = LogSanitizer.sanitize(str(exc))
            st.session_state.engine_error = error_msg
            return None
    return st.session_state.engine


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sidebar() -> None:
    engine = st.session_state.engine

    with st.sidebar:
        st.markdown("### 🧠 SPARC-LTM")
        st.caption("Real persistent memory system")
        st.divider()

        # New Chat
        if st.button("＋  New Chat", use_container_width=True):
            cid = _cid()
            n = len(st.session_state.chats) + 1
            sid = engine.new_session() if engine else f"s{n}"
            st.session_state.chats[cid] = {
                "name": f"Chat {n}",
                "messages": [],
                "session_id": sid,
            }
            st.session_state.active_chat = cid
            st.session_state.view = "chat"
            st.rerun()

        st.markdown("**Chats**")
        for cid, chat in list(st.session_state.chats.items()):
            is_active = cid == st.session_state.active_chat
            col_c, col_d = st.columns([6, 1])
            with col_c:
                label = ("▶ " if is_active else "   ") + chat["name"]
                if st.button(label, key=f"cb_{cid}", use_container_width=True,
                             type="primary" if is_active else "secondary"):
                    st.session_state.active_chat = cid
                    st.session_state.view = "chat"
                    st.rerun()
            with col_d:
                if len(st.session_state.chats) > 1:
                    if st.button("✕", key=f"del_{cid}"):
                        del st.session_state.chats[cid]
                        remaining_chats = list(st.session_state.chats.keys())
                        st.session_state.active_chat = remaining_chats[-1] if remaining_chats else _cid()
                        st.rerun()

        st.divider()

        # Live memory counts
        if engine:
            counts = engine.status_counts()
            st.markdown("**Memory**")
            c1, c2, c3 = st.columns(3)
            c1.metric("🟢", counts.get("active", 0), help="Active memories")
            c2.metric("🟡", counts.get("compressed", 0), help="Compressed")
            c3.metric("⚫", counts.get("forgotten", 0), help="Forgotten")
            pressure = engine.lifecycle_pressure()
            if pressure > 0:
                st.progress(min(pressure, 1.0), text=f"Capacity {pressure*100:.0f}%")

        st.divider()
        st.markdown("**Views**")
        _nav("🔍 Memory Inspector", "inspector")
        _nav("ℹ️  About", "about")


def _nav(label: str, view: str) -> None:
    active = st.session_state.view == view
    if st.button(label, key=f"nav_{view}", use_container_width=True,
                 type="primary" if active else "secondary"):
        st.session_state.view = view
        st.rerun()


# ---------------------------------------------------------------------------
# Chat view
# ---------------------------------------------------------------------------

def _view_chat() -> None:
    engine = _engine()
    if engine is None:
        st.error(f"Engine not ready: {st.session_state.engine_error}")
        st.info("Make sure `OPENROUTER_API_KEY` is set in `.env` and restart.")
        return

    cid = st.session_state.active_chat
    chat = st.session_state.chats[cid]

    # Editable name + generate toggle on same row
    col_name, col_gen = st.columns([5, 2])
    with col_name:
        new_name = st.text_input("", value=chat["name"],
                                 label_visibility="collapsed", key=f"name_{cid}")
        if new_name != chat["name"]:
            chat["name"] = new_name
    with col_gen:
        do_gen = st.toggle("✨ Generate answer", value=True, key=f"gen_{cid}",
                           help="Use OpenRouter to generate a grounded answer")

    messages = chat["messages"]

    if not messages:
        st.markdown(
            "<div style='text-align:center;padding:3rem 0;color:#64748b'>"
            "<h2>🧠 SPARC-LTM</h2>"
            "<p>Everything you say is remembered, understood, and retrievable.<br>"
            "Memories persist across all chats and app restarts.</p>"
            "</div>", unsafe_allow_html=True)
    else:
        for msg in messages:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🧠"):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    _render_process(msg.get("process"))
                    _render_hits(msg.get("hits", []), msg.get("retrieval_ms", 0))

    prompt = st.chat_input("Say something or ask a question…", key=f"inp_{cid}")
    if prompt:
        # Validate and sanitize input
        try:
            sanitized = InputValidator.validate_text_input(
                prompt, 
                max_length=_MAX_INPUT_LENGTH,
                field_name="message"
            )
            
            # Check rate limit
            try:
                st.session_state.rate_limiter.check_limit(cid)
            except RateLimitExceeded as e:
                st.error(f"⚠️ {str(e)}")
                return
            
            # Basic prompt injection check
            if not InputValidator.is_safe_for_llm(sanitized):
                st.warning("⚠️ Your message contains patterns that may not be processed correctly. Please rephrase.")
                return
            
            _handle_message(engine, cid, chat, sanitized, do_gen)
            st.rerun()
        except ValidationError as e:
            st.error(f"⚠️ {str(e)}")
        except Exception as e:
            st.error(f"⚠️ An error occurred. Please try again.")


def _handle_message(engine: SystemEngine, cid: str, chat: dict, text: str, do_gen: bool) -> None:
    chat["messages"].append({"role": "user", "content": text})
    sid = chat["session_id"]

    # Determine intent: statement vs question
    is_question = text.strip().endswith("?") or text.lower().startswith(
        ("what", "who", "where", "when", "how", "why", "tell me", "do i", "did i",
         "have i", "am i", "is my", "was my", "which")
    )

    process: ProcessResult | None = None
    hits = []
    answer = ""
    retrieval_ms = 0.0

    if is_question:
        # Question: retrieve + optionally generate answer
        result = engine.ask(text, session_id=sid, generate=do_gen)
        hits = result.hits
        retrieval_ms = result.retrieval_latency_ms
        answer = result.answer
        if not do_gen and hits and len(hits) > 0:
            # Safe access to first hit
            answer = f"Based on memory: **{hits[0].mu.claim}**"
    else:
        # Statement: ingest into memory
        process = engine.process_message(text, speaker="User", session_id=sid)
        n = len(process.extracted_mus)
        if n == 0:
            answer = "I didn't find any specific facts to remember from that message."
        else:
            facts = [f"• {mu.claim}" for mu in process.extracted_mus]
            facts_str = "\n".join(facts)
            answer = f"Stored **{n}** {'fact' if n == 1 else 'facts'} in memory:\n{facts_str}"
            if process.superseded_ids:
                answer += f"\n\n⚡ Updated {len(process.superseded_ids)} older fact(s) that this supersedes."
            if process.lifecycle and (process.lifecycle.n_compressed or process.lifecycle.n_forgotten):
                answer += (
                    f"\n\n🔄 Lifecycle: compressed {process.lifecycle.n_compressed}, "
                    f"forgot {process.lifecycle.n_forgotten} low-salience facts."
                )

    chat["messages"].append({
        "role": "assistant",
        "content": answer,
        "hits": hits,
        "process": process,
        "retrieval_ms": retrieval_ms,
    })

    if len(chat["messages"]) == 2 and chat["name"].startswith("Chat "):
        chat["name"] = text[:35] + ("…" if len(text) > 35 else "")


def _render_process(process: ProcessResult | None) -> None:
    if process is None or not process.extracted_mus:
        return
    with st.expander(f"🧠 {len(process.extracted_mus)} fact(s) extracted & stored", expanded=False):
        for mu in process.extracted_mus:
            icon, label = _TIER.get(mu.status, ("⚪", mu.status.value))
            st.markdown(
                f"<div class='hit-card'>"
                f"<b>{icon} {label}</b> "
                f"<span style='color:#94a3b8;font-size:0.8em'>salience={mu.salience_score:.2f} "
                f"· conf={mu.confidence:.2f}</span><br>{mu.claim}"
                f"</div>", unsafe_allow_html=True)
        if process.contradictions_found:
            st.warning(f"⚡ {process.contradictions_found} contradiction/update edge(s) created.")


def _render_hits(hits: list, retrieval_ms: float) -> None:
    if not hits:
        return
    with st.expander(f"📂 {len(hits)} memories retrieved  ·  {retrieval_ms:.0f} ms", expanded=False):
        for i, hit in enumerate(hits, 1):
            mu = hit.mu
            icon, label = _TIER.get(mu.status, ("⚪", mu.status.value))
            section = ""
            if hit.relation_meta.superseded_by:
                section = " · HISTORICAL"
            elif hit.relation_meta.conflicts_with:
                section = " · CONFLICTING"
            st.markdown(
                f"<div class='hit-card'>"
                f"<b>#{i} {icon} {label}{section}</b> "
                f"<span style='color:#94a3b8;font-size:0.8em'>session={mu.session_id} "
                f"· salience={mu.salience_score:.2f} · rrf={hit.rrf_score:.4f}</span><br>"
                f"{mu.claim}"
                f"</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Memory Inspector
# ---------------------------------------------------------------------------

def _view_inspector() -> None:
    st.markdown("## 🔍 Memory Inspector")
    engine = _engine()
    if engine is None:
        st.error("Engine not ready.")
        return

    counts = engine.status_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Active", counts.get("active", 0))
    c2.metric("🟡 Compressed", counts.get("compressed", 0))
    c3.metric("⚫ Forgotten", counts.get("forgotten", 0))

    pressure = engine.lifecycle_pressure()
    if pressure > 0:
        st.progress(min(pressure, 1.0), text=f"Capacity {pressure*100:.0f}%  (compression fires at 90%)")

    st.divider()

    sel = st.multiselect("Show layers", ["active", "compressed", "forgotten", "archived", "deleted"],
                         default=["active", "compressed", "forgotten"])

    all_mus = engine.get_memories()
    filtered = [m for m in all_mus if m.status.value in sel]

    if not filtered:
        st.info("No memories match. Try different layer filters.")
        return

    for mu in filtered:
        icon, label = _TIER.get(mu.status, ("⚪", mu.status.value))
        with st.expander(
            f"{icon} [{label}]  {mu.claim[:70]}{'…' if len(mu.claim)>70 else ''}",
            expanded=False,
        ):
            col_l, col_r = st.columns(2)
            col_l.code(mu.mu_id)
            col_l.markdown(f"**Session:** `{mu.session_id}`  |  **Speaker:** {mu.source_speaker}")
            col_l.markdown(f"**Claim:** {mu.claim}")
            col_r.markdown(f"**Salience:** {mu.salience_score:.2f}  |  **Confidence:** {mu.confidence:.2f}")
            col_r.markdown(f"**Retrieval count:** {mu.retrieval_count}")
            if mu.compressed_label_id:
                col_r.markdown(f"**Label:** `{mu.compressed_label_id}`")


# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------

def _view_about() -> None:
    st.markdown("## ℹ️ SPARC-LTM — Real System")
    st.markdown("""
**SPARC-LTM** = Salience & Provenance Aware Reconciliation and Compression for Long-Term Memory

### What happens when you type a statement
1. **Fact Extractor** (LLM — claude-3-haiku) reads your message and extracts atomic claims
2. **Salience Scorer** scores each claim by importance, recency, confidence, retrieval frequency
3. **SQLite store** persists every fact — survives restarts
4. **Contradiction Resolver** checks if any new fact contradicts or supersedes an older one
5. **FAISS + BM25 index** makes the fact searchable immediately
6. **Lifecycle Engine** auto-compresses / forgets low-salience facts when capacity reaches 90%

### What happens when you ask a question
1. **Hybrid Retriever** (FAISS dense + BM25 sparse fusion) finds top-5 relevant memories
2. **Context Builder** organises them into: Active / Historical (superseded) / Conflicting
3. **OpenRouter LLM** (claude-3-haiku) generates a grounded answer from the evidence

### Memory layers
| Layer | Meaning |
|-------|---------|
| 🟢 Active | In working memory — retrieved by default |
| 🟡 Compressed | Low-salience: short label stored, full text archived |
| ⚫ Forgotten | Very low salience: hidden from retrieval, restorable |

### Storage
All memories are stored in `data/system/memory.db` (persistent SQLite).
LLM responses are cached in `data/system/llm_cache/` to save API cost.
""")
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        # Mask API key for security
        masked_key = APIKeyValidator.mask_key(key, visible_chars=4)
        st.success(f"API key loaded ({masked_key})")
    else:
        st.error("OPENROUTER_API_KEY not set")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _init()

    # Ensure engine loads on first run
    _engine()

    _sidebar()

    view = st.session_state.get("view", "chat")
    if view == "chat":
        _view_chat()
    elif view == "inspector":
        _view_inspector()
    elif view == "about":
        _view_about()
    else:
        _view_chat()


if __name__ == "__main__":
    main()
