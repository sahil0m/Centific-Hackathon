"""SPARC-LTM — Real System UI.

Run:
    streamlit run app.py

Every message you type is processed by the full Phase 2 pipeline:
  - LLM fact extraction (anthropic/claude-3-haiku via OpenRouter)
  - Salience scoring (Ebbinghaus + topic importance + graph penalty)
  - Contradiction / supersession detection (NLI + rule patterns)
  - Persistent SQLite storage with provenance edges
  - Automatic lifecycle (compress / forget at 90 % capacity)

Every question retrieves from real persistent memory (FAISS + BM25 + label
search + forgotten worker, fused via RRF) and generates a grounded answer
via the same OpenRouter Claude-3-haiku model.
"""

from __future__ import annotations

import html
import os
import sys
import uuid
from pathlib import Path
from collections import deque

import streamlit as st
from loguru import logger

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
from locomo_memory.security.sanitizers import LogSanitizer
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
/* Sidebar chat-list buttons — force readable text regardless of theme */
div[data-testid="stSidebarContent"] .stButton button {
    width: 100%; text-align: left; background: transparent;
    border: none; padding: 0.4rem 0.6rem; border-radius: 6px; font-size: 0.88rem;
    color: #1e293b !important;
}
div[data-testid="stSidebarContent"] .stButton button:hover {
    background: rgba(0,0,0,0.06) !important;
    color: #0f172a !important;
}
/* Active chat button — tinted highlight */
div[data-testid="stSidebarContent"] .stButton button[kind="primaryFormSubmit"],
div[data-testid="stSidebarContent"] .stButton button[data-testid="baseButton-primary"] {
    background: rgba(99,102,241,0.12) !important;
    color: #1e293b !important;
    font-weight: 600;
}
.hit-card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px; padding: 0.6rem 0.8rem; margin-bottom: 0.4rem; font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
from locomo_memory.system.engine import SystemEngine, ProcessResult  # noqa: E402
from locomo_memory.phase2.schemas import MemoryStatus  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TIER = {
    MemoryStatus.ACTIVE:     ("🟢", "Active"),
    MemoryStatus.COMPRESSED: ("🟡", "Compressed (label)"),  # legacy label-only rows
    MemoryStatus.ARCHIVED:   ("🔷", "Compressed"),           # original data layer (new design)
    MemoryStatus.FORGOTTEN:  ("⚫", "Forgotten"),
}
_DB_PATH = os.environ.get("SPARC_DB_PATH", "data/system/memory.db")
_MAX_MESSAGES_PER_CHAT = 100  # Prevent unbounded memory growth
_MAX_INPUT_LENGTH = 2000  # Prevent DoS via large inputs
_DEFAULT_CAP = 500  # Default active memory cap


def _cap() -> int:
    """Return the current active memory cap from session state."""
    return int(st.session_state.get("active_cap", _DEFAULT_CAP))


def _engine_cap() -> int:
    """Return the cap the running engine is actually enforcing.

    Falls back to session-state cap when engine is not yet initialised.
    This is the authoritative value to display in the memory tracker bar.
    """
    eng = st.session_state.get("engine")
    if eng is not None:
        try:
            return eng.lifecycle.config.active_cap
        except Exception as exc:
            logger.debug("_engine_cap: could not read engine cap: {}", exc)
    return _cap()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _reset_engine() -> None:
    """Destroy the current engine + clear the DB so the demo starts fresh."""
    st.session_state.engine = None
    st.session_state.engine_error = None
    # Clear chat histories
    for chat in st.session_state.get("chats", {}).values():
        chat["messages"] = deque(maxlen=_MAX_MESSAGES_PER_CHAT)
    # Delete the DB so memory is wiped
    try:
        Path(_DB_PATH).unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("_reset_engine: could not unlink DB at {}: {}", _DB_PATH, exc)


def _init() -> None:
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "engine_error" not in st.session_state:
        st.session_state.engine_error = None
    if "active_cap" not in st.session_state:
        st.session_state.active_cap = _DEFAULT_CAP
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
            e = SystemEngine(db_path=_DB_PATH, api_key=key, active_cap=_cap())
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
            # Use the same bounded deque as _init() so message history can't
            # grow without limit on chats created after the first.
            st.session_state.chats[cid] = {
                "name": f"Chat {n}",
                "messages": deque(maxlen=_MAX_MESSAGES_PER_CHAT),
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
                        # Pop chat-scoped widget keys so session_state doesn't
                        # accumulate orphaned entries across long sessions.
                        for _key_prefix in ("name_", "gen_", "inp_", "cb_", "del_"):
                            st.session_state.pop(f"{_key_prefix}{cid}", None)
                        remaining_chats = list(st.session_state.chats.keys())
                        st.session_state.active_chat = remaining_chats[-1] if remaining_chats else _cid()
                        st.rerun()

        st.divider()

        # Live memory counts
        if engine:
            counts = engine.status_counts()
            active = counts.get("active", 0)
            compressed = counts.get("compressed", 0)
            archived = counts.get("archived", 0)
            forgotten = counts.get("forgotten", 0)
            # Always read the cap straight off the running engine so the
            # display can never disagree with what the engine is enforcing.
            sb_cap = _engine_cap()
            pressure = engine.lifecycle_pressure()
            pct = min(pressure * 100, 100)

            st.markdown("**Memory Context**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🟢 Active", active)
            c2.metric("🔷 Comp.", compressed)
            c3.metric("📦 Arch.", archived)
            c4.metric("⚫ Forg.", forgotten)

            if pct >= 90:
                bar_color, label = "#ef4444", f"⚠️ {pct:.0f}% of {sb_cap} — Auto-managing"
            elif pct >= 70:
                bar_color, label = "#f59e0b", f"🟡 {pct:.0f}% of {sb_cap} — Filling"
            else:
                bar_color, label = "#22c55e", f"🟢 {active} / {sb_cap} ({pct:.0f}%)"

            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.06);border-radius:4px;height:8px;overflow:hidden;margin:4px 0">
              <div style="width:{pct:.1f}%;background:{bar_color};height:100%;border-radius:4px"></div>
            </div>
            <div style="font-size:0.75rem;color:#94a3b8;margin-top:2px">{label}</div>
            <div style="font-size:0.7rem;color:#64748b;margin-top:1px">
              Fires at 90% ({sb_cap * 90 // 100}) · Targets 70% ({sb_cap * 70 // 100})
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown("**Views**")
        _nav("🔍 Memory Inspector", "inspector")
        _nav("ℹ️  About", "about")

        st.divider()
        st.markdown("**⚙️ Demo Mode**")

        # Surface what the engine is actually enforcing right now.
        if engine is not None:
            running_cap = _engine_cap()
            try:
                forget_th = engine.lifecycle.config.salience_forget_threshold
                compress_th = engine.lifecycle.config.salience_compress_threshold
            except Exception:
                forget_th = compress_th = None
            if running_cap <= 20:
                st.success(
                    f"⚡ Engine running with **cap = {running_cap}** · "
                    f"Lifecycle fires at **{running_cap * 90 // 100}** facts"
                )
                if forget_th is not None:
                    st.caption(
                        f"📊 **Demo thresholds active**: salience < {forget_th:.2f} → ⚫ Forgotten · "
                        f"salience ≥ {forget_th:.2f} → 🔷 Compressed"
                    )
            else:
                st.caption(
                    f"⚡ Engine running with cap = {running_cap} · "
                    f"Lifecycle fires at {running_cap * 90 // 100}"
                )
                if forget_th is not None:
                    st.caption(
                        f"📊 Prod thresholds: salience < {forget_th:.2f} → ⚫ Forgotten · "
                        f"salience ≥ {forget_th:.2f} → 🔷 Compressed"
                    )
        else:
            st.caption("Lower the cap to see lifecycle fire fast.")

        # Two big preset buttons.  Use ``on_click`` callbacks rather than
        # checking ``if st.button(...):`` because Streamlit forbids writing to
        # a widget-owned ``session_state`` key (here, ``cap_input``) once the
        # widget has been registered in any prior run — and the inline-handler
        # pattern hits exactly that error.  Callbacks fire BEFORE the script
        # rerun, where state writes are always permitted.
        def _set_demo_cap() -> None:
            st.session_state.active_cap = 10
            st.session_state.cap_input = 10
            _reset_engine()
            st.toast("✅ Demo cap = 10 applied · memory reset")

        def _set_prod_cap() -> None:
            st.session_state.active_cap = 500
            st.session_state.cap_input = 500
            _reset_engine()
            st.toast("✅ Prod cap = 500 applied · memory reset")

        def _on_cap_input_change() -> None:
            """Auto-apply: number_input's on_change fires before the rerun."""
            try:
                new_val = int(st.session_state.cap_input)
            except (TypeError, ValueError):
                return
            if new_val != st.session_state.active_cap:
                st.session_state.active_cap = new_val
                _reset_engine()
                st.toast(f"✅ Cap changed to {new_val} — memory reset")

        col_demo, col_prod = st.columns(2)
        with col_demo:
            st.button(
                "🚀 Demo\n(cap=10)",
                use_container_width=True,
                help="Set cap=10 — lifecycle fires after ~9 messages",
                type="primary" if _engine_cap() != 10 else "secondary",
                on_click=_set_demo_cap,
                key="btn_demo_cap",
            )
        with col_prod:
            st.button(
                "🏭 Prod\n(cap=500)",
                use_container_width=True,
                help="Restore default cap=500",
                type="primary" if _engine_cap() != 500 else "secondary",
                on_click=_set_prod_cap,
                key="btn_prod_cap",
            )

        # Custom cap with auto-apply via on_change.  The widget owns
        # ``cap_input``; the callback updates ``active_cap`` (a plain
        # session-state key, not widget-owned, so writing is fine).
        st.number_input(
            "Custom cap (auto-applies on Enter)",
            min_value=5, max_value=500,
            value=st.session_state.active_cap,
            step=1,
            key="cap_input",
            on_change=_on_cap_input_change,
            help=(
                "Type a value and press Enter to apply.  This will reset memory "
                "(wipes the DB) so the new cap takes effect cleanly."
            ),
        )

        st.divider()
        if st.button("🗑️ Reset Memory", use_container_width=True,
                     help="Wipe all stored memories and restart engine (keeps cap)"):
            _reset_engine()
            st.rerun()


def _nav(label: str, view: str) -> None:
    active = st.session_state.view == view
    if st.button(label, key=f"nav_{view}", use_container_width=True,
                 type="primary" if active else "secondary"):
        st.session_state.view = view
        st.rerun()


# ---------------------------------------------------------------------------
# Memory context tracker
# ---------------------------------------------------------------------------

def _render_memory_tracker() -> None:
    """Memory fill bar + problem-solved badges."""
    engine = st.session_state.engine
    if engine is None:
        return

    # Sanity check — if session-state cap has drifted from the engine's cap,
    # surface a loud warning rather than letting the bar silently report
    # different numbers than the user expects.  This should never trigger
    # under normal use because both Demo/Prod buttons and the auto-apply
    # number_input set both keys atomically, but a bug or a manual session
    # state edit could cause drift.
    _session_cap = int(st.session_state.get("active_cap", _DEFAULT_CAP))
    _engine_cap_v = engine.lifecycle.config.active_cap
    if _session_cap != _engine_cap_v:
        st.warning(
            f"⚠️ Cap mismatch: sidebar shows **{_session_cap}**, engine is enforcing "
            f"**{_engine_cap_v}**.  Click Demo or Prod (or change the custom cap) "
            f"to reset and sync.",
            icon="⚠️",
        )

    counts = engine.status_counts()
    active     = counts.get("active", 0)
    compressed = counts.get("compressed", 0)
    archived   = counts.get("archived", 0)
    forgotten  = counts.get("forgotten", 0)

    # Always use the engine's actual cap — session-state cap may lag by one
    # render cycle after a Demo/Prod button click that recreates the engine.
    ec = max(_engine_cap(), 1)
    pressure = engine.lifecycle_pressure()
    pct = min(pressure * 100, 100)

    if pct >= 90:
        status_html = (
            "<span style='color:#ef4444;font-weight:600'>⚠️ Memory at capacity — "
            "auto-compressing low-salience facts &amp; forgetting very-low-salience facts</span>"
        )
    elif pct >= 70:
        status_html = (
            f"<span style='color:#f59e0b'>🟡 Memory filling — "
            f"lifecycle fires at 90% ({ec * 90 // 100} active facts)</span>"
        )
    else:
        status_html = "<span style='color:#64748b'>Memory healthy</span>"

    def _bar_pct(n: int) -> str:
        return f"{max(min(n / ec * 100, 100), 1 if n > 0 else 0):.1f}"

    # Problem badges: highlight which research problem is actively being solved
    sal_active    = compressed > 0 or forgotten > 0   # salience forgetting fired
    contra_active = archived > 0                       # contradiction resolver fired
    sal_badge = (
        "<span style='background:#065f46;color:#6ee7b7;border-radius:4px;"
        "padding:1px 7px;font-size:0.7rem;font-weight:600'>"
        "✓ Salience-Aware Forgetting</span>"
        if sal_active else
        "<span style='background:rgba(255,255,255,0.05);color:#64748b;border-radius:4px;"
        "padding:1px 7px;font-size:0.7rem'>"
        "○ Salience-Aware Forgetting</span>"
    )
    contra_badge = (
        "<span style='background:#4c1d95;color:#c4b5fd;border-radius:4px;"
        "padding:1px 7px;font-size:0.7rem;font-weight:600'>"
        "✓ Contradiction Reconciliation</span>"
        if contra_active else
        "<span style='background:rgba(255,255,255,0.05);color:#64748b;border-radius:4px;"
        "padding:1px 7px;font-size:0.7rem'>"
        "○ Contradiction Reconciliation</span>"
    )

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);
                border-radius:10px;padding:0.75rem 1.1rem;margin-bottom:0.9rem">

      <!-- Row 1: title + status -->
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
        <span style="font-size:0.82rem;font-weight:700;color:#cbd5e1;letter-spacing:0.03em">
          🧠 MEMORY CONTEXT &nbsp;
          <span style="font-weight:400;color:#94a3b8">{active} / {ec} active slots used</span>
        </span>
        <span style="font-size:0.78rem">{status_html}</span>
      </div>

      <!-- Row 1b: problem badges -->
      <div style="display:flex;gap:0.5rem;margin-bottom:6px">
        {sal_badge}
        {contra_badge}
      </div>

      <!-- Segmented bar -->
      <div style="background:rgba(0,0,0,0.12);border-radius:6px;height:14px;
                  overflow:hidden;display:flex;gap:1px;position:relative">
        <div style="width:{_bar_pct(active)}%;
                    background:#22c55e;height:100%;transition:width 0.4s ease"></div>
        <div style="width:{_bar_pct(compressed)}%;
                    background:#3b82f6;height:100%;transition:width 0.4s ease"></div>
        <div style="width:{_bar_pct(archived)}%;
                    background:#b45309;height:100%;transition:width 0.4s ease"></div>
        <div style="width:{_bar_pct(forgotten)}%;
                    background:#475569;height:100%;transition:width 0.4s ease"></div>
        <div style="position:absolute;right:8px;top:50%;transform:translateY(-50%);
                    font-size:0.7rem;color:rgba(255,255,255,0.65);font-weight:600">{pct:.0f}%</div>
      </div>

      <!-- Legend -->
      <div style="display:flex;gap:1.2rem;margin-top:6px;font-size:0.77rem;color:#94a3b8;align-items:center;flex-wrap:wrap">
        <span>🟢 <b style="color:#e2e8f0">{active}</b> Active</span>
        <span>🔷 <b style="color:#e2e8f0">{compressed}</b> Compressed</span>
        <span>📦 <b style="color:#e2e8f0">{archived}</b> Archived</span>
        <span>⚫ <b style="color:#e2e8f0">{forgotten}</b> Forgotten</span>
        <span style="margin-left:auto;font-size:0.72rem">
          Cap: {ec} &nbsp;·&nbsp;
          Fires at 90% ({ec * 90 // 100}) &nbsp;·&nbsp;
          Target 70% ({ec * 70 // 100})
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)


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

    # Memory context tracker — full-width bar above chat
    _render_memory_tracker()

    # Editable name + generate toggle on same row
    col_name, col_gen = st.columns([5, 2])
    with col_name:
        new_name = st.text_input("Chat name", value=chat["name"],
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
                if msg["role"] == "assistant":
                    intent = msg.get("intent")
                    proc = msg.get("process")
                    if intent == "question":
                        n_hits = len(msg.get("hits", []) or [])
                        st.caption(
                            f"🔍 Retrieved {n_hits} memor{'y' if n_hits == 1 else 'ies'}"
                            if n_hits else "🔍 Retrieved from memory"
                        )
                    elif intent == "statement":
                        # Only claim "stored" when something was actually stored.
                        # Otherwise show a neutral caption so the UI matches reality.
                        n_stored = len(proc.extracted_mus) if proc else 0
                        if n_stored > 0:
                            st.caption(
                                f"💾 Stored {n_stored} fact"
                                f"{'s' if n_stored != 1 else ''} to memory"
                            )
                        else:
                            st.caption("💬 Note received (nothing extractable to store)")
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    _render_process(msg.get("process"))
                    _render_hits(msg.get("hits", []), msg.get("retrieval_ms", 0))

    prompt = st.chat_input(
        "Tell me something to remember, or ask a question (use ? or start with what/where/who/how/when)…",
        key=f"inp_{cid}",
    )
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
                st.error(f"⚠️ {LogSanitizer.sanitize(str(e))}")
                return

            # Basic prompt injection check
            if not InputValidator.is_safe_for_llm(sanitized):
                st.warning("⚠️ Your message contains patterns that may not be processed correctly. Please rephrase.")
                return

            _handle_message(engine, cid, chat, sanitized, do_gen)
            st.rerun()
        except ValidationError as e:
            st.error(f"⚠️ {LogSanitizer.sanitize(str(e))}")
        except Exception as e:
            logger.exception("Unhandled error in chat")
            # Sanitize before display so accidental key leakage in the
            # exception's repr (rare but possible from httpx errors) is masked.
            st.error(
                f"⚠️ **Error:** `{type(e).__name__}` — "
                f"{LogSanitizer.sanitize(str(e))[:300]}"
            )


def _detect_intent(text: str, engine=None) -> str:
    """Classify message as 'question' or 'statement'.

    Three-tier approach:
      1. Fast: explicit "?" → question
      2. Fast: starts with unambiguous question word → question
      3. LLM: call claude-3-haiku for any ambiguous case (cached, cheap, reliable)
    """
    t = text.strip()
    lower = t.lower()

    # Tier 1: explicit question mark
    if "?" in t:
        return "question"

    # Tier 2: unambiguous question starters (word-boundary safe — include trailing space)
    _DEFINITE_Q = (
        "what ", "who ", "where ", "when ", "how ", "why ", "which ",
        "tell me ", "remind me ", "recall ", "retrieve ",
        "show me ", "find ", "search ", "look up ", "fetch ",
        "do i ", "did i ", "have i ", "am i ",
        "is my ", "was my ", "are my ", "were my ",
        "what's my", "what is my", "who is my", "where is my",
        "do you know", "do you remember", "what do you know",
        "list my", "list all", "give me ",
    )
    if any(lower.startswith(q) for q in _DEFINITE_Q):
        return "question"

    # Tier 3: LLM classification for ambiguous cases
    if engine is not None:
        try:
            import hashlib as _hashlib
            _prompt = (
                "Classify this message as either 'question' (the user is asking for "
                "information or wants a retrieval) or 'statement' (the user is sharing "
                "or telling information to store).\n\n"
                f"Message: \"{t}\"\n\n"
                "Reply with exactly one word: question or statement"
            )
            _resp = engine._llm.chat_completion(
                model="anthropic/claude-3-haiku",
                messages=[{"role": "user", "content": _prompt}],
                prompt_template_version="intent_v1",
                cache_input=_hashlib.sha256(t.encode()).hexdigest()[:16],
                max_tokens=5,
                temperature=0.0,
            )
            _result = _resp.content.strip().lower()
            if "question" in _result:
                return "question"
            if "statement" in _result:
                return "statement"
        except Exception:
            pass  # Fall through to heuristic below

    # Tier 3 fallback heuristic (when LLM unavailable)
    _FALLBACK_Q = (
        "can you ", "could you ", "would you ", "is there ", "are there ",
        "do you ", "have you ", "was there ", "were there ",
    )
    return "question" if any(lower.startswith(q) for q in _FALLBACK_Q) else "statement"


def _handle_message(engine: SystemEngine, cid: str, chat: dict, text: str, do_gen: bool) -> None:
    chat["messages"].append({"role": "user", "content": text})
    sid = chat["session_id"]

    intent = _detect_intent(text, engine=engine)
    process: ProcessResult | None = None
    hits: list = []
    answer = ""
    retrieval_ms = 0.0

    if intent == "question":
        # ── Retrieval + optional LLM generation ──────────────────────────────
        # Spinner gives the user feedback during the 2–5 s blocking call —
        # without this the UI freezes silently while we hit the LLM.
        try:
            with st.spinner("🔍 Searching memory…" if not do_gen else "🧠 Thinking…"):
                result = engine.ask(text, session_id=sid, generate=do_gen)
            hits = result.hits
            retrieval_ms = result.retrieval_latency_ms

            if not hits:
                answer = (
                    "💭 I searched my memory but couldn't find anything relevant.\n\n"
                    "Try telling me about this topic first, then ask again."
                )
            elif do_gen:
                raw = result.answer.strip() if result.answer else ""
                answer = raw if raw else (
                    "I found relevant memories but the answer generation returned empty. "
                    "Here's the top memory: **" + hits[0].mu.claim + "**"
                )
            else:
                # Generation disabled — show top hit
                answer = f"**From memory:** {hits[0].mu.claim}"
                if len(hits) > 1:
                    answer += (
                        f"\n\n*{len(hits) - 1} more related memory(ies) found — "
                        "enable ✨ Generate answer for a complete response.*"
                    )
        except Exception as exc:
            logger.exception("engine.ask failed")
            answer = (
                f"⚠️ **Retrieval error:** `{type(exc).__name__}` — "
                f"{LogSanitizer.sanitize(str(exc))[:200]}"
            )

    else:
        # ── Fact ingestion ────────────────────────────────────────────────────
        try:
            with st.spinner("💾 Storing memory…"):
                process = engine.process_message(text, speaker="User", session_id=sid)
            n = len(process.extracted_mus)
            if n == 0:
                answer = (
                    "💬 I heard you, but didn't extract any concrete facts to store. "
                    "Try stating something more specific, e.g. *'I live in New York'* or "
                    "*'My favourite food is pizza'*."
                )
            else:
                facts_str = "\n".join(f"• {mu.claim}" for mu in process.extracted_mus)
                answer = f"Stored **{n}** {'fact' if n == 1 else 'facts'} in memory:\n{facts_str}"
                if process.superseded_ids:
                    answer += (
                        f"\n\n⚡ **Updated** {len(process.superseded_ids)} older "
                        "fact(s) that this supersedes."
                    )
                if process.lifecycle and (
                    process.lifecycle.n_compressed or process.lifecycle.n_forgotten
                ):
                    answer += (
                        f"\n\n🔄 **Lifecycle:** compressed "
                        f"{process.lifecycle.n_compressed}, "
                        f"forgot {process.lifecycle.n_forgotten} low-salience facts."
                    )
        except Exception as exc:
            logger.exception("engine.process_message failed")
            answer = (
                f"⚠️ **Storage error:** `{type(exc).__name__}` — "
                f"{LogSanitizer.sanitize(str(exc))[:200]}"
            )

    chat["messages"].append({
        "role": "assistant",
        "content": answer,
        "hits": hits,
        "process": process,
        "retrieval_ms": retrieval_ms,
        "intent": intent,
    })

    if len(chat["messages"]) == 2 and chat["name"].startswith("Chat "):
        chat["name"] = text[:35] + ("…" if len(text) > 35 else "")


def _render_process(process: ProcessResult | None) -> None:
    if process is None or not process.extracted_mus:
        return
    with st.expander(f"🧠 {len(process.extracted_mus)} fact(s) extracted & stored", expanded=False):
        for mu in process.extracted_mus:
            icon, label = _TIER.get(mu.status, ("⚪", mu.status.value))
            # html.escape on the user-supplied claim so HTML/JS injected via the
            # extractor (e.g. an LLM that ever surfaces raw user input) cannot
            # execute when rendered with unsafe_allow_html=True.
            st.markdown(
                f"<div class='hit-card'>"
                f"<b>{icon} {label}</b> "
                f"<span style='color:#94a3b8;font-size:0.8em'>salience={mu.salience_score:.2f} "
                f"· conf={mu.confidence:.2f}</span><br>{html.escape(mu.claim)}"
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
            _rel = getattr(hit, "relation_meta", None)
            if _rel and getattr(_rel, "superseded_by", None):
                section = " · HISTORICAL"
            elif _rel and getattr(_rel, "conflicts_with", None):
                section = " · CONFLICTING"
            st.markdown(
                f"<div class='hit-card'>"
                f"<b>#{i} {icon} {label}{section}</b> "
                f"<span style='color:#94a3b8;font-size:0.8em'>session={html.escape(str(mu.session_id))} "
                f"· salience={mu.salience_score:.2f} · rrf={hit.rrf_score:.4f}</span><br>"
                f"{html.escape(mu.claim)}"
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
    active_n     = counts.get("active", 0)
    compressed_n = counts.get("compressed", 0)
    archived_n   = counts.get("archived", 0)
    forgotten_n  = counts.get("forgotten", 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🟢 Active", active_n)
    c2.metric("🔷 Compressed", compressed_n)
    c3.metric("📦 Archived", archived_n)
    c4.metric("⚫ Forgotten", forgotten_n)

    # Problem-solved summary row
    ec = _engine_cap()
    pb1_on = compressed_n > 0 or forgotten_n > 0
    pb2_on = archived_n > 0
    pb1_style = "background:#065f46;color:#6ee7b7" if pb1_on else "background:#1e293b;color:#64748b"
    pb2_style = "background:#4c1d95;color:#c4b5fd" if pb2_on else "background:#1e293b;color:#64748b"
    st.markdown(
        f"<div style='display:flex;gap:0.6rem;margin:0.4rem 0'>"
        f"<span style='{pb1_style};border-radius:5px;padding:3px 10px;font-size:0.75rem;font-weight:600'>"
        f"{'✓' if pb1_on else '○'} Salience-Aware Forgetting (cap={ec})</span>"
        f"<span style='{pb2_style};border-radius:5px;padding:3px 10px;font-size:0.75rem;font-weight:600'>"
        f"{'✓' if pb2_on else '○'} Contradiction Reconciliation</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    pressure = engine.lifecycle_pressure()
    if pressure > 0:
        st.progress(
            min(pressure, 1.0),
            text=f"Capacity {pressure*100:.0f}%  (fires at 90% of cap={ec}, targets 70%)"
        )

    st.divider()

    _all_statuses = [s.value for s in MemoryStatus]
    _default_statuses = [s.value for s in (
        MemoryStatus.ACTIVE, MemoryStatus.ARCHIVED,
        MemoryStatus.COMPRESSED, MemoryStatus.FORGOTTEN,
    )]
    _status_filter_labels = {
        MemoryStatus.ACTIVE.value:     "🟢 Active",
        MemoryStatus.ARCHIVED.value:   "🔷 Compressed / 📦 Archived",
        MemoryStatus.COMPRESSED.value: "🔷 Compressed (label)",
        MemoryStatus.FORGOTTEN.value:  "⚫ Forgotten",
    }
    sel = st.multiselect(
        "Show layers", _all_statuses, default=_default_statuses,
        format_func=lambda v: _status_filter_labels.get(v, v),
    )

    all_mus = engine.get_memories()
    filtered = [m for m in all_mus if m.status.value in sel]

    if not filtered:
        st.info("No memories yet. Go to Chat and tell me something to remember!")
    else:
        st.caption(f"{len(filtered)} memor{'y' if len(filtered)==1 else 'ies'} shown")

    # Pre-compute MU-id → claim map so we can resolve "replaced by" provenance
    # without an N+1 DB lookup per card.
    _mu_claim_lookup = {m.mu_id: m.claim for m in all_mus}

    for mu in filtered:
        # 4-layer display:
        # - ACTIVE                                              → 🟢 Active
        # - ARCHIVED w/ compressed_label_id (lifecycle-comp)   → 🔷 Compressed
        # - ARCHIVED w/o label  (replaced by newer / decayed)  → 📦 Archived
        # - COMPRESSED  (legacy label-only state)              → 🔷 Compressed
        # - FORGOTTEN                                           → ⚫ Forgotten
        if mu.status == MemoryStatus.ARCHIVED and mu.compressed_label_id is None:
            icon, label = ("📦", "Archived")
            problem_tag = "Contradiction Reconciliation"
            problem_color = "#b45309"
        elif mu.status == MemoryStatus.ARCHIVED:
            icon, label = ("🔷", "Compressed")
            problem_tag = "Salience-Aware Forgetting"
            problem_color = "#3b82f6"
        elif mu.status == MemoryStatus.COMPRESSED:
            icon, label = ("🔷", "Compressed")
            problem_tag = "Salience-Aware Forgetting"
            problem_color = "#3b82f6"
        elif mu.status == MemoryStatus.FORGOTTEN:
            icon, label = ("⚫", "Forgotten")
            problem_tag = "Salience-Aware Forgetting"
            problem_color = "#475569"
        else:
            icon, label = ("🟢", "Active")
            problem_tag = None
            problem_color = None

        # Look up SUPERSEDED_BY edges so we can show "Replaced by …" provenance.
        # This is the audit trail that lets users see why an Archived memory
        # ended up there even though the layer is no longer named "Superseded".
        replaced_by_claims: list[str] = []
        try:
            from locomo_memory.phase2.schemas import EdgeType as _EdgeType
            for _edge in engine.store.edges_from(mu.mu_id, _EdgeType.SUPERSEDED_BY):
                target_claim = _mu_claim_lookup.get(_edge.target_mu_id)
                if target_claim is None:
                    target_mu = engine.store.get_memory_unit(_edge.target_mu_id)
                    if target_mu is not None:
                        target_claim = target_mu.claim
                if target_claim:
                    replaced_by_claims.append(target_claim)
        except Exception:
            pass  # provenance is best-effort; never block the card render

        with st.expander(
            f"{icon} [{label}]  {mu.claim[:70]}{'…' if len(mu.claim)>70 else ''}",
            expanded=False,
        ):
            # Problem badge
            if problem_tag:
                st.markdown(
                    f"<span style='background:{problem_color}22;color:{problem_color};"
                    f"border:1px solid {problem_color}55;border-radius:4px;"
                    f"padding:2px 8px;font-size:0.72rem;font-weight:600'>"
                    f"Solves: {problem_tag}</span>",
                    unsafe_allow_html=True,
                )

            # Provenance: if this MU was replaced by newer same-topic facts,
            # show them right at the top of the card so the audit trail is
            # immediately visible.  Each claim text is HTML-escaped so an LLM
            # extractor that ever lets HTML through cannot inject script tags.
            if replaced_by_claims:
                _replacements = "  •  ".join(html.escape(c[:90]) for c in replaced_by_claims[:3])
                st.markdown(
                    f"<div style='background:#7c2d1222;border-left:3px solid #b45309;"
                    f"padding:6px 10px;margin:6px 0;font-size:0.83rem;color:#fbbf24'>"
                    f"📌 <b>Replaced by:</b> {_replacements}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Speculative / low-confidence warning
            if mu.confidence < 0.5:
                st.markdown(
                    f"<div style='background:#3b00821a;border-left:3px solid #fbbf24;"
                    f"padding:6px 10px;margin:6px 0;font-size:0.82rem;color:#fcd34d'>"
                    f"⚠️ <b>Low-confidence claim</b> "
                    f"(confidence={mu.confidence:.2f}). This fact is treated "
                    f"as speculative and cannot supersede higher-confidence facts."
                    f"</div>",
                    unsafe_allow_html=True,
                )

            col_l, col_r = st.columns(2)

            # --- Claim ---
            col_l.markdown(f"**Claim:**")
            col_l.info(mu.claim)

            # --- Provenance ---
            col_l.markdown("**Provenance**")
            _created = mu.created_at.strftime("%Y-%m-%d %H:%M:%S") if mu.created_at else "—"
            _accessed = mu.last_accessed.strftime("%Y-%m-%d %H:%M:%S") if mu.last_accessed else "never"
            col_l.markdown(
                f"- **Session:** `{mu.session_id}`\n"
                f"- **Speaker:** {mu.source_speaker or '—'}\n"
                f"- **Extracted:** {_created}\n"
                f"- **Last used:** {_accessed}\n"
                f"- **Retrieved:** {mu.retrieval_count}×"
            )
            if mu.original_text:
                col_l.markdown("**Original message:**")
                col_l.caption(f'"{mu.original_text[:200]}{"…" if len(mu.original_text) > 200 else ""}"')

            # --- Salience breakdown ---
            col_r.markdown("**Salience**")
            col_r.markdown(
                f"- **Score:** `{mu.salience_score:.3f}`\n"
                f"- **Importance:** `{mu.importance:.2f}`\n"
                f"- **Confidence:** `{mu.confidence:.2f}`"
            )
            _sal_bar = int(mu.salience_score * 100)
            col_r.progress(_sal_bar, text=f"Salience {mu.salience_score:.2f}")

            # --- Status detail ---
            col_r.markdown("**Status**")
            col_r.markdown(f"`{label}`")
            if mu.compressed_label_id:
                col_r.markdown(f"- **Label ID:** `{mu.compressed_label_id[:16]}…`")
            if mu.archived_entry_id:
                col_r.markdown(f"- **Archive ID:** `{mu.archived_entry_id[:16]}…`")
            col_r.markdown(f"- **MU ID:** `{mu.mu_id[:16]}…`")

            st.markdown("---")
            btn_col1, btn_col2, _ = st.columns([2, 2, 4])
            with btn_col1:
                if st.button("🗑️ Delete", key=f"del_mu_{mu.mu_id}",
                             help="Permanently delete this memory (audit row kept)"):
                    try:
                        engine.store.delete_atomic(mu.mu_id, deleted_by="user_ui")
                        with engine._index_lock:
                            engine.faiss_index.rebuild_from_store(
                                engine.store, conversation_id=engine.conversation_id)
                            engine.bm25_index.rebuild_from_store(
                                engine.store, conversation_id=engine.conversation_id)
                            engine.label_index.rebuild_from_store(
                                engine.store, conversation_id=engine.conversation_id)
                        st.success(f"Deleted `{mu.mu_id[:12]}`.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Delete failed: {LogSanitizer.sanitize(str(exc))}")
            with btn_col2:
                _restorable = mu.status in (
                    MemoryStatus.FORGOTTEN,
                    MemoryStatus.ARCHIVED,
                    MemoryStatus.COMPRESSED,
                )
                if _restorable:
                    if mu.status == MemoryStatus.ARCHIVED and mu.compressed_label_id is None:
                        _restore_tip = "Restore archived memory back to active"
                    else:
                        _restore_tip = {
                            MemoryStatus.FORGOTTEN: "Promote forgotten memory back to active",
                            MemoryStatus.ARCHIVED:  "Restore compressed memory back to active",
                            MemoryStatus.COMPRESSED: "Restore compressed memory back to active",
                        }[mu.status]
                    if st.button("♻️ Restore", key=f"restore_mu_{mu.mu_id}",
                                 help=_restore_tip):
                        try:
                            if mu.status == MemoryStatus.FORGOTTEN:
                                engine.store.restore_from_forgotten(mu.mu_id)
                            else:
                                engine.store.restore_atomic(mu.mu_id)
                            with engine._index_lock:
                                engine.faiss_index.rebuild_from_store(
                                    engine.store, conversation_id=engine.conversation_id)
                                engine.bm25_index.rebuild_from_store(
                                    engine.store, conversation_id=engine.conversation_id)
                                engine.label_index.rebuild_from_store(
                                    engine.store, conversation_id=engine.conversation_id)
                            st.success("Memory restored to Active.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Restore failed: {LogSanitizer.sanitize(str(exc))}")

    # ── Deletion audit log (at bottom) ────────────────────────────────────
    audit_rows = engine.store.list_deletion_audit(conversation_id=engine.conversation_id)
    if audit_rows:
        st.divider()
        st.markdown("**🗑️ Deletion Audit Log**")
        for row in audit_rows:
            _dt = row.deleted_at
            _dt_str = _dt.isoformat()[:19] if hasattr(_dt, "isoformat") else str(_dt)[:19]
            st.markdown(
                f"`{row.mu_id[:14]}` — deleted at `{_dt_str}` "
                f"by `{row.deleted_by}`"
            )


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
