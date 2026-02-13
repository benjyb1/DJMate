"""
DJMate — Similar Tracks Explorer
Run: streamlit run streamSimilar.py
"""

import streamlit as st
import asyncio
import os
import sys
import numpy as np
from dotenv import load_dotenv

def _load_env():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        p = os.path.join(d, ".env")
        if os.path.exists(p):
            load_dotenv(p, override=True)
            return
        d = os.path.dirname(d)
    load_dotenv()
_load_env()

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

from Backend.data.db_interface import DatabaseManager

st.set_page_config(page_title="DJMate — Similar Tracks", page_icon="🔀", layout="wide")

st.markdown("""
<style>
  .result-card {
    background: #111; border: 1px solid #2a2a2a;
    border-radius: 10px; padding: 14px 18px; margin-bottom: 4px;
  }
  .result-card:hover { border-color: #444; }
  .rank   { font-size: 28px; color: #333; font-weight: bold; line-height: 1; }
  .title  { font-size: 15px; font-weight: bold; color: #fff; }
  .artist { font-size: 12px; color: #aaa; margin-top: 2px; }
  .meta   { font-size: 11px; color: #666; margin-top: 6px; }
  .sim-bar { height: 4px; border-radius: 2px; margin-top: 8px; }
  .direction-badge {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 600; margin-top: 6px;
  }
  .inferred-badge {
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-size: 10px; font-weight: 500; margin-top: 4px;
    background: #1a1a00; color: #aaaa00; border: 1px solid #444400;
  }
  .tag {
    display: inline-block; background: #1a1a3a; color: #6666cc;
    border-radius: 10px; padding: 2px 7px; font-size: 10px; margin: 2px;
  }
  .relax-notice {
    background: #1a1100; border: 1px solid #443300; border-radius: 8px;
    padding: 8px 14px; margin-bottom: 14px; font-size: 12px; color: #cc9900;
  }
</style>
""", unsafe_allow_html=True)

# ── DB singleton ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_db():
    return DatabaseManager()

db = get_db()

# ── Async helper ──────────────────────────────────────────────────────────────
def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)

# ── Track index for autocomplete ──────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_track_index():
    if not db.client:
        return []
    try:
        resp = db.client.table("tracks") \
            .select("trackid, title, artist, bpm, key") \
            .order("artist").execute()
        results = []
        for r in resp.data or []:
            label = f"{r.get('artist','?')} — {r.get('title','?')}"
            if r.get("bpm"):
                label += f"  ({int(r['bpm'])} BPM)"
            results.append((label, r["trackid"]))
        return results
    except Exception as e:
        st.error(f"Failed to load track index: {e}")
        return []

# ── Direction badge ───────────────────────────────────────────────────────────
def describe_direction(source, candidate) -> tuple[str, str]:
    def _get(obj, key):
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

    clues = []
    s_e = float(_get(source,    "energy") or 0.5)
    c_e = float(_get(candidate, "energy") or 0.5)
    if c_e - s_e >  0.15: clues.append(("higher energy", "#ff4444"))
    if c_e - s_e < -0.15: clues.append(("lower energy",  "#4488ff"))

    s_b = _get(source,    "bpm")
    c_b = _get(candidate, "bpm")
    if s_b and c_b:
        diff = float(c_b) - float(s_b)
        if diff >  5: clues.append((f"+{int(diff)} BPM faster", "#ffaa00"))
        if diff < -5: clues.append((f"{int(diff)} BPM slower", "#aaaaff"))

    def _tags(o):
        v = o.get("semantic_tags") if isinstance(o, dict) else getattr(o, "semantic_tags", None)
        return set(v or [])

    def _vibes(o):
        v = (o.get("vibe") or o.get("vibe_descriptors")) if isinstance(o, dict) \
            else getattr(o, "vibe_descriptors", None)
        return set(v if isinstance(v, list) else ([v] if v else []))

    new_tags  = _tags(candidate)  - _tags(source)
    new_vibes = _vibes(candidate) - _vibes(source)
    if new_tags:  clues.append((f"more {next(iter(new_tags))}", "#00ff88"))
    if new_vibes: clues.append((next(iter(new_vibes)), "#cc88ff"))

    if not clues: return ("similar vibe", "#555555")
    for desc, col in clues:
        if "more " in desc: return (desc, col)
    return clues[0]

# ── Cosine similarity ─────────────────────────────────────────────────────────
def cosine_similarity(v1, v2) -> float:
    try:
        a, b = np.array(v1, dtype=np.float32), np.array(v2, dtype=np.float32)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0
    except Exception:
        return 0.0


# =============================================================================
# ★ _render_results — defined HERE, before any tabs that call it
#
#   Audio is in its own st.columns() cell, completely isolated from the HTML
#   markdown block. This is the exact pattern from Tagger.py that works.
# =============================================================================
def _render_results(pairs: list, source=None):
    for i, (sim_score, track) in enumerate(pairs, 1):

        if isinstance(track, dict):
            title        = track.get("title")         or "Unknown"
            artist       = track.get("artist")        or "Unknown"
            bpm          = track.get("bpm")
            key          = track.get("key")
            energy       = track.get("energy")
            tags_l       = track.get("semantic_tags") or []
            filepath     = track.get("filepath")
            inferred     = track.get("_inferred",     False)
            score_detail = track.get("_score_detail", {})
        else:
            title        = getattr(track, "title",         None) or "Unknown"
            artist       = getattr(track, "artist",        None) or "Unknown"
            bpm          = getattr(track, "bpm",           None)
            key          = getattr(track, "key",           None)
            energy       = getattr(track, "energy",        None)
            tags_l       = getattr(track, "semantic_tags", None) or []
            filepath     = getattr(track, "filepath",      None)
            inferred     = getattr(track, "_inferred",     False)
            score_detail = getattr(track, "_score_detail", {})

        direction_html = ""
        if source is not None:
            direction, dc = describe_direction(source, track)
            direction_html = (
                f'<span class="direction-badge" style="background:{dc}22;'
                f'color:{dc};border:1px solid {dc}66">↗ {direction}</span>'
            )

        inferred_html = (
            '<span class="inferred-badge">🔮 inferred — not yet tagged</span>'
            if inferred else ""
        )

        bar_pct    = int(sim_score * 100)
        bar_colour = "#00ff88" if sim_score > 0.7 else "#ffaa00" if sim_score > 0.4 else "#ff4444"

        meta_parts = []
        if bpm:                meta_parts.append(f"{int(bpm)} BPM")
        if key:                meta_parts.append(key)
        if energy is not None: meta_parts.append(f"energy {float(energy):.2f}")
        meta = " · ".join(meta_parts)

        matched_tags = score_detail.get("matched_tags", {})
        tags_html = ""
        for t in tags_l[:6]:
            conf = matched_tags.get(t) or matched_tags.get(t.lower())
            if conf:
                tags_html += (
                    f'<span class="tag" style="opacity:{max(0.4,conf):.1f};'
                    f'border:1px solid #4444aa">{t} {int(conf*100)}%</span>'
                )
            else:
                tags_html += f'<span class="tag">{t}</span>'

        # ── Two columns: card info | audio player ─────────────────────────────
        # Mirrors Tagger.py's col_left / col_center pattern exactly.
        # st.audio() is ONLY ever called inside a column, never adjacent to
        # a st.markdown(unsafe_allow_html=True) block.
        col_card, col_play = st.columns([3, 1])

        with col_card:
            st.markdown(f"""
            <div class="result-card">
              <div style="display:flex;gap:14px;align-items:flex-start">
                <div class="rank">{i}</div>
                <div style="flex:1">
                  <div class="title">{title}</div>
                  <div class="artist">{artist}</div>
                  <div class="meta">{meta}</div>
                  <div style="margin-top:6px">{tags_html}</div>
                  <div style="margin-top:4px">{inferred_html} {direction_html}</div>
                  <div class="sim-bar" style="width:{bar_pct}%;background:{bar_colour}"></div>
                  <div style="font-size:10px;color:#555;margin-top:2px">{bar_pct}% match</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_play:
            # Exact same call as Tagger.py — string path, not bytes
            if filepath and os.path.exists(filepath):
                st.audio(filepath, format="audio/mp3")
            elif filepath:
                st.caption(f"⚠️ File not found")


# ── Similarity search ─────────────────────────────────────────────────────────
async def find_similar(source_id: str, n: int = 7):
    source = await db.get_track_by_id(source_id)
    if not source:
        return None, []

    embedding = getattr(source, "embedding", None)
    if not embedding:
        return source, []

    raw = await db.find_similar_tracks(query_embedding=embedding, limit=n + 1, threshold=0.2)
    results = [r for r in raw if str(r.get("id") or r.get("trackid")) != str(source_id)][:n]

    enriched = []
    for r in results:
        rid  = r.get("id") or r.get("trackid")
        full = await db.get_track_by_id(str(rid))
        sim  = float(r.get("similarity", 0.5))
        if full:
            c_emb = getattr(full, "embedding", None)
            if c_emb and embedding:
                sim = cosine_similarity(embedding, c_emb)
            enriched.append((sim, full))
        else:
            enriched.append((sim, r))

    enriched.sort(key=lambda x: x[0], reverse=True)
    return source, enriched


# =============================================================================
# UI — tabs come AFTER all function definitions
# =============================================================================
st.title("🔀 DJMate — Similar Tracks Explorer")

track_index = load_track_index()
if not track_index:
    st.error("No tracks loaded — check your Supabase connection.")
    st.stop()

labels    = [t[0] for t in track_index]
track_ids = [t[1] for t in track_index]

tab_pick, tab_describe = st.tabs(["🎵 Pick a track", "💬 Describe what you want"])

# =============================================================================
# TAB 1 — Pick a track
# =============================================================================
with tab_pick:
    st.caption("Pick a track and find similar ones by sound.")

    selected_label = st.selectbox(
        "Search for a track:", options=labels, index=None,
        placeholder="Type to search…", key="track_select",
    )
    selected_id = track_ids[labels.index(selected_label)] if selected_label else None

    search_btn = st.button("Find Similar Tracks", type="primary", key="pick_btn")
    st.divider()

    if search_btn and selected_id:
        with st.spinner("Computing similarity…"):
            source, similar = run_async(find_similar(selected_id, 7))

        if source is None:
            st.error("Track not found.")
            st.stop()

        s_title  = getattr(source, "title",    None) or "Unknown"
        s_artist = getattr(source, "artist",   None) or "Unknown"
        s_bpm    = getattr(source, "bpm",      None)
        s_key    = getattr(source, "key",      None)
        s_energy = getattr(source, "energy",   None)
        s_path   = getattr(source, "filepath", None)

        # Source track — same two-column layout
        col_src, col_src_play = st.columns([3, 1])
        with col_src:
            st.markdown(f"""
            <div style="background:#0a0a1a;border:1px solid #00ffff;border-radius:10px;padding:16px 20px;margin-bottom:8px">
              <div style="font-size:11px;color:#00ffff;opacity:.7;margin-bottom:4px">SOURCE TRACK</div>
              <div style="font-size:18px;font-weight:bold;color:#fff">{s_title}</div>
              <div style="font-size:13px;color:#aaa;margin-top:3px">{s_artist}</div>
              <div style="font-size:12px;color:#666;margin-top:6px">
                {f'{int(s_bpm)} BPM' if s_bpm else ''}
                {' · ' + s_key if s_key else ''}
                {f' · energy {s_energy:.2f}' if s_energy else ''}
              </div>
            </div>
            """, unsafe_allow_html=True)
        with col_src_play:
            if s_path and os.path.exists(s_path):
                st.audio(s_path, format="audio/mp3")

        if not similar:
            st.warning("No embedding found — run your embedding pipeline first.")
            st.stop()

        st.markdown(f"### {len(similar)} similar tracks")
        _render_results(similar, source)


# =============================================================================
# TAB 2 — Describe what you want
# =============================================================================
with tab_describe:
    st.caption('Describe the vibe. e.g. "fast kicking house", "dark minimal give me 8"')

    query_input = st.text_input(
        "What do you want to play next?",
        placeholder='e.g. "upbeat deep house", "dark minimal techno give me 8"',
        key="semantic_query",
    )
    describe_btn = st.button("Search", type="primary", key="describe_btn")
    st.divider()

    if describe_btn and query_input.strip():
        try:
            from Backend.llm_interpreter import SemanticInterpreter, InterpretationContext
        except ImportError:
            st.error("SemanticInterpreter not found — check Backend/llm_interpreter.py")
            st.stop()

        @st.cache_resource
        def get_interpreter():
            interp = SemanticInterpreter(supabase_client=db.client)
            run_async(interp.initialize())
            return interp

        interpreter = get_interpreter()

        with st.spinner("Interpreting your request…"):
            params = run_async(interpreter.interpret(query_input.strip()))

        with st.spinner(f"Finding up to {params.get('track_count', 5)} tracks…"):
            tracks, meta = run_async(interpreter.search(params, db_manager=db))

        if meta.get("relaxation_step", 0) > 0:
            st.markdown(
                f'<div class="relax-notice">⚡ Widened search ({meta.get("relaxation_label","")})</div>',
                unsafe_allow_html=True,
            )
        if meta.get("inferred_count", 0) > 0:
            st.markdown(
                f'<div class="relax-notice">🔮 {meta["inferred_count"]} track(s) inferred by sound similarity</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f"**{len(tracks)} tracks** · _{params.get('reasoning', '')}_ "
            f"· confidence {params.get('confidence', 0):.0%} "
            f"· `{params.get('model_used', 'fallback')}`"
        )

        _render_results([(t.get("_relevance_score", 0.5), t) for t in tracks])