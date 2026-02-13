"""
DJMate — Similar Tracks Explorer
Run: streamlit run streamSimilar.py

Pick a track (with autocomplete from your DB), get similar tracks,
each described in terms of direction: "more tech house", "lower energy", etc.

Track count is no longer a manual slider — it comes from the LLM when you
use the semantic search path, and defaults to 7 for the direct similarity path.
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

st.set_page_config(
    page_title="DJMate — Similar Tracks",
    page_icon="🔀",
    layout="wide",
)

st.markdown("""
<style>
  .result-card {
    background: #111;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    position: relative;
  }
  .result-card:hover { border-color: #444; }
  .rank    { font-size: 28px; color: #333; font-weight: bold; line-height: 1; }
  .title   { font-size: 15px; font-weight: bold; color: #fff; }
  .artist  { font-size: 12px; color: #aaa; margin-top: 2px; }
  .meta    { font-size: 11px; color: #666; margin-top: 6px; }
  .sim-bar { height: 4px; border-radius: 2px; margin-top: 8px; }
  .direction-badge {
    display: inline-block;
    padding: 3px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 600; margin-top: 6px;
  }
  .inferred-badge {
    display: inline-block;
    padding: 2px 8px; border-radius: 10px;
    font-size: 10px; font-weight: 500; margin-top: 4px;
    background: #1a1a00; color: #aaaa00; border: 1px solid #444400;
  }
  .tag { display: inline-block; background: #1a1a3a; color: #6666cc;
         border-radius: 10px; padding: 2px 7px; font-size: 10px; margin: 2px; }
  .relax-notice {
    background: #1a1100; border: 1px solid #443300; border-radius: 8px;
    padding: 8px 14px; margin-bottom: 14px; font-size: 12px; color: #cc9900;
  }
</style>
""", unsafe_allow_html=True)

# ── Singleton DB ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_db():
    return DatabaseManager()

db = get_db()

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

# ── Load all track names for autocomplete ─────────────────────────────────────
@st.cache_data(ttl=300)
def load_track_index():
    if not db.client:
        return []
    try:
        resp = db.client.table("tracks") \
            .select("trackid, title, artist, bpm, key") \
            .order("artist") \
            .execute()
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

# ── Direction analysis ────────────────────────────────────────────────────────
def describe_direction(source, candidate) -> tuple[str, str]:
    clues = []

    s_energy = getattr(source, "energy", None) or (source.get("energy") if isinstance(source, dict) else None) or 0.5
    c_energy = getattr(candidate, "energy", None) or (candidate.get("energy") if isinstance(candidate, dict) else None) or 0.5
    diff_e = float(c_energy) - float(s_energy)
    if diff_e > 0.15:
        clues.append(("higher energy", "#ff4444"))
    elif diff_e < -0.15:
        clues.append(("lower energy", "#4488ff"))

    s_bpm = getattr(source, "bpm", None) or (source.get("bpm") if isinstance(source, dict) else None)
    c_bpm = getattr(candidate, "bpm", None) or (candidate.get("bpm") if isinstance(candidate, dict) else None)
    if s_bpm and c_bpm:
        diff_b = float(c_bpm) - float(s_bpm)
        if diff_b > 5:
            clues.append((f"+{int(diff_b)} BPM faster", "#ffaa00"))
        elif diff_b < -5:
            clues.append((f"{int(diff_b)} BPM slower", "#aaaaff"))

    def _tags(obj):
        if isinstance(obj, dict):
            return set(obj.get("semantic_tags") or [])
        return set(getattr(obj, "semantic_tags", None) or [])

    def _vibes(obj):
        if isinstance(obj, dict):
            v = obj.get("vibe") or obj.get("vibe_descriptors") or []
        else:
            v = getattr(obj, "vibe_descriptors", None) or []
        return set(v) if isinstance(v, list) else {v}

    new_tags  = _tags(candidate)  - _tags(source)
    new_vibes = _vibes(candidate) - _vibes(source)

    if new_tags:
        clues.append((f"more {next(iter(new_tags))}", "#00ff88"))
    if new_vibes:
        clues.append((next(iter(new_vibes)), "#cc88ff"))

    if not clues:
        return ("similar vibe", "#555555")
    for desc, colour in clues:
        if "more " in desc:
            return (desc, colour)
    return clues[0]

def cosine_similarity(v1, v2) -> float:
    try:
        a = np.array(v1, dtype=np.float32)
        b = np.array(v2, dtype=np.float32)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    except Exception:
        return 0.0
# =============================================================================
# Shared result renderer
# =============================================================================
def _render_results(pairs: list, source=None):
    """
    Render a list of (similarity_score, track) pairs as result cards.
    `source` is optional — used for direction badges.
    """
    for i, (sim_score, track) in enumerate(pairs, 1):
        if hasattr(track, "title"):
            title    = track.title    or "Unknown"
            artist   = track.artist   or "Unknown"
            bpm      = track.bpm
            key      = track.key
            energy   = track.energy
            tags_l   = track.semantic_tags or []
            filepath = getattr(track, "filepath", None)
            inferred = getattr(track, "_inferred", False)
        elif isinstance(track, dict):
            title    = track.get("title")   or "Unknown"
            artist   = track.get("artist")  or "Unknown"
            bpm      = track.get("bpm")
            key      = track.get("key")
            energy   = track.get("energy")
            tags_l   = track.get("semantic_tags") or []
            filepath = track.get("filepath")
            inferred = track.get("_inferred", False)
        else:
            title = artist = "Unknown"
            bpm = key = energy = None
            tags_l = []
            filepath = None
            inferred = False

        direction_html = ""
        if source is not None:
            direction, d_colour = describe_direction(source, track)
            direction_html = f"""
              <div>
                <span class="direction-badge"
                  style="background:{d_colour}22;color:{d_colour};border:1px solid {d_colour}66">
                  ↗ {direction}
                </span>
              </div>"""

        inferred_html = '<div><span class="inferred-badge">🔮 inferred — not yet tagged</span></div>' if inferred else ""

        bar_pct    = int(sim_score * 100)
        bar_colour = "#00ff88" if sim_score > 0.7 else "#ffaa00" if sim_score > 0.4 else "#ff4444"

        meta_parts = []
        if bpm:    meta_parts.append(f"{int(bpm)} BPM")
        if key:    meta_parts.append(key)
        if energy: meta_parts.append(f"energy {float(energy):.2f}")
        meta = " · ".join(meta_parts)

        tags_html = "".join(f'<span class="tag">{t}</span>' for t in tags_l[:5])

        st.markdown(f"""
        <div class="result-card">
          <div style="display:flex;gap:14px;align-items:flex-start">
            <div class="rank">{i}</div>
            <div style="flex:1">
              <div class="title">{title}</div>
              <div class="artist">{artist}</div>
              <div class="meta">{meta}</div>
              <div style="margin-top:6px">{tags_html}</div>
              {inferred_html}
              {direction_html}
              <div class="sim-bar" style="width:{bar_pct}%;background:{bar_colour}"></div>
              <div style="font-size:10px;color:#555;margin-top:2px">{bar_pct}% match</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    st.audio(f.read(), format="audio/mp3")
            except Exception:
                pass
# ── Direct embedding similarity search (for "pick a track" mode) ─────────────
async def find_similar(source_id: str, n: int = 7):
    source = await db.get_track_by_id(source_id)
    if not source:
        return None, []

    embedding = getattr(source, "embedding", None)
    if not embedding:
        return source, []

    raw = await db.find_similar_tracks(
        query_embedding=embedding,
        limit=n + 1,
        threshold=0.2,
    )

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


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔀 DJMate — Similar Tracks Explorer")

track_index = load_track_index()
if not track_index:
    st.error("No tracks loaded — check your Supabase connection.")
    st.stop()

labels    = [t[0] for t in track_index]
track_ids = [t[1] for t in track_index]

# Two tabs: pick a track directly, or describe what you want
tab_pick, tab_describe = st.tabs(["🎵 Pick a track", "💬 Describe what you want"])

# =============================================================================
# TAB 1 — Direct similarity (original behaviour, slider removed)
# =============================================================================
with tab_pick:
    st.caption("Pick a track and find similar ones by sound.")

    selected_label = st.selectbox(
        "Search for a track:",
        options=labels,
        index=None,
        placeholder="Type to search…",
        key="track_select",
    )
    selected_id = None
    if selected_label:
        selected_id = track_ids[labels.index(selected_label)]

    # ── Track count: no slider — LLM / natural language driven ───────────────
    # The default is 7 for this direct-pick mode. In "Describe" mode below,
    # the LLM parses the count from the query (e.g. "give me 6 tracks").
    DEFAULT_N = 7

    search_btn = st.button("Find Similar Tracks", type="primary", key="pick_btn")
    st.divider()

    if search_btn and selected_id:
        with st.spinner("Computing similarity…"):
            source, similar = run_async(find_similar(selected_id, DEFAULT_N))

        if source is None:
            st.error("Track not found.")
            st.stop()

        # Source card
        s_title  = getattr(source, "title",  None) or "Unknown"
        s_artist = getattr(source, "artist", None) or "Unknown"
        s_bpm    = getattr(source, "bpm",    None)
        s_key    = getattr(source, "key",    None)
        s_energy = getattr(source, "energy", None)

        st.markdown(f"""
        <div style="background:#0a0a1a;border:1px solid #00ffff;border-radius:10px;padding:16px 20px;margin-bottom:20px">
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

        source_path = getattr(source, "filepath", None)
        if source_path and os.path.exists(source_path):
            try:
                with open(source_path, "rb") as f:
                    st.audio(f.read(), format="audio/mp3")
            except Exception:
                pass

        if not similar:
            st.warning("No embedding found for this track — run your embedding pipeline first.")
            st.stop()

        st.markdown(f"### {len(similar)} similar tracks")
        _render_results(similar, source)


# =============================================================================
# TAB 2 — Semantic / natural language search
# =============================================================================
with tab_describe:
    st.caption(
        'Describe the vibe you want. Mention a count if you like: "6 upbeat deep house tracks".\n'
        "Untagged tracks are included via sound similarity when tagged matches run short."
    )

    query_input = st.text_input(
        "What do you want to play next?",
        placeholder='e.g. "upbeat deep house", "dark minimal techno, give me 8"',
        key="semantic_query",
    )
    describe_btn = st.button("Search", type="primary", key="describe_btn")
    st.divider()

    if describe_btn and query_input.strip():
        # Lazy-import to avoid hard dependency when only using Tab 1
        try:
            from Backend.llm_interpreter import SemanticInterpreter, InterpretationContext
        except ImportError:
            st.error("SemanticInterpreter not found — check Backend/semantic_interpreter.py")
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

        # Relaxation notice
        step = meta.get("relaxation_step", 0)
        if step > 0:
            label = meta.get("relaxation_label", "")
            st.markdown(
                f'<div class="relax-notice">⚡ No exact matches — showing best results '
                f'after widening search ({label})</div>',
                unsafe_allow_html=True,
            )

        if meta.get("inferred_count", 0) > 0:
            n_inf = meta["inferred_count"]
            st.markdown(
                f'<div class="relax-notice">🔮 {n_inf} track(s) inferred from sound '
                f'similarity — not yet manually tagged</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f"**{len(tracks)} tracks** · _{params.get('reasoning', '')}_ "
            f"· confidence {params.get('confidence', 0):.0%} "
            f"· `{params.get('model_used', 'fallback')}`"
        )

        # Convert to (score, track_dict) pairs that _render_results expects
        pairs = [(t.get("_relevance_score", 0.5), t) for t in tracks]
        _render_results(pairs, source=None)


