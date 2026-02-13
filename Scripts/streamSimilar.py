"""
DJMate — Similar Tracks Explorer
Run: streamlit run similar_tracks_tool.py

Pick a track (with autocomplete from your DB), get 7 similar tracks,
each described in terms of direction: "more tech house", "lower energy", etc.
"""

import streamlit as st
import asyncio
import os
import sys
import numpy as np
from dotenv import load_dotenv

# Walk up from this file until we find the .env (handles any folder depth)
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

# ── Page config ───────────────────────────────────────────────────────────────
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
  .tag { display: inline-block; background: #1a1a3a; color: #6666cc;
         border-radius: 10px; padding: 2px 7px; font-size: 10px; margin: 2px; }
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
    """Returns list of (display_label, trackid) tuples for the selectbox."""
    if not db.client:
        return []
    try:
        resp = db.client.table("tracks") \
            .select("trackid, title, artist, bpm, key") \
            .order("artist") \
            .execute()
        rows = resp.data or []
        results = []
        for r in rows:
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
    """
    Compare two tracks and return a plain-English direction description
    plus a colour for the badge.

    Returns: (description, hex_colour)
    """
    clues = []

    # Energy direction
    s_energy = getattr(source,    "energy", None) or 0.5
    c_energy = getattr(candidate, "energy", None) or 0.5
    diff_e = c_energy - s_energy
    if diff_e > 0.15:
        clues.append(("higher energy", "#ff4444"))
    elif diff_e < -0.15:
        clues.append(("lower energy",  "#4488ff"))

    # BPM direction
    s_bpm = getattr(source,    "bpm", None)
    c_bpm = getattr(candidate, "bpm", None)
    if s_bpm and c_bpm:
        diff_b = c_bpm - s_bpm
        if diff_b > 5:
            clues.append((f"+{int(diff_b)} BPM faster", "#ffaa00"))
        elif diff_b < -5:
            clues.append((f"{int(diff_b)} BPM slower", "#aaaaff"))

    # Tag direction (what unique genres does the candidate add?)
    s_tags = set(getattr(source,    "semantic_tags", None) or [])
    c_tags = set(getattr(candidate, "semantic_tags", None) or [])
    new_tags = c_tags - s_tags
    if new_tags:
        label = next(iter(new_tags))          # just show the most distinct one
        clues.append((f"more {label}", "#00ff88"))

    # Vibe direction
    s_vibes = set(getattr(source,    "vibe_descriptors", None) or [])
    c_vibes = set(getattr(candidate, "vibe_descriptors", None) or [])
    new_vibes = c_vibes - s_vibes
    if new_vibes:
        clues.append((next(iter(new_vibes)), "#cc88ff"))

    if not clues:
        return ("similar vibe", "#555555")

    # Pick the most "interesting" clue (prefer genre > energy > BPM)
    for desc, colour in clues:
        if "more " in desc:
            return (desc, colour)
    return clues[0]

def cosine_similarity(v1, v2) -> float:
    try:
        a, b = np.array(v1, dtype=np.float32), np.array(v2, dtype=np.float32)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    except Exception:
        return 0.0

# ── Main similarity search ────────────────────────────────────────────────────
async def find_similar(source_id: str, n: int = 7):
    source = await db.get_track_by_id(source_id)
    if not source:
        return None, []

    embedding = getattr(source, "embedding", None)
    if not embedding:
        return source, []           # No embedding — can't do similarity

    raw = await db.find_similar_tracks(
        query_embedding=embedding,
        limit=n + 1,
        threshold=0.2,
    )

    # Exclude the source itself
    results = [r for r in raw if str(r.get("id") or r.get("trackid")) != str(source_id)][:n]

    # Enrich each result with full metadata for direction analysis
    enriched = []
    for r in results:
        rid = r.get("id") or r.get("trackid")
        full = await db.get_track_by_id(str(rid))
        sim  = float(r.get("similarity", 0.5))
        if full:
            # Re-compute cosine similarity if we have both embeddings
            c_emb = getattr(full, "embedding", None)
            if c_emb and embedding:
                sim = cosine_similarity(embedding, c_emb)
            enriched.append((sim, full))
        else:
            enriched.append((sim, r))       # fallback to raw dict

    enriched.sort(key=lambda x: x[0], reverse=True)
    return source, enriched

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔀 DJMate — Similar Tracks Explorer")
st.caption("Pick a track and see 7 similar tracks with directional descriptions.")

track_index = load_track_index()

if not track_index:
    st.error("No tracks loaded — check your Supabase connection.")
    st.stop()

col1, col2 = st.columns([3, 1])

with col1:
    labels    = [t[0] for t in track_index]
    track_ids = [t[1] for t in track_index]

    # Searchable selectbox
    selected_label = st.selectbox(
        "Search for a track:",
        options=labels,
        index=None,
        placeholder="Type to search...",
        key="track_select"
    )

    selected_id = None
    if selected_label:
        selected_id = track_ids[labels.index(selected_label)]

with col2:
    n_similar = st.slider("Results", 4, 12, 7)
    search_btn = st.button("Find Similar Tracks", type="primary", use_container_width=True)

st.divider()

if search_btn and selected_id:
    with st.spinner("Computing similarity..."):
        source, similar = run_async(find_similar(selected_id, n_similar))

    if source is None:
        st.error("Track not found.")
        st.stop()

    # ── Source track card ─────────────────────────────────────────────────────
    s_title  = getattr(source, "title",  None) or "Unknown"
    s_artist = getattr(source, "artist", None) or "Unknown"
    s_bpm    = getattr(source, "bpm",    None)
    s_key    = getattr(source, "key",    None)
    s_energy = getattr(source, "energy", None)
    s_tags   = getattr(source, "semantic_tags", None) or []

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

    # Source track audio
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

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown(f"### {len(similar)} similar tracks")

    for i, (sim_score, track) in enumerate(similar, 1):
        if hasattr(track, "title"):
            title  = track.title or "Unknown"
            artist = track.artist or "Unknown"
            bpm    = track.bpm
            key    = track.key
            energy = track.energy
            tags_l = track.semantic_tags or []
            filepath = getattr(track, "filepath", None)
        elif isinstance(track, dict):
            title  = track.get("title")  or "Unknown"
            artist = track.get("artist") or "Unknown"
            bpm    = track.get("bpm")
            key    = track.get("key")
            energy = track.get("energy")
            tags_l = track.get("semantic_tags") or []
            filepath = track.get("filepath")
        else:
            title = artist = "Unknown"
            bpm = key = energy = None
            tags_l = []
            filepath = None

        direction, d_colour = describe_direction(source, track)

        # Similarity bar colour (green → yellow → red)
        bar_pct   = int(sim_score * 100)
        bar_colour = "#00ff88" if sim_score > 0.7 else "#ffaa00" if sim_score > 0.4 else "#ff4444"

        meta_parts = []
        if bpm:    meta_parts.append(f"{int(bpm)} BPM")
        if key:    meta_parts.append(key)
        if energy: meta_parts.append(f"energy {energy:.2f}")
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
              <div>
                <span class="direction-badge" style="background:{d_colour}22;color:{d_colour};border:1px solid {d_colour}66">
                  ↗ {direction}
                </span>
              </div>
              <div class="sim-bar" style="width:{bar_pct}%;background:{bar_colour}"></div>
              <div style="font-size:10px;color:#555;margin-top:2px">{bar_pct}% similarity</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Native Streamlit audio playback (supports local file paths)
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    st.audio(f.read(), format="audio/mp3")
            except Exception:
                pass