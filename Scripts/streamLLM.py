"""
DJMate — LLM Query Tool with Debug Mode
Run: streamlit run streamLLM.py

Enhanced with detailed debugging to troubleshoot matching issues.
"""

import streamlit as st
import asyncio
import os
import sys
from dotenv import load_dotenv
import json

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

# Scripts/ dir on sys.path so `from Backend.xxx` works
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

from Backend.llm_interpreter import SemanticInterpreter, InterpretationContext
from Backend.data.db_interface import DatabaseManager

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DJMate — LLM Query (Debug)",
    page_icon="🎧",
    layout="wide",
)

st.markdown("""
<style>
  .track-card {
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
  }
  .track-title { font-size: 15px; font-weight: bold; color: #fff; }
  .track-meta  { font-size: 12px; color: #aaa; margin-top: 3px; }
  .tag         { display: inline-block; background: #2a2a4a; color: #8888ff;
                 border-radius: 12px; padding: 2px 8px; font-size: 11px; margin: 2px; }
  .reasoning   { font-size: 12px; color: #00ff88; margin-top: 4px; font-style: italic; }
  .chip        { display: inline-block; background: #1a2a1a; color: #00ff88;
                 border: 1px solid #00ff88; border-radius: 12px;
                 padding: 3px 10px; font-size: 12px; margin: 3px; }
  .confidence-high { color: #00ff88; }
  .confidence-med  { color: #ffaa00; }
  .confidence-low  { color: #ff6666; }
  .debug-box {
    background: #0a0a0a;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 10px;
    margin: 10px 0;
    font-family: monospace;
    font-size: 11px;
  }
  .debug-title {
    color: #ffaa00;
    font-weight: bold;
    margin-bottom: 5px;
  }
  .debug-success { color: #00ff88; }
  .debug-warning { color: #ffaa00; }
  .debug-error { color: #ff6666; }
</style>
""", unsafe_allow_html=True)

# ── Helper: run async functions from Streamlit ────────────────────────────────
def run_async(coro):
    """Execute async coroutine in Streamlit context"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)

# ── Simplified Initialization (Supabase Only!) ────────────────────────────────
@st.cache_resource
def get_db():
    """Get database manager instance"""
    return DatabaseManager()

@st.cache_resource
def get_interpreter(_db):
    """Get interpreter instance - only needs Supabase client!"""

    if not _db.client:
        st.error("❌ Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")
        st.stop()

    # Create interpreter with Supabase client (no PostgreSQL needed!)
    interpreter = SemanticInterpreter(supabase_client=_db.client)

    # Initialize to load tags
    try:
        run_async(interpreter.initialize())
        st.success(f"✅ Loaded {len(interpreter.available_tags.semantic_tags)} tags, {len(interpreter.available_tags.vibes)} vibes")
    except Exception as e:
        st.error(f"❌ Failed to load tags: {e}")
        st.info("💡 Check that your track_labels table exists with columns: semantic_tags, vibe, energy")

    return interpreter

# Initialize
try:
    db = get_db()
    interpreter = get_interpreter(db)
except Exception as e:
    st.error(f"❌ Initialization failed: {e}")
    st.stop()

# ── Enhanced Query Pipeline with Debug Info ──────────────────────────────────
async def query_tracks_with_debug(user_query: str, n_results: int = 8, debug_mode: bool = False):
    """Parse intent → filter DB → return scored tracks WITH DEBUG INFO."""

    debug_info = {
        "steps": [],
        "warnings": [],
        "errors": []
    }

    # Ensure interpreter is initialized
    if not interpreter._tags_loaded:
        await interpreter.initialize()
        debug_info["steps"].append("🔄 Loaded interpreter tags")

    # STEP 1: Parse natural language
    debug_info["steps"].append("📝 Step 1: Interpreting natural language query...")
    structured = await interpreter.interpret(user_query)

    debug_info["llm_interpretation"] = {
        "semantic_tags": structured.get("semantic_tags", []),
        "vibes": structured.get("vibes", []),
        "energy_range": structured.get("energy_range"),
        "bpm_range": structured.get("bpm_range"),
        "direction": structured.get("direction"),
        "confidence": structured.get("confidence", 0),
        "reasoning": structured.get("reasoning", ""),
        "suggestions": structured.get("suggestions", [])
    }

    # Check if tags match what's in database
    requested_tags = set(structured.get("semantic_tags", []))
    available_tags = interpreter.available_tags.semantic_tags
    matched_tags = requested_tags & available_tags
    unmatched_tags = requested_tags - available_tags

    requested_vibes = set(structured.get("vibes", []))
    available_vibes = interpreter.available_tags.vibes
    matched_vibes = requested_vibes & available_vibes
    unmatched_vibes = requested_vibes - available_vibes

    if unmatched_tags:
        debug_info["warnings"].append(f"⚠️ Requested tags not in database: {list(unmatched_tags)}")
    if unmatched_vibes:
        debug_info["warnings"].append(f"⚠️ Requested vibes not in database: {list(unmatched_vibes)}")

    if matched_tags:
        debug_info["steps"].append(f"✅ Matched {len(matched_tags)} tags: {list(matched_tags)}")
    else:
        debug_info["warnings"].append("⚠️ NO semantic tags matched!")

    if matched_vibes:
        debug_info["steps"].append(f"✅ Matched {len(matched_vibes)} vibes: {list(matched_vibes)}")

    # STEP 2: Build database filter
    debug_info["steps"].append("🔍 Step 2: Building database query...")
    db_filter = {
        "semantic_tags": structured.get("semantic_tags", []),
        "vibes": structured.get("vibes", []),
        "energy_range": structured.get("energy_range"),
        "bpm_range": structured.get("bpm_range"),
        "key_compatibility": structured.get("key_compatibility"),
    }

    debug_info["database_query"] = db_filter

    # STEP 3: Get candidates from Supabase
    debug_info["steps"].append("💾 Step 3: Querying database...")
    try:
        candidates = await db.get_tracks_by_semantic_filter(db_filter, limit=200)
        debug_info["steps"].append(f"📊 Found {len(candidates)} candidate tracks from database")

        if not candidates:
            debug_info["errors"].append("❌ Database returned 0 tracks")
            debug_info["errors"].append("💡 Check if track_labels table has data")
            debug_info["errors"].append("💡 Try broader search terms")
            return structured, [], debug_info

    except Exception as e:
        debug_info["errors"].append(f"❌ Database query failed: {str(e)}")
        return structured, [], debug_info

    # STEP 4: Score and rank candidates
    debug_info["steps"].append("🎯 Step 4: Scoring and ranking tracks...")
    scored = []
    semantic_tags = set(structured.get("semantic_tags", []))
    vibes = set(structured.get("vibes", []))
    energy_range = structured.get("energy_range")
    bpm_range = structured.get("bpm_range")

    scoring_breakdown = {
        "tracks_with_tag_match": 0,
        "tracks_with_vibe_match": 0,
        "tracks_with_energy_match": 0,
        "tracks_with_bpm_match": 0,
        "tracks_scored_above_0": 0
    }

    for t in candidates:
        score = 0.0
        reasons = []

        # Semantic tag match (higher weight)
        t_tags = set(getattr(t, "semantic_tags", None) or [])
        overlap = semantic_tags & t_tags
        if overlap:
            score += len(overlap) * 0.4
            reasons.append(f"tags: {', '.join(overlap)}")
            scoring_breakdown["tracks_with_tag_match"] += 1

        # Vibe match - handle both single value and list
        t_vibe_descriptors = getattr(t, "vibe_descriptors", None) or []
        if isinstance(t_vibe_descriptors, str):
            t_vibe_descriptors = [t_vibe_descriptors]
        t_vibes = set(t_vibe_descriptors)

        vibe_overlap = vibes & t_vibes
        if vibe_overlap:
            score += len(vibe_overlap) * 0.3
            reasons.append(f"vibe: {', '.join(vibe_overlap)}")
            scoring_breakdown["tracks_with_vibe_match"] += 1

        # Energy match
        energy = getattr(t, "energy", None)
        if energy_range and energy is not None:
            mid = sum(energy_range) / 2
            diff = abs(energy - mid)
            e_score = max(0, 1 - diff * 3)
            score += e_score * 0.2
            if e_score > 0.6:
                reasons.append(f"energy {energy:.2f} ≈ {mid:.2f}")
                scoring_breakdown["tracks_with_energy_match"] += 1

        # BPM match
        bpm = getattr(t, "bpm", None)
        if bpm_range and bpm:
            if bpm_range[0] <= bpm <= bpm_range[1]:
                score += 0.1
                reasons.append(f"{int(bpm)} BPM in range")
                scoring_breakdown["tracks_with_bpm_match"] += 1

        if score > 0:
            scoring_breakdown["tracks_scored_above_0"] += 1

        scored.append((score, t, reasons))

    debug_info["scoring_breakdown"] = scoring_breakdown

    if scoring_breakdown["tracks_scored_above_0"] == 0:
        debug_info["errors"].append("❌ NO tracks scored above 0 - no matches found!")
        if scoring_breakdown["tracks_with_tag_match"] == 0:
            debug_info["errors"].append("💡 No tracks matched requested tags - try different genres")
        if scoring_breakdown["tracks_with_vibe_match"] == 0 and vibes:
            debug_info["errors"].append("💡 No tracks matched requested vibes")
    else:
        debug_info["steps"].append(f"✅ {scoring_breakdown['tracks_scored_above_0']} tracks scored above 0")

    scored.sort(key=lambda x: x[0], reverse=True)
    final_results = scored[:n_results]

    debug_info["steps"].append(f"🎵 Returning top {len(final_results)} tracks")

    return structured, final_results, debug_info

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🎧 DJMate — LLM Query Tool")
st.caption("Describe a vibe, genre, or energy level and get matching tracks from your library.")

# Show tag statistics in sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    n_results = st.slider("Number of results", 4, 20, 8, key="num_results_slider")

    # Debug mode toggle
    debug_mode = st.checkbox("🐛 Debug Mode", value=True, help="Show detailed debugging information")

    with st.expander("🏷️ Available Tags in Database"):
        if st.button("Refresh Tags", key="refresh_tags_btn"):
            with st.spinner("Loading..."):
                stats = run_async(interpreter.get_tag_statistics())
                st.session_state["tag_stats"] = stats

        if interpreter._tags_loaded:
            st.caption(f"**{len(interpreter.available_tags.semantic_tags)} Semantic Tags:**")
            # Show all tags in a compact format
            all_tags = sorted(list(interpreter.available_tags.semantic_tags))
            for i in range(0, len(all_tags), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i+j < len(all_tags):
                        col.caption(f"• {all_tags[i+j]}")

            st.divider()

            st.caption(f"**{len(interpreter.available_tags.vibes)} Vibes:**")
            all_vibes = sorted(list(interpreter.available_tags.vibes))
            for vibe in all_vibes:
                st.caption(f"• {vibe}")

        if "tag_stats" in st.session_state:
            st.divider()
            stats = st.session_state["tag_stats"]
            st.caption("**Most Common Tags:**")
            for tag_info in stats['most_common_tags'][:10]:
                st.caption(f"• {tag_info['tag']}: {tag_info['count']}")

    st.divider()
    st.markdown("**💡 Example queries:**")
    examples = [
        "Give me 6 dark techno tracks",
        "High energy bangers around 130 BPM",
        "Chill late-night deep house",
        "Groovy tech house for peak time",
        "Melodic and uplifting progressive",
        "Underground minimal around 125 BPM",
        "Something dreamy and hypnotic",
        "Intense driving techno"
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"example_{ex[:20]}"):
            st.session_state["prefill"] = ex

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            st.markdown(msg["content"], unsafe_allow_html=True)

# Input box
prefill = st.session_state.pop("prefill", "")
user_input = st.chat_input("Describe the vibe you want...", )

query = user_input or (prefill if prefill else None)

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Run query with debug info
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching your library..."):
            structured, results, debug_info = run_async(query_tracks_with_debug(query, n_results, debug_mode))

        # DEBUG MODE: Show detailed debugging information
        if debug_mode:
            st.markdown("### 🐛 Debug Information")

            # Step-by-step process
            debug_html = '<div class="debug-box">'
            debug_html += '<div class="debug-title">🔍 Query Processing Steps:</div>'
            for step in debug_info.get("steps", []):
                debug_html += f'<div>{step}</div>'
            debug_html += '</div>'

            # LLM Interpretation
            debug_html += '<div class="debug-box">'
            debug_html += '<div class="debug-title">🤖 LLM Interpretation:</div>'
            llm_interp = debug_info.get("llm_interpretation", {})
            debug_html += f'<div><strong>Semantic Tags:</strong> {llm_interp.get("semantic_tags", [])}</div>'
            debug_html += f'<div><strong>Vibes:</strong> {llm_interp.get("vibes", [])}</div>'
            debug_html += f'<div><strong>Energy Range:</strong> {llm_interp.get("energy_range")}</div>'
            debug_html += f'<div><strong>BPM Range:</strong> {llm_interp.get("bpm_range")}</div>'
            debug_html += f'<div><strong>Confidence:</strong> {llm_interp.get("confidence", 0):.2%}</div>'
            if llm_interp.get("reasoning"):
                debug_html += f'<div><strong>Reasoning:</strong> {llm_interp.get("reasoning")}</div>'
            debug_html += '</div>'

            # Database Query
            debug_html += '<div class="debug-box">'
            debug_html += '<div class="debug-title">💾 Database Query Parameters:</div>'
            db_query = debug_info.get("database_query", {})
            debug_html += f'<div><pre>{json.dumps(db_query, indent=2)}</pre></div>'
            debug_html += '</div>'

            # Scoring Breakdown
            if "scoring_breakdown" in debug_info:
                debug_html += '<div class="debug-box">'
                debug_html += '<div class="debug-title">🎯 Scoring Breakdown:</div>'
                scoring = debug_info["scoring_breakdown"]
                debug_html += f'<div class="debug-success">✓ Tracks with tag match: {scoring.get("tracks_with_tag_match", 0)}</div>'
                debug_html += f'<div class="debug-success">✓ Tracks with vibe match: {scoring.get("tracks_with_vibe_match", 0)}</div>'
                debug_html += f'<div class="debug-success">✓ Tracks with energy match: {scoring.get("tracks_with_energy_match", 0)}</div>'
                debug_html += f'<div class="debug-success">✓ Tracks with BPM match: {scoring.get("tracks_with_bpm_match", 0)}</div>'
                debug_html += f'<div class="debug-success">✓ <strong>Total scored above 0: {scoring.get("tracks_scored_above_0", 0)}</strong></div>'
                debug_html += '</div>'

            # Warnings
            if debug_info.get("warnings"):
                debug_html += '<div class="debug-box">'
                debug_html += '<div class="debug-title">⚠️ Warnings:</div>'
                for warning in debug_info["warnings"]:
                    debug_html += f'<div class="debug-warning">{warning}</div>'
                debug_html += '</div>'

            # Errors
            if debug_info.get("errors"):
                debug_html += '<div class="debug-box">'
                debug_html += '<div class="debug-title">❌ Issues Found:</div>'
                for error in debug_info["errors"]:
                    debug_html += f'<div class="debug-error">{error}</div>'
                debug_html += '</div>'

            st.markdown(debug_html, unsafe_allow_html=True)
            st.divider()

        # Show interpretation chips
        chips_html = ""

        # Semantic tags
        for tag in structured.get("semantic_tags", []):
            chips_html += f'<span class="chip">{tag}</span>'

        # Vibes
        for vibe in structured.get("vibes", []):
            chips_html += f'<span class="chip" style="border-color:#aaaaff;color:#aaaaff">{vibe}</span>'

        # BPM range
        if structured.get("bpm_range"):
            b = structured["bpm_range"]
            chips_html += f'<span class="chip" style="border-color:#ffaa00;color:#ffaa00">{int(b[0])}–{int(b[1])} BPM</span>'

        # Energy range
        if structured.get("energy_range"):
            e = structured["energy_range"]
            chips_html += f'<span class="chip" style="border-color:#ff6666;color:#ff6666">energy {e[0]:.1f}–{e[1]:.1f}</span>'

        # Direction
        if structured.get("direction"):
            dir_icon = {"build": "📈", "maintain": "➡️", "breakdown": "📉"}.get(structured["direction"], "")
            chips_html += f'<span class="chip" style="border-color:#88ff88;color:#88ff88">{dir_icon} {structured["direction"]}</span>'

        # Confidence score with color coding
        conf = structured.get("confidence", 0)
        conf_class = "confidence-high" if conf > 0.8 else "confidence-med" if conf > 0.5 else "confidence-low"
        conf_text = f'<span class="{conf_class}">({int(conf*100)}% confidence)</span>'

        # Reasoning
        reasoning_text = ""
        if structured.get("reasoning"):
            reasoning_text = f'<div style="font-size:11px;color:#888;margin-top:4px;font-style:italic">💭 {structured["reasoning"]}</div>'

        if chips_html:
            st.markdown(
                f'<div style="margin-bottom:12px"><strong>Interpreted as:</strong> {chips_html} {conf_text}{reasoning_text}</div>',
                unsafe_allow_html=True,
            )

        # Show suggestions if confidence is low
        if conf < 0.6 and structured.get("suggestions"):
            sugg_html = '<div style="font-size:12px;color:#ffaa00;margin-bottom:8px">💡 Suggestions: '
            sugg_html += ', '.join(f'<span style="border:1px solid #ffaa00;padding:2px 6px;border-radius:4px;margin:2px">{s}</span>'
                                   for s in structured["suggestions"][:3])
            sugg_html += '</div>'
            st.markdown(sugg_html, unsafe_allow_html=True)

        if not results:
            response_html = "<p style='color:#ff4444'>❌ No matching tracks found.</p>"
            if debug_mode:
                response_html += "<p style='color:#aaa;font-size:12px'>Check the debug information above to see why no matches were found.</p>"
            else:
                response_html += "<p style='color:#aaa;font-size:12px'>💡 Enable Debug Mode in the sidebar to see detailed diagnostics.</p>"
            st.markdown(response_html, unsafe_allow_html=True)
        else:
            cards = ""
            for i, (score, track, reasons) in enumerate(results, 1):
                title = getattr(track, "title", None) or "Unknown"
                artist = getattr(track, "artist", None) or "Unknown"
                bpm = getattr(track, "bpm", None)
                key = getattr(track, "key", None)
                energy = getattr(track, "energy", None)
                tags_l = getattr(track, "semantic_tags", None) or []

                # Handle vibe as single value or list
                vibe_desc = getattr(track, "vibe_descriptors", None)
                if isinstance(vibe_desc, list):
                    vibe = vibe_desc[0] if vibe_desc else None
                else:
                    vibe = vibe_desc

                meta_parts = []
                if bpm: meta_parts.append(f"{int(bpm)} BPM")
                if key: meta_parts.append(key)
                if energy: meta_parts.append(f"energy {energy:.2f}")
                if vibe: meta_parts.append(f"🎭 {vibe}")
                meta = " · ".join(meta_parts)

                tags_html = "".join(f'<span class="tag">{t}</span>' for t in tags_l[:4])
                reasoning = ", ".join(reasons) if reasons else "general match"

                # Score indicator
                score_color = "#00ff88" if score > 0.5 else "#ffaa00" if score > 0.3 else "#888"
                score_width = min(100, int(score * 100))

                cards += f"""
                <div class="track-card">
                  <div style="display:flex;align-items:center;gap:10px">
                    <div style="font-size:20px;color:#555;width:28px;text-align:center">{i}</div>
                    <div style="flex:1">
                      <div class="track-title">{title}</div>
                      <div class="track-meta">{artist}{(' — ' + meta) if meta else ''}</div>
                      <div style="margin-top:5px">{tags_html}</div>
                      <div class="reasoning">↳ {reasoning}</div>
                      <div style="margin-top:4px;height:3px;background:#222;border-radius:2px">
                        <div style="width:{score_width}%;height:100%;background:{score_color};border-radius:2px"></div>
                      </div>
                    </div>
                    <div style="font-size:13px;color:#555;font-weight:bold">{score:.2f}</div>
                  </div>
                </div>"""

            response_html = f"<p style='color:#aaa;margin-bottom:12px'>✨ Found {len(results)} tracks:</p>" + cards
            st.markdown(response_html, unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": response_html})

# ── Footer with interpreter info ──────────────────────────────────────────────
st.divider()
col1, col2, col3, col4 = st.columns(4)
with col1:
    if interpreter._tags_loaded:
        st.metric("Semantic Tags", len(interpreter.available_tags.semantic_tags))
with col2:
    if interpreter._tags_loaded:
        st.metric("Vibes", len(interpreter.available_tags.vibes))
with col3:
    st.metric("Connection", "Supabase ✓")
with col4:
    st.metric("Debug Mode", "🐛 ON" if debug_mode else "OFF")