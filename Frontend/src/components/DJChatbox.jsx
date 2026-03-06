// src/components/DJChatbox.jsx
import React, { useState, useRef, useEffect, useCallback, forwardRef, useImperativeHandle } from 'react';
import { apiClient } from '../api/apiClient';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ── SVG Icons ─────────────────────────────────────────────────────────────
const IconSend = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <line x1="22" y1="2" x2="11" y2="13" /><polygon points="22,2 15,22 11,13 2,9" />
  </svg>
);
const IconMinus = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true">
    <line x1="5" y1="12" x2="19" y2="12" />
  </svg>
);
const IconClose = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true">
    <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
  </svg>
);
const IconWave = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
    <path d="M2 12 Q4 4 6 12 Q8 20 10 12 Q12 4 14 12 Q16 20 18 12 Q20 4 22 12" />
  </svg>
);
const IconSearch = () => (
  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true">
    <circle cx="11" cy="11" r="7" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>
);
const IconUp = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true">
    <polyline points="18,15 12,9 6,15" />
  </svg>
);
const IconDown = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true">
    <polyline points="6,9 12,15 18,9" />
  </svg>
);

// ── Utility: direction delta between two tracks ────────────────────────────
function describeDirection(source, candidate) {
  const clues = [];
  const sE = parseFloat(source?.energy  ?? 0.5);
  const cE = parseFloat(candidate?.energy ?? 0.5);
  if (cE - sE >  0.15) clues.push({ label: 'higher energy', color: '#ef4444' });
  if (cE - sE < -0.15) clues.push({ label: 'lower energy',  color: '#0ea5e9' });

  const sB = source?.bpm, cB = candidate?.bpm;
  if (sB && cB) {
    const d = parseFloat(cB) - parseFloat(sB);
    if (d >  5) clues.push({ label: `+${Math.round(d)} BPM`, color: '#f59e0b' });
    if (d < -5) clues.push({ label: `${Math.round(d)} BPM`, color: '#818cf8' });
  }

  const newTags  = (candidate?.semantic_tags   || []).filter(t => !(source?.semantic_tags   || []).includes(t));
  const newVibes = (candidate?.vibe_descriptors|| []).filter(v => !(source?.vibe_descriptors|| []).includes(v));
  if (newTags.length)  clues.push({ label: newTags[0],  color: '#a855f7' });
  if (newVibes.length) clues.push({ label: newVibes[0], color: '#00d4ff' });

  if (!clues.length) return { label: 'similar vibe', color: '#475569' };
  return clues.find(c => c.label.startsWith('more ')) || clues[0];
}

// ── Similarity bar ─────────────────────────────────────────────────────────
function SimBar({ score }) {
  const pct   = Math.round(score * 100);
  const color = score > 0.7 ? '#00d4ff' : score > 0.4 ? '#7c3aed' : '#475569';
  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ height: 1, background: 'rgba(124,58,237,0.12)', overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: `linear-gradient(90deg, #7c3aed, ${color})`, transition: 'width 400ms ease' }} />
      </div>
      <div style={{ fontSize: 9, color: '#475569', marginTop: 3, fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.05em' }}>
        {pct}% MATCH
      </div>
    </div>
  );
}

// ── Supabase storage safe filename (mirrors upload_album_covers.py) ────────
const SUPABASE_COVERS_BASE = 'https://cvermotfxamubejfnoje.supabase.co/storage/v1/object/public/album-covers/';
function makeSupabaseCoverUrl(artist, title) {
  if (!artist || !title) return null;
  let safe = `${artist}_${title}`.toLowerCase();
  safe = safe.split('').map(c => /[a-z0-9\-_]/.test(c) ? c : '_').join('');
  safe = safe.split('_').filter(Boolean).join('_').slice(0, 150);
  return `${SUPABASE_COVERS_BASE}${safe}.jpg`;
}

// ── Album art: Supabase storage → iTunes → generative fallback ─────────────
function AlbumArt({ title, artist, directUrl, size = 48 }) {
  const [url, setUrl] = useState(directUrl || null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    setErr(false); // Always reset error when track changes
    if (directUrl) { setUrl(directUrl); return; }
    if (!title || !artist) return;
    // Try Supabase storage first
    const supabaseUrl = makeSupabaseCoverUrl(artist, title);
    if (supabaseUrl) { setUrl(supabaseUrl); return; }
  }, [directUrl, title, artist]);

  const handleImgError = () => {
    // Supabase 404 → fall through to iTunes
    if (url && url.includes('supabase')) {
      let cancelled = false;
      const term = encodeURIComponent(`${artist} ${title}`);
      fetch(`https://itunes.apple.com/search?term=${term}&entity=song&limit=1`)
        .then(r => r.json())
        .then(data => {
          if (cancelled) return;
          const raw = data.results?.[0]?.artworkUrl100;
          if (raw) setUrl(raw.replace('100x100bb', '300x300bb'));
          else setErr(true);
        })
        .catch(() => { if (!cancelled) setErr(true); });
      return () => { cancelled = true; };
    }
    setErr(true);
  };

  // Old iTunes-only effect removed — now we chain via onError
  useEffect(() => {}, [title, artist]);

  if (url && !err) {
    return (
      <img
        src={url}
        alt={`${title} artwork`}
        width={size}
        height={size}
        onError={handleImgError}
        style={{ width: size, height: size, objectFit: 'cover', display: 'block', flexShrink: 0 }}
      />
    );
  }

  // Generative fallback — initials + hashed gradient
  let hash = 0;
  for (let i = 0; i < ((title || '') + (artist || '')).length; i++) {
    hash = ((hash << 5) - hash) + ((title || '') + (artist || '')).charCodeAt(i);
    hash |= 0;
  }
  const hue = 200 + (Math.abs(hash) % 100);
  const initials = (title || '?').replace(/[^a-zA-Z0-9]/g, '').slice(0, 2).toUpperCase() || '??';

  return (
    <div style={{
      width: size, height: size, flexShrink: 0,
      background: `linear-gradient(135deg, hsl(${hue},50%,8%), hsl(${hue},70%,5%))`,
      border: `1px solid hsla(${hue},70%,50%,0.35)`,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: "'JetBrains Mono', monospace", fontSize: size * 0.28, fontWeight: 600,
      color: `hsla(${hue},80%,70%,0.9)`, letterSpacing: '0.02em',
    }}>
      {initials}
    </div>
  );
}

// ── Track card ─────────────────────────────────────────────────────────────
function TrackCard({ rank, track, score, source, onClick, onFindSimilar }) {
  const [audioError, setAudioError] = useState(false);
  const direction = source ? describeDirection(source, track) : null;

  const meta = [];
  if (track.bpm)          meta.push(`${Math.round(track.bpm)} BPM`);
  if (track.key)          meta.push(track.key);
  if (track.energy != null) meta.push(`${parseFloat(track.energy).toFixed(2)} NRG`);

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => onClick?.(track.trackid)}
      onKeyDown={e => e.key === 'Enter' && onClick?.(track.trackid)}
      style={{
        display: 'flex', gap: 10, alignItems: 'flex-start',
        background: 'rgba(8,8,20,0.6)',
        border: '1px solid rgba(124,58,237,0.14)',
        padding: '10px 12px',
        marginBottom: 3,
        cursor: onClick ? 'pointer' : 'default',
        transition: '150ms ease',
        position: 'relative',
        overflow: 'hidden',
      }}
      onMouseEnter={e => {
        e.currentTarget.style.borderColor = 'rgba(124,58,237,0.38)';
        e.currentTarget.style.background = 'rgba(12,12,28,0.8)';
      }}
      onMouseLeave={e => {
        e.currentTarget.style.borderColor = 'rgba(124,58,237,0.14)';
        e.currentTarget.style.background = 'rgba(8,8,20,0.6)';
      }}
    >
      {/* Hover left accent */}
      <div style={{
        position: 'absolute', left: 0, top: 0, bottom: 0, width: 1,
        background: 'linear-gradient(180deg, #7c3aed, #00d4ff)',
        opacity: 0, transition: '150ms ease',
      }} />

      {/* Rank */}
      <div style={{
        fontSize: 22, color: 'rgba(124,58,237,0.2)', fontWeight: 900,
        lineHeight: 1, minWidth: 22, fontFamily: "'JetBrains Mono', monospace",
        flexShrink: 0, paddingTop: 1,
      }}>
        {String(rank).padStart(2, '0')}
      </div>

      {/* Album art */}
      <AlbumArt title={track.title} artist={track.artist} directUrl={track.album_art_url || null} size={46} />

      {/* Main info */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{
          fontSize: 13, fontWeight: 700, color: '#e2e8f0',
          whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
          marginBottom: 1,
        }}>
          {track.title || 'Unknown'}
        </div>
        <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 5, letterSpacing: '0.01em' }}>
          {track.artist || 'Unknown'}
        </div>

        {/* Metadata */}
        {meta.length > 0 && (
          <div style={{
            fontSize: 9, color: '#475569', marginBottom: 6,
            fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.08em',
          }}>
            {meta.join(' · ')}
          </div>
        )}

        {/* Genre tags (purple) */}
        {track.semantic_tags?.length > 0 && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginBottom: 4 }}>
            {track.semantic_tags.slice(0, 4).map(tag => (
              <span key={tag} style={{
                background: 'rgba(124,58,237,0.1)', color: 'rgba(168,85,247,0.8)',
                border: '1px solid rgba(124,58,237,0.22)',
                padding: '1px 6px', fontSize: 9, letterSpacing: '0.04em',
                fontFamily: "'JetBrains Mono', monospace",
              }}>{tag}</span>
            ))}
          </div>
        )}

        {/* Vibe tags (cyan) */}
        {track.vibe_descriptors?.length > 0 && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginBottom: 4 }}>
            {track.vibe_descriptors.slice(0, 4).map(v => (
              <span key={v} style={{
                background: 'rgba(0,212,255,0.07)', color: 'rgba(0,212,255,0.7)',
                border: '1px solid rgba(0,212,255,0.18)',
                padding: '1px 6px', fontSize: 9, letterSpacing: '0.04em',
                fontFamily: "'JetBrains Mono', monospace",
              }}>{v}</span>
            ))}
          </div>
        )}

        {/* Badges */}
        <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap', alignItems: 'center', marginBottom: 2 }}>
          {track.inferred && (
            <span style={{
              background: 'rgba(245,158,11,0.08)', color: 'rgba(245,158,11,0.7)',
              border: '1px solid rgba(245,158,11,0.2)', padding: '1px 6px', fontSize: 9,
              fontFamily: 'monospace',
            }}>INFERRED</span>
          )}
          {direction && (
            <span style={{
              background: `${direction.color}12`, color: direction.color,
              border: `1px solid ${direction.color}38`,
              padding: '1px 7px', fontSize: 9, fontWeight: 600,
              fontFamily: 'monospace', letterSpacing: '0.04em',
            }}>{direction.label}</span>
          )}

          {/* Find similar */}
          {onFindSimilar && (
            <button
              onClick={e => { e.stopPropagation(); onFindSimilar(track.trackid); }}
              aria-label={`Find similar to ${track.title}`}
              style={{
                display: 'flex', alignItems: 'center', gap: 4,
                padding: '1px 7px',
                background: 'transparent',
                border: '1px solid rgba(0,212,255,0.22)',
                color: 'rgba(0,212,255,0.6)',
                cursor: 'pointer', fontSize: 9, fontFamily: 'inherit',
                letterSpacing: '0.04em', transition: '150ms ease',
              }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(0,212,255,0.5)'; e.currentTarget.style.color = '#00d4ff'; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(0,212,255,0.22)'; e.currentTarget.style.color = 'rgba(0,212,255,0.6)'; }}
            >
              <IconSearch /> SIMILAR
            </button>
          )}
        </div>

        <SimBar score={score} />
      </div>

      {/* Audio */}
      <div style={{ flexShrink: 0, display: 'flex', alignItems: 'center' }}>
        {(track.audio_url || track.trackid) && !audioError ? (
          <audio
            controls
            src={track.audio_url || `${API_BASE}/tracks/${track.trackid}/audio`}
            onError={() => setAudioError(true)}
            style={{ width: 120, height: 28, accentColor: '#7c3aed', opacity: 0.7 }}
          />
        ) : (
          <div style={{ fontSize: 9, color: '#2d3748', fontFamily: 'monospace', width: 50, textAlign: 'center' }}>
            {track.trackid ? 'NO AUDIO' : '—'}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Alternate candidates bar ───────────────────────────────────────────────
function CandidateBar({ candidates, onSelect }) {
  if (!candidates || candidates.length <= 1) return null;
  return (
    <div style={{
      background: 'rgba(8,8,24,0.8)', border: '1px solid rgba(124,58,237,0.2)',
      padding: '8px 12px', marginBottom: 8,
    }}>
      <div style={{ fontSize: 8, letterSpacing: '0.25em', color: 'rgba(124,58,237,0.6)', marginBottom: 6, fontFamily: 'monospace' }}>
        DID YOU MEAN
      </div>
      {candidates.slice(1).map((c, i) => (
        <div
          key={c.trackid}
          role="button"
          tabIndex={0}
          onClick={() => onSelect(c.trackid, c.title)}
          onKeyDown={e => e.key === 'Enter' && onSelect(c.trackid, c.title)}
          style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '5px 0', cursor: 'pointer', transition: '150ms ease',
            borderBottom: i < candidates.length - 2 ? '1px solid rgba(124,58,237,0.08)' : 'none',
          }}
          onMouseEnter={e => e.currentTarget.style.opacity = '0.65'}
          onMouseLeave={e => e.currentTarget.style.opacity = '1'}
        >
          <div>
            <span style={{ fontSize: 12, color: '#94a3b8' }}>{c.title}</span>
            <span style={{ fontSize: 10, color: '#475569', marginLeft: 8 }}>{c.artist}</span>
          </div>
          <span style={{ fontSize: 9, color: 'rgba(124,58,237,0.6)', fontFamily: 'monospace' }}>
            {Math.round(c.match_score * 100)}%
          </span>
        </div>
      ))}
    </div>
  );
}

// ── Search results ─────────────────────────────────────────────────────────
function SearchResults({ result, source, onTrackClick, onFindSimilar, onCandidateSelect }) {
  if (!result) return null;
  const isSimilar = result.intent === 'find_similar_track';

  return (
    <div style={{ marginTop: 10, animation: 'fadeIn 200ms ease' }}>

      {/* Track not found */}
      {isSimilar && result.tracks.length === 0 && (
        <div style={{
          background: 'rgba(16,5,5,0.8)', border: '1px solid rgba(239,68,68,0.25)',
          padding: '10px 12px', marginBottom: 8, fontSize: 11, color: 'rgba(239,68,68,0.8)',
        }}>
          {result.reasoning || "Track not found in library."}
          <div style={{ fontSize: 9, color: '#475569', marginTop: 4, fontFamily: 'monospace' }}>
            Try the full track name, or use vibe / genre search.
          </div>
        </div>
      )}

      {/* Candidates */}
      {isSimilar && result.track_candidates && (
        <CandidateBar candidates={result.track_candidates} onSelect={onCandidateSelect} />
      )}

      {/* Widened search notice */}
      {!isSimilar && result.relaxation_step > 0 && (
        <div style={{
          background: 'rgba(10,8,0,0.7)', border: '1px solid rgba(245,158,11,0.25)',
          padding: '6px 12px', marginBottom: 8, fontSize: 10,
          color: 'rgba(245,158,11,0.7)', fontFamily: 'monospace', letterSpacing: '0.04em',
        }}>
          ⟳ WIDENED SEARCH — {result.relaxation_label}
        </div>
      )}

      {result.inferred_count > 0 && (
        <div style={{
          background: 'rgba(10,8,0,0.7)', border: '1px solid rgba(245,158,11,0.2)',
          padding: '6px 12px', marginBottom: 8, fontSize: 10,
          color: 'rgba(245,158,11,0.6)', fontFamily: 'monospace',
        }}>
          {result.inferred_count} TRACK{result.inferred_count > 1 ? 'S' : ''} INFERRED BY SIMILARITY
        </div>
      )}

      {/* Summary */}
      {result.tracks.length > 0 && (
        <div style={{
          fontSize: 10, color: '#475569', marginBottom: 8,
          display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center',
          fontFamily: "'JetBrains Mono', monospace",
        }}>
          <span style={{ color: '#94a3b8', fontWeight: 600 }}>{result.tracks.length} TRACKS</span>
          {result.confidence > 0 && (
            <span>· <span style={{ color: '#7c3aed' }}>{Math.round(result.confidence * 100)}% CONF</span></span>
          )}
          {result.reasoning && (
            <span style={{ color: '#2d3748', fontSize: 9 }}>· {result.reasoning}</span>
          )}
          {result.model_used && (
            <span style={{
              background: 'rgba(8,8,20,0.8)', border: '1px solid rgba(124,58,237,0.12)',
              padding: '1px 6px', fontSize: 8, color: '#2d3748', letterSpacing: '0.04em',
            }}>{result.model_used}</span>
          )}
        </div>
      )}

      {/* Track cards */}
      {result.tracks.map((track, i) => (
        <React.Fragment key={track.trackid || i}>
          {isSimilar && i === 0 && (
            <div style={{ fontSize: 8, color: '#00d4ff', letterSpacing: '0.25em', marginBottom: 5, fontFamily: 'monospace', opacity: 0.7 }}>
              ◈ MATCHED TRACK
            </div>
          )}
          {isSimilar && i === 1 && (
            <div style={{ fontSize: 8, color: '#475569', letterSpacing: '0.25em', margin: '12px 0 5px', fontFamily: 'monospace' }}>
              SIMILAR TRACKS
            </div>
          )}
          <TrackCard
            rank={i + 1}
            track={track}
            score={track.relevance_score ?? 0.5}
            source={isSimilar && i === 0 ? null : (isSimilar ? result.tracks[0] : source)}
            onClick={onTrackClick}
            onFindSimilar={onFindSimilar}
          />
        </React.Fragment>
      ))}
    </div>
  );
}

// ── Quick prompts ──────────────────────────────────────────────────────────
const QUICK_PROMPTS = [
  { label: 'DEEP HOUSE', color: '#00d4ff' },
  { label: '140 BPM', color: '#a855f7' },
  { label: 'MINIMAL TECHNO', color: '#7c3aed' },
  { label: 'MINOR KEY', color: '#00d4ff' },
];

// ── Beacon button ──────────────────────────────────────────────────────────
function BeaconButton({ onClick }) {
  return (
    <button
      onClick={onClick}
      aria-label="Open DJMate Search"
      title="Open Search Terminal"
      style={{
        position: 'fixed', bottom: 48, left: 16, zIndex: 1000,
        width: 44, height: 44,
        background: 'rgba(5,5,7,0.96)',
        border: '1px solid rgba(124,58,237,0.5)',
        cursor: 'pointer',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        backdropFilter: 'blur(12px)',
        animation: 'pulseGlow 2.5s ease-in-out infinite',
        transition: '150ms ease',
      }}
      onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(124,58,237,0.9)'; e.currentTarget.style.background = 'rgba(12,12,28,0.98)'; }}
      onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(124,58,237,0.5)'; e.currentTarget.style.background = 'rgba(5,5,7,0.96)'; }}
    >
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
        <circle cx="11" cy="11" r="7" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
      </svg>
    </button>
  );
}

// ── Main DJChatbox ─────────────────────────────────────────────────────────
const DJChatbox = forwardRef(function DJChatbox({ selectedTrack, trackCount, onTrackSelect }, ref) {
  const [isOpen,      setIsOpen]      = useState(false);
  const [isMinimised, setIsMinimised] = useState(false);
  const [query,       setQuery]       = useState('');
  const [status,      setStatus]      = useState('idle');
  const [result,      setResult]      = useState(null);
  const [errorMsg,    setErrorMsg]    = useState('');
  const [history,     setHistory]     = useState([]);
  const inputRef   = useRef(null);
  const resultsRef = useRef(null);

  useEffect(() => {
    if (isOpen && !isMinimised) setTimeout(() => inputRef.current?.focus(), 120);
  }, [isOpen, isMinimised]);

  useEffect(() => {
    if (result) resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [result]);

  useEffect(() => {
    if (selectedTrack && isOpen) {
      setQuery(`find something similar to ${selectedTrack.name}`);
      inputRef.current?.focus();
    }
  }, [selectedTrack]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleCandidateSelect = useCallback(async (trackid, title) => {
    setStatus('searching'); setResult(null); setErrorMsg('');
    try {
      const res = await fetch(`${API_BASE}/tracks/${trackid}/similar?limit=7`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const searchResult = {
        tracks: data.tracks || [], relaxation_step: 0,
        relaxation_label: 'embedding similarity', inferred_count: 0,
        reasoning: `Similar to ${data.source?.title || title} by ${data.source?.artist || 'unknown'}`,
        confidence: 1.0, model_used: 'embedding similarity', intent: 'find_similar_track',
      };
      setResult(searchResult);
      setHistory(prev => [{ query: `similar to ${title}`, result: searchResult }, ...prev.slice(0, 9)]);
      setStatus('done');
    } catch (err) {
      setErrorMsg(err.message || 'Search failed'); setStatus('error');
    }
  }, []);

  const runSearch = useCallback(async (queryText) => {
    const q = (queryText || query).trim();
    if (!q) return;
    setStatus('interpreting'); setResult(null); setErrorMsg('');
    try {
      const interpretPayload = { query: q };
      if (selectedTrack) {
        interpretPayload.current_track = {
          trackid:          selectedTrack.id || selectedTrack.trackid,
          title:            selectedTrack.name || selectedTrack.title,
          artist:           selectedTrack.artist,
          bpm:              selectedTrack.bpm,
          key:              selectedTrack.key,
          energy:           selectedTrack.energy,
          semantic_tags:    selectedTrack.tags || [],
          vibe_descriptors: selectedTrack.vibe || [],
        };
      }
      const { params } = await apiClient.post('/chat/interpret', interpretPayload);
      setStatus('searching');
      const searchResult = await apiClient.post('/chat/search', { params });
      setResult(searchResult);
      setHistory(prev => [{ query: q, result: searchResult }, ...prev.slice(0, 9)]);
      setStatus('done');
    } catch (err) {
      setErrorMsg(err.message || 'Unknown error'); setStatus('error');
    }
  }, [query, selectedTrack]);

  const runFindSimilar = useCallback(async (trackid) => {
    if (!trackid) return;
    setStatus('searching'); setResult(null); setErrorMsg('');
    try {
      const res = await fetch(`${API_BASE}/tracks/${trackid}/similar?limit=7`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      const searchResult = {
        tracks: data.tracks || [], relaxation_step: 0, relaxation_label: '', inferred_count: 0,
        reasoning: `Similar to ${data.source?.title || 'selected track'} by ${data.source?.artist || 'unknown'}`,
        confidence: 1.0, model_used: 'embedding similarity',
      };
      setResult(searchResult);
      setHistory(prev => [{ query: `similar to ${data.source?.title || trackid}`, result: searchResult }, ...prev.slice(0, 9)]);
      setStatus('done');
    } catch (err) {
      setErrorMsg(err.message || 'Similar search failed'); setStatus('error');
    }
  }, []);

  useImperativeHandle(ref, () => ({
    openAndSearch: (queryText) => {
      setIsOpen(true); setIsMinimised(false); setQuery(queryText);
      setTimeout(() => runSearch(queryText), 150);
    },
    openAndFindSimilar: (trackid) => {
      setIsOpen(true); setIsMinimised(false);
      setQuery(`similar to track ${trackid}`);
      setTimeout(() => runFindSimilar(trackid), 150);
    },
  }), [runSearch, runFindSimilar]);

  const handleFindSimilarInChat = useCallback((trackid) => {
    setQuery(`similar to track ${trackid}`);
    runFindSimilar(trackid);
  }, [runFindSimilar]);

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); runSearch(); }
  };

  const isBusy = status === 'interpreting' || status === 'searching';

  if (!isOpen) return <BeaconButton onClick={() => setIsOpen(true)} />;

  return (
    <div
      role="dialog"
      aria-label="DJMate Search Terminal"
      style={{
        position: 'absolute', top: 16, left: 16, bottom: 48, zIndex: 1001,
        width: 380,
        display: 'flex', flexDirection: 'column',
        background: 'rgba(5,5,7,0.95)',
        border: '1px solid rgba(124,58,237,0.25)',
        backdropFilter: 'blur(28px)',
        boxShadow: '0 0 60px rgba(124,58,237,0.08), 0 0 100px rgba(0,0,0,0.7)',
        overflow: 'hidden',
        fontFamily: "'Inter', system-ui, sans-serif",
        animation: 'fadeIn 200ms ease',
      }}
    >
      {/* Top accent line */}
      <div style={{ height: 1, background: 'linear-gradient(90deg, #7c3aed, #00d4ff, rgba(0,212,255,0.1))', flexShrink: 0 }} />

      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div
        onClick={() => setIsMinimised(v => !v)}
        style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '10px 14px', flexShrink: 0,
          background: 'rgba(8,8,24,0.7)',
          borderBottom: isMinimised ? 'none' : '1px solid rgba(124,58,237,0.12)',
          cursor: 'pointer', userSelect: 'none',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {/* Search icon */}
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="rgba(124,58,237,0.6)" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true">
            <circle cx="11" cy="11" r="7" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <span style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600, letterSpacing: '0.12em', fontFamily: "'JetBrains Mono', monospace" }}>
            SEARCH TERMINAL
          </span>
          <span style={{ fontSize: 9, color: '#2d3748', fontFamily: "'JetBrains Mono', monospace" }}>
            V0.64
          </span>
        </div>

        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {/* Status dot */}
          <div style={{
            width: 6, height: 6,
            background: isBusy ? '#f59e0b' : status === 'error' ? '#ef4444' : '#00d4ff',
            boxShadow: isBusy ? '0 0 6px #f59e0b' : status === 'error' ? '0 0 6px #ef4444' : '0 0 6px #00d4ff',
            animation: isBusy ? 'blink 0.7s ease-in-out infinite' : 'none',
            flexShrink: 0,
          }} />
          <button
            onClick={e => { e.stopPropagation(); setIsMinimised(v => !v); }}
            aria-label={isMinimised ? 'Expand' : 'Collapse'}
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#475569', display: 'flex', padding: 2, transition: '150ms ease' }}
            onMouseEnter={e => e.currentTarget.style.color = '#94a3b8'}
            onMouseLeave={e => e.currentTarget.style.color = '#475569'}
          >
            {isMinimised ? <IconDown /> : <IconUp />}
          </button>
          <button
            onClick={e => { e.stopPropagation(); setIsOpen(false); }}
            aria-label="Close search panel"
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#475569', display: 'flex', padding: 2, transition: '150ms ease' }}
            onMouseEnter={e => e.currentTarget.style.color = '#ef4444'}
            onMouseLeave={e => e.currentTarget.style.color = '#475569'}
          >
            <IconClose />
          </button>
        </div>
      </div>

      {!isMinimised && (
        <div style={{
          flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column',
          scrollbarWidth: 'thin', scrollbarColor: 'rgba(124,58,237,0.3) transparent',
        }}>

          {/* ── Input ───────────────────────────────────────────────────── */}
          <div style={{ padding: '12px 14px', borderBottom: '1px solid rgba(124,58,237,0.08)', flexShrink: 0 }}>
            <div style={{
              fontSize: 8, letterSpacing: '0.25em', color: 'rgba(124,58,237,0.4)',
              fontFamily: "'JetBrains Mono', monospace", marginBottom: 8,
            }}>
              INPUT PROMPT OR METADATA SEQUENCE...
            </div>

            <div style={{ display: 'flex', gap: 8 }}>
              <textarea
                ref={inputRef}
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={handleKey}
                disabled={isBusy}
                placeholder='e.g. "dark kicking techno 140"  ·  "peak time banger"  ·  "give me 5 deep house"'
                rows={2}
                style={{
                  flex: 1,
                  background: 'rgba(8,8,20,0.8)',
                  border: '1px solid rgba(124,58,237,0.18)',
                  color: '#e2e8f0',
                  fontSize: 13,
                  padding: '10px 12px',
                  fontFamily: "'JetBrains Mono', monospace",
                  resize: 'none',
                  outline: 'none',
                  lineHeight: 1.5,
                  caretColor: '#7c3aed',
                  opacity: isBusy ? 0.5 : 1,
                  transition: '150ms ease',
                  borderRadius: 0,
                }}
                onFocus={e => { e.target.style.borderColor = 'rgba(124,58,237,0.55)'; e.target.style.boxShadow = '0 0 0 1px rgba(124,58,237,0.12) inset'; }}
                onBlur={e => { e.target.style.borderColor = 'rgba(124,58,237,0.18)'; e.target.style.boxShadow = 'none'; }}
              />
              <button
                onClick={() => runSearch()}
                disabled={isBusy || !query.trim()}
                aria-label="Search"
                style={{
                  padding: '0 14px',
                  background: (isBusy || !query.trim())
                    ? 'rgba(8,8,20,0.5)'
                    : 'linear-gradient(135deg, rgba(124,58,237,0.5), rgba(0,212,255,0.3))',
                  border: `1px solid ${(isBusy || !query.trim()) ? 'rgba(124,58,237,0.1)' : 'rgba(124,58,237,0.45)'}`,
                  color: (isBusy || !query.trim()) ? '#2d3748' : '#e2e8f0',
                  cursor: (isBusy || !query.trim()) ? 'default' : 'pointer',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  transition: '150ms ease', alignSelf: 'stretch', flexShrink: 0,
                  borderRadius: 0,
                }}
              >
                {isBusy
                  ? <div style={{ width: 14, height: 14, border: '1.5px solid rgba(124,58,237,0.2)', borderTop: '1.5px solid #7c3aed', animation: 'spin 0.8s linear infinite' }} />
                  : <IconSend />
                }
              </button>
            </div>

            {/* Filter chips */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginTop: 8 }}>
              {QUICK_PROMPTS.map(p => (
                <button
                  key={p.label}
                  disabled={isBusy}
                  onClick={() => { setQuery(p.label.toLowerCase()); inputRef.current?.focus(); }}
                  style={{
                    background: `${p.color}08`,
                    border: `1px solid ${p.color}35`,
                    color: `${p.color}99`,
                    fontSize: 9, fontWeight: 600,
                    padding: '4px 12px', cursor: 'pointer',
                    fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.08em',
                    transition: '150ms ease', borderRadius: 0,
                  }}
                  onMouseEnter={e => { e.target.style.borderColor = `${p.color}70`; e.target.style.color = p.color; e.target.style.background = `${p.color}14`; }}
                  onMouseLeave={e => { e.target.style.borderColor = `${p.color}35`; e.target.style.color = `${p.color}99`; e.target.style.background = `${p.color}08`; }}
                >{p.label}</button>
              ))}
            </div>
          </div>

          {/* ── Status ──────────────────────────────────────────────────── */}
          {isBusy && (
            <div style={{
              padding: '12px 16px', display: 'flex', alignItems: 'center', gap: 10,
              color: 'rgba(124,58,237,0.6)', fontSize: 10, flexShrink: 0,
              fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.08em',
            }}>
              <div style={{
                width: 12, height: 12,
                border: '1.5px solid rgba(124,58,237,0.15)', borderTop: '1.5px solid #7c3aed',
                animation: 'spin 0.8s linear infinite',
              }} />
              {status === 'interpreting' ? 'PARSING QUERY...' : 'SEARCHING LIBRARY...'}
            </div>
          )}

          {/* ── Error ───────────────────────────────────────────────────── */}
          {status === 'error' && (
            <div style={{
              margin: '10px 16px',
              background: 'rgba(16,5,5,0.8)', border: '1px solid rgba(239,68,68,0.25)',
              padding: '10px 12px', fontSize: 11, color: 'rgba(239,68,68,0.8)',
              flexShrink: 0,
            }}>
              {errorMsg}
              <div style={{ fontSize: 9, color: '#475569', marginTop: 4, fontFamily: 'monospace' }}>
                Ensure backend is running at localhost:8000 with /chat/* routes mounted.
              </div>
            </div>
          )}

          {/* ── Results ─────────────────────────────────────────────────── */}
          <div ref={resultsRef} style={{ padding: '0 12px 12px' }}>
            <SearchResults
              result={result}
              source={selectedTrack}
              onTrackClick={onTrackSelect}
              onFindSimilar={handleFindSimilarInChat}
              onCandidateSelect={handleCandidateSelect}
            />
          </div>

          {/* ── History ─────────────────────────────────────────────────── */}
          {history.length > 1 && !isBusy && (
            <div style={{
              borderTop: '1px solid rgba(124,58,237,0.08)',
              padding: '8px 12px', flexShrink: 0,
            }}>
              <div style={{
                fontSize: 8, color: 'rgba(124,58,237,0.35)', marginBottom: 6,
                letterSpacing: '0.25em', fontFamily: 'monospace',
              }}>
                RECENT
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {history.slice(1).map((h, i) => (
                  <button
                    key={i}
                    onClick={() => { setQuery(h.query); setResult(h.result); setStatus('done'); }}
                    style={{
                      background: 'rgba(8,8,20,0.6)', border: '1px solid rgba(124,58,237,0.1)',
                      color: '#2d3748', fontSize: 9,
                      padding: '3px 10px', cursor: 'pointer',
                      fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.03em',
                      transition: '150ms ease', borderRadius: 0,
                    }}
                    onMouseEnter={e => { e.target.style.color = '#7c3aed'; e.target.style.borderColor = 'rgba(124,58,237,0.3)'; }}
                    onMouseLeave={e => { e.target.style.color = '#2d3748'; e.target.style.borderColor = 'rgba(124,58,237,0.1)'; }}
                  >{h.query}</button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
});

export default DJChatbox;
