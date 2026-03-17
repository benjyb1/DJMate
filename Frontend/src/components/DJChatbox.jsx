// src/components/DJChatbox.jsx
import React, { useState, useRef, useEffect, useCallback, forwardRef, useImperativeHandle } from 'react';
import { m, AnimatePresence } from 'framer-motion';
import { apiClient } from '../api/apiClient';
import GlassPanel from './ui/GlassPanel';
import TagEditor from './TagEditor';
import { makeSupabaseCoverUrl } from '../utils/coverUrl';
import { IconPlay, IconPause, IconSearch, IconClose, IconSend, IconUp, IconDown, IconEdit } from './icons';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Aliases for DJChatbox (was IconPlayFill/IconPauseFill with size=10)
const IconPlayFill = () => <IconPlay size={10} />;
const IconPauseFill = () => <IconPause size={10} />;

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
      <div style={{ height: 2, background: 'rgba(124,58,237,0.1)', borderRadius: 'var(--radius-pill)', overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: `linear-gradient(90deg, #7c3aed, ${color})`, borderRadius: 'var(--radius-pill)', transition: 'width 400ms ease' }} />
      </div>
      <div style={{ fontSize: 9, color: '#475569', marginTop: 3, fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.05em' }}>
        {pct}% MATCH
      </div>
    </div>
  );
}

// ── Album art: Supabase storage → iTunes → generative fallback ─────────────
function AlbumArt({ title, artist, directUrl, size = 48 }) {
  const [url, setUrl] = useState(directUrl || null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    setErr(false);
    if (directUrl) { setUrl(directUrl); return; }
    if (!title || !artist) return;
    const supabaseUrl = makeSupabaseCoverUrl(artist, title);
    if (supabaseUrl) { setUrl(supabaseUrl); return; }
  }, [directUrl, title, artist]);

  const handleImgError = () => {
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

  if (url && !err) {
    return (
      <img
        src={url}
        alt={`${title} artwork`}
        width={size}
        height={size}
        onError={handleImgError}
        style={{ width: size, height: size, objectFit: 'cover', display: 'block', flexShrink: 0, borderRadius: 'var(--radius-sm)' }}
      />
    );
  }

  // Generative fallback
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
      borderRadius: 'var(--radius-sm)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: "'JetBrains Mono', monospace", fontSize: size * 0.28, fontWeight: 600,
      color: `hsla(${hue},80%,70%,0.9)`, letterSpacing: '0.02em',
    }}>
      {initials}
    </div>
  );
}

// ── Track card with framer-motion ──────────────────────────────────────────
const TrackCard = React.memo(function TrackCard({ rank, track, score, source, onClick, onFindSimilar, onEdit, isPlaying, onPlay, index = 0 }) {
  const direction = source ? describeDirection(source, track) : null;

  const meta = [];
  if (track.bpm)          meta.push(`${Math.round(track.bpm)} BPM`);
  if (track.key)          meta.push(track.key);
  if (track.energy != null) meta.push(`${parseFloat(track.energy).toFixed(2)} NRG`);

  return (
    <m.div
      role="button"
      tabIndex={0}
      onClick={() => onClick?.(track.trackid)}
      onKeyDown={e => e.key === 'Enter' && onClick?.(track.trackid)}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: 'spring', damping: 26, stiffness: 300, delay: index * 0.04 }}
      whileHover={{ scale: 1.01, backgroundColor: 'rgba(16,16,36,0.7)' }}
      style={{
        display: 'flex', gap: 10, alignItems: 'flex-start',
        background: 'var(--bg-card)',
        border: '1px solid var(--glass-border)',
        borderRadius: 'var(--radius-md)',
        padding: '10px 12px',
        marginBottom: 4,
        cursor: onClick ? 'pointer' : 'default',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Rank */}
      <div style={{
        fontSize: 22, color: 'rgba(124,58,237,0.2)', fontWeight: 900,
        lineHeight: 1, minWidth: 22, fontFamily: "'JetBrains Mono', monospace",
        flexShrink: 0, paddingTop: 1,
      }}>
        {String(rank).padStart(2, '0')}
      </div>

      {/* Album art with play overlay */}
      <div
        style={{ position: 'relative', flexShrink: 0, cursor: 'pointer', width: 46, height: 46 }}
        onClick={e => { e.stopPropagation(); onPlay?.(track); }}
      >
        <AlbumArt title={track.title} artist={track.artist} directUrl={track.album_art_url || null} size={46} />
        <m.div
          style={{
            position: 'absolute', inset: 0,
            borderRadius: 'var(--radius-sm)',
            background: isPlaying ? 'rgba(5,5,7,0.52)' : 'rgba(5,5,7,0)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}
          whileHover={{ background: 'rgba(5,5,7,0.52)' }}
        >
          <div style={{ color: isPlaying ? '#a855f7' : '#e2e8f0', opacity: isPlaying ? 1 : 0, transition: '120ms ease' }}>
            {isPlaying ? <IconPauseFill /> : <IconPlayFill />}
          </div>
        </m.div>
      </div>

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

        {/* Genre tags (purple pills) */}
        {track.semantic_tags?.length > 0 && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginBottom: 4 }}>
            {track.semantic_tags.slice(0, 4).map(tag => (
              <span key={tag} style={{
                background: 'rgba(124,58,237,0.1)', color: 'rgba(168,85,247,0.8)',
                border: '1px solid rgba(124,58,237,0.22)',
                borderRadius: 'var(--radius-pill)',
                padding: '1px 8px', fontSize: 9, letterSpacing: '0.04em',
                fontFamily: "'JetBrains Mono', monospace",
              }}>{tag}</span>
            ))}
          </div>
        )}

        {/* Vibe tags (cyan pills) */}
        {track.vibe_descriptors?.length > 0 && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginBottom: 4 }}>
            {track.vibe_descriptors.slice(0, 4).map(v => (
              <span key={v} style={{
                background: 'rgba(0,212,255,0.07)', color: 'rgba(0,212,255,0.7)',
                border: '1px solid rgba(0,212,255,0.18)',
                borderRadius: 'var(--radius-pill)',
                padding: '1px 8px', fontSize: 9, letterSpacing: '0.04em',
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
              border: '1px solid rgba(245,158,11,0.2)', borderRadius: 'var(--radius-pill)',
              padding: '1px 8px', fontSize: 9, fontFamily: 'monospace',
            }}>INFERRED</span>
          )}
          {direction && (
            <span style={{
              background: `${direction.color}12`, color: direction.color,
              border: `1px solid ${direction.color}38`,
              borderRadius: 'var(--radius-pill)',
              padding: '1px 8px', fontSize: 9, fontWeight: 600,
              fontFamily: 'monospace', letterSpacing: '0.04em',
            }}>{direction.label}</span>
          )}

          {/* Find similar */}
          {onFindSimilar && (
            <m.button
              onClick={e => { e.stopPropagation(); onFindSimilar(track.trackid); }}
              aria-label={`Find similar to ${track.title}`}
              whileHover={{ scale: 1.05, borderColor: 'rgba(0,212,255,0.5)', color: '#00d4ff' }}
              whileTap={{ scale: 0.95 }}
              style={{
                display: 'flex', alignItems: 'center', gap: 4,
                padding: '1px 8px',
                background: 'transparent',
                border: '1px solid rgba(0,212,255,0.22)',
                borderRadius: 'var(--radius-pill)',
                color: 'rgba(0,212,255,0.6)',
                cursor: 'pointer', fontSize: 9, fontFamily: 'inherit',
                letterSpacing: '0.04em',
              }}
            >
              <IconSearch /> SIMILAR
            </m.button>
          )}

          {/* Edit tags */}
          {onEdit && (
            <m.button
              onClick={e => { e.stopPropagation(); onEdit(track); }}
              aria-label={`Edit tags for ${track.title}`}
              whileHover={{ scale: 1.05, borderColor: 'rgba(124,58,237,0.5)', color: '#a855f7' }}
              whileTap={{ scale: 0.95 }}
              style={{
                display: 'flex', alignItems: 'center', gap: 3,
                padding: '1px 8px',
                background: 'rgba(124,58,237,0.08)',
                border: '1px solid rgba(124,58,237,0.22)',
                borderRadius: 'var(--radius-pill)',
                color: 'rgba(168,85,247,0.6)',
                cursor: 'pointer', fontSize: 9, fontFamily: 'inherit',
                letterSpacing: '0.04em',
              }}
            >
              <IconEdit /> EDIT
            </m.button>
          )}
        </div>

        <SimBar score={score} />
      </div>
    </m.div>
  );
});

// ── Alternate candidates bar ───────────────────────────────────────────────
function CandidateBar({ candidates, onSelect }) {
  if (!candidates || candidates.length <= 1) return null;
  return (
    <div style={{
      background: 'var(--bg-card)', border: '1px solid var(--glass-border)',
      borderRadius: 'var(--radius-md)',
      padding: '8px 12px', marginBottom: 8,
    }}>
      <div style={{ fontSize: 8, letterSpacing: '0.25em', color: 'rgba(124,58,237,0.6)', marginBottom: 6, fontFamily: 'monospace' }}>
        DID YOU MEAN
      </div>
      {candidates.slice(1).map((c, i) => (
        <m.div
          key={c.trackid}
          role="button"
          tabIndex={0}
          onClick={() => onSelect(c.trackid, c.title)}
          onKeyDown={e => e.key === 'Enter' && onSelect(c.trackid, c.title)}
          whileHover={{ opacity: 0.65 }}
          style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '5px 0', cursor: 'pointer',
            borderBottom: i < candidates.length - 2 ? '1px solid rgba(124,58,237,0.08)' : 'none',
          }}
        >
          <div>
            <span style={{ fontSize: 12, color: '#94a3b8' }}>{c.title}</span>
            <span style={{ fontSize: 10, color: '#475569', marginLeft: 8 }}>{c.artist}</span>
          </div>
          <span style={{ fontSize: 9, color: 'rgba(124,58,237,0.6)', fontFamily: 'monospace' }}>
            {Math.round(c.match_score * 100)}%
          </span>
        </m.div>
      ))}
    </div>
  );
}

// ── Search results ─────────────────────────────────────────────────────────
function SearchResults({ result, source, onTrackClick, onFindSimilar, onCandidateSelect, playingId, onPlay }) {
  if (!result) return null;
  const isSimilar = result.intent === 'find_similar_track';

  return (
    <m.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: 'spring', damping: 26, stiffness: 300 }}
      style={{ marginTop: 10 }}
    >
      {/* Track not found */}
      {isSimilar && result.tracks.length === 0 && (
        <div style={{
          background: 'rgba(16,5,5,0.8)', border: '1px solid rgba(239,68,68,0.25)',
          borderRadius: 'var(--radius-md)',
          padding: '10px 12px', marginBottom: 8, fontSize: 11, color: 'rgba(239,68,68,0.8)',
        }}>
          {result.reasoning || "Track not found in library."}
          <div style={{ fontSize: 9, color: '#475569', marginTop: 4, fontFamily: 'monospace' }}>
            Try the full track name, or use vibe / genre search.
          </div>
        </div>
      )}

      {isSimilar && result.track_candidates && (
        <CandidateBar candidates={result.track_candidates} onSelect={onCandidateSelect} />
      )}

      {/* Widened search notice */}
      {!isSimilar && result.relaxation_step > 0 && (
        <div style={{
          background: 'rgba(10,8,0,0.7)', border: '1px solid rgba(245,158,11,0.25)',
          borderRadius: 'var(--radius-sm)',
          padding: '6px 12px', marginBottom: 8, fontSize: 10,
          color: 'rgba(245,158,11,0.7)', fontFamily: 'monospace', letterSpacing: '0.04em',
        }}>
          WIDENED SEARCH — {result.relaxation_label}
        </div>
      )}

      {result.inferred_count > 0 && (
        <div style={{
          background: 'rgba(10,8,0,0.7)', border: '1px solid rgba(245,158,11,0.2)',
          borderRadius: 'var(--radius-sm)',
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
              background: 'var(--bg-card)', border: '1px solid rgba(124,58,237,0.12)',
              borderRadius: 'var(--radius-pill)',
              padding: '1px 8px', fontSize: 8, color: '#2d3748', letterSpacing: '0.04em',
            }}>{result.model_used}</span>
          )}
        </div>
      )}

      {/* Track cards */}
      {result.tracks.map((track, i) => (
        <React.Fragment key={track.trackid || i}>
          {isSimilar && i === 0 && (
            <div style={{ fontSize: 8, color: '#00d4ff', letterSpacing: '0.25em', marginBottom: 5, fontFamily: 'monospace', opacity: 0.7 }}>
              MATCHED TRACK
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
            onEdit={setEditingTrack}
            isPlaying={playingId === String(track.trackid)}
            onPlay={onPlay}
            index={i}
          />
        </React.Fragment>
      ))}
    </m.div>
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
    <m.button
      onClick={onClick}
      aria-label="Open DJMate Search"
      title="Open Search Terminal"
      initial={{ scale: 0, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      exit={{ scale: 0, opacity: 0 }}
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.9 }}
      transition={{ type: 'spring', damping: 20, stiffness: 400 }}
      style={{
        position: 'fixed', bottom: 24, left: 20, zIndex: 1000,
        width: 52, height: 52,
        borderRadius: '50%',
        background: 'var(--glass-bg)',
        border: '1px solid rgba(124,58,237,0.5)',
        cursor: 'pointer',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        backdropFilter: 'blur(20px)',
        WebkitBackdropFilter: 'blur(20px)',
        boxShadow: 'var(--shadow-float)',
        animation: 'pulseGlow 2.5s ease-in-out infinite',
      }}
    >
      {/* Pulse rings */}
      <div style={{
        position: 'absolute', inset: -6, borderRadius: '50%',
        border: '1px solid rgba(124,58,237,0.15)',
        animation: 'glowPulse 3s ease-in-out infinite',
      }} />
      <div style={{
        position: 'absolute', inset: -14, borderRadius: '50%',
        border: '1px solid rgba(124,58,237,0.08)',
        animation: 'glowPulse 3s ease-in-out infinite 0.5s',
      }} />
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
        <circle cx="11" cy="11" r="7" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
      </svg>
    </m.button>
  );
}

// ── Panel animation variants ──────────────────────────────────────────────
const panelVariants = {
  hidden: { opacity: 0, x: -24, scale: 0.96 },
  visible: { opacity: 1, x: 0, scale: 1 },
  exit:   { opacity: 0, x: -24, scale: 0.96 },
};

const panelTransition = { type: 'spring', damping: 28, stiffness: 320 };

// ── Main DJChatbox ─────────────────────────────────────────────────────────
const DJChatbox = forwardRef(function DJChatbox({ selectedTrack, trackCount, onTrackSelect }, ref) {
  const [isOpen,      setIsOpen]      = useState(false);
  const [isMinimised, setIsMinimised] = useState(false);
  const [query,       setQuery]       = useState('');
  const [status,      setStatus]      = useState('idle');
  const [result,      setResult]      = useState(null);
  const [errorMsg,    setErrorMsg]    = useState('');
  const [history,     setHistory]     = useState([]);
  const [playingId,   setPlayingId]   = useState(null);
  const [editingTrack, setEditingTrack] = useState(null);
  const [availableTags, setAvailableTags] = useState([]);
  const [availableVibes, setAvailableVibes] = useState([]);
  const inputRef   = useRef(null);
  const resultsRef = useRef(null);
  const chatAudioRef = useRef(new Audio());

  // Audio lifecycle
  useEffect(() => {
    const a = chatAudioRef.current;
    a.onended = () => setPlayingId(null);
    return () => { a.pause(); a.src = ''; };
  }, []);

  // Fetch available tags for autocomplete
  useEffect(() => {
    apiClient.get('/tags/available').then(data => {
      setAvailableTags(data.semantic_tags || []);
      setAvailableVibes(data.vibes || []);
    }).catch(() => {});
  }, []);

  const handlePlay = useCallback((track) => {
    const id  = String(track.trackid);
    const src = track.audio_url || `${API_BASE}/tracks/${track.trackid}/audio`;
    const a   = chatAudioRef.current;
    if (playingId === id) {
      a.pause();
      setPlayingId(null);
    } else {
      a.pause();
      a.src = src;
      a.play().catch(() => {});
      setPlayingId(id);
    }
  }, [playingId]);

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

  return (
    <>
      <AnimatePresence>
        {!isOpen && <BeaconButton onClick={() => setIsOpen(true)} />}
      </AnimatePresence>

      <AnimatePresence>
        {isOpen && (
          <m.div
            role="dialog"
            aria-label="DJMate Search Terminal"
            variants={panelVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
            transition={panelTransition}
            style={{
              position: 'absolute', bottom: 80, left: 20, zIndex: 1001,
              width: 380,
              maxHeight: 'calc(100vh - 160px)',
              display: 'flex', flexDirection: 'column',
              background: 'var(--glass-bg)',
              border: '1px solid var(--glass-border)',
              borderRadius: 'var(--radius-xl)',
              backdropFilter: 'blur(32px)',
              WebkitBackdropFilter: 'blur(32px)',
              boxShadow: 'var(--shadow-float)',
              overflow: 'hidden',
              fontFamily: "'Inter', system-ui, sans-serif",
            }}
          >
            {/* Top accent line */}
            <div style={{
              height: 2, flexShrink: 0,
              background: 'linear-gradient(90deg, #7c3aed, #00d4ff, rgba(0,212,255,0.1))',
              borderRadius: 'var(--radius-xl) var(--radius-xl) 0 0',
            }} />

            {/* ── Header ────────────────────────────────────────────────── */}
            <div
              onClick={() => setIsMinimised(v => !v)}
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: '10px 16px', flexShrink: 0,
                background: 'rgba(8,8,24,0.5)',
                borderBottom: isMinimised ? 'none' : '1px solid rgba(124,58,237,0.1)',
                cursor: 'pointer', userSelect: 'none',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
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
                  width: 6, height: 6, borderRadius: '50%',
                  background: isBusy ? '#f59e0b' : status === 'error' ? '#ef4444' : '#00d4ff',
                  boxShadow: isBusy ? '0 0 6px #f59e0b' : status === 'error' ? '0 0 6px #ef4444' : '0 0 6px #00d4ff',
                  animation: isBusy ? 'blink 0.7s ease-in-out infinite' : 'none',
                  flexShrink: 0,
                }} />
                <m.button
                  onClick={e => { e.stopPropagation(); setIsMinimised(v => !v); }}
                  aria-label={isMinimised ? 'Expand' : 'Collapse'}
                  whileHover={{ color: '#94a3b8' }}
                  whileTap={{ scale: 0.9 }}
                  style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#475569', display: 'flex', padding: 2 }}
                >
                  {isMinimised ? <IconDown /> : <IconUp />}
                </m.button>
                <m.button
                  onClick={e => { e.stopPropagation(); setIsOpen(false); }}
                  aria-label="Close search panel"
                  whileHover={{ color: '#ef4444' }}
                  whileTap={{ scale: 0.9 }}
                  style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#475569', display: 'flex', padding: 2 }}
                >
                  <IconClose />
                </m.button>
              </div>
            </div>

            <AnimatePresence initial={false}>
              {!isMinimised && (
                <m.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ type: 'spring', damping: 28, stiffness: 300 }}
                  style={{
                    flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column',
                    scrollbarWidth: 'thin', scrollbarColor: 'rgba(124,58,237,0.3) transparent',
                  }}
                >

                  {/* ── Input ─────────────────────────────────────────────── */}
                  <div style={{ padding: '12px 16px', borderBottom: '1px solid rgba(124,58,237,0.08)', flexShrink: 0 }}>
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
                          background: 'rgba(8,8,20,0.6)',
                          border: '1px solid rgba(124,58,237,0.18)',
                          borderRadius: 'var(--radius-md)',
                          color: '#e2e8f0',
                          fontSize: 13,
                          padding: '10px 12px',
                          fontFamily: "'JetBrains Mono', monospace",
                          resize: 'none',
                          outline: 'none',
                          lineHeight: 1.5,
                          caretColor: '#7c3aed',
                          opacity: isBusy ? 0.5 : 1,
                          transition: '200ms ease',
                        }}
                        onFocus={e => { e.target.style.borderColor = 'rgba(124,58,237,0.55)'; e.target.style.boxShadow = '0 0 0 3px rgba(124,58,237,0.08)'; }}
                        onBlur={e => { e.target.style.borderColor = 'rgba(124,58,237,0.18)'; e.target.style.boxShadow = 'none'; }}
                      />
                      <m.button
                        onClick={() => runSearch()}
                        disabled={isBusy || !query.trim()}
                        aria-label="Search"
                        whileHover={(isBusy || !query.trim()) ? {} : { scale: 1.05 }}
                        whileTap={(isBusy || !query.trim()) ? {} : { scale: 0.95 }}
                        style={{
                          padding: '0 14px',
                          background: (isBusy || !query.trim())
                            ? 'rgba(8,8,20,0.5)'
                            : 'linear-gradient(135deg, rgba(124,58,237,0.5), rgba(0,212,255,0.3))',
                          border: `1px solid ${(isBusy || !query.trim()) ? 'rgba(124,58,237,0.1)' : 'rgba(124,58,237,0.45)'}`,
                          borderRadius: 'var(--radius-sm)',
                          color: (isBusy || !query.trim()) ? '#2d3748' : '#e2e8f0',
                          cursor: (isBusy || !query.trim()) ? 'default' : 'pointer',
                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                          alignSelf: 'stretch', flexShrink: 0,
                        }}
                      >
                        {isBusy
                          ? <div style={{ width: 14, height: 14, border: '1.5px solid rgba(124,58,237,0.2)', borderTop: '1.5px solid #7c3aed', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
                          : <IconSend />
                        }
                      </m.button>
                    </div>

                    {/* Quick prompt chips */}
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginTop: 8 }}>
                      {QUICK_PROMPTS.map(p => (
                        <m.button
                          key={p.label}
                          disabled={isBusy}
                          onClick={() => { setQuery(p.label.toLowerCase()); inputRef.current?.focus(); }}
                          whileHover={{ scale: 1.05, borderColor: `${p.color}70` }}
                          whileTap={{ scale: 0.95 }}
                          style={{
                            background: `${p.color}08`,
                            border: `1px solid ${p.color}35`,
                            borderRadius: 'var(--radius-pill)',
                            color: `${p.color}99`,
                            fontSize: 9, fontWeight: 600,
                            padding: '4px 12px', cursor: 'pointer',
                            fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.08em',
                          }}
                        >{p.label}</m.button>
                      ))}
                    </div>
                  </div>

                  {/* ── Status ────────────────────────────────────────────── */}
                  <AnimatePresence>
                    {isBusy && (
                      <m.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        style={{
                          padding: '12px 16px', display: 'flex', alignItems: 'center', gap: 10,
                          color: 'rgba(124,58,237,0.6)', fontSize: 10, flexShrink: 0,
                          fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.08em',
                        }}
                      >
                        <div style={{
                          width: 12, height: 12, borderRadius: '50%',
                          border: '1.5px solid rgba(124,58,237,0.15)', borderTop: '1.5px solid #7c3aed',
                          animation: 'spin 0.8s linear infinite',
                        }} />
                        {status === 'interpreting' ? 'PARSING QUERY...' : 'SEARCHING LIBRARY...'}
                      </m.div>
                    )}
                  </AnimatePresence>

                  {/* ── Error ─────────────────────────────────────────────── */}
                  <AnimatePresence>
                    {status === 'error' && (
                      <m.div
                        initial={{ opacity: 0, y: -4 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        style={{
                          margin: '10px 16px',
                          background: 'rgba(16,5,5,0.8)', border: '1px solid rgba(239,68,68,0.25)',
                          borderRadius: 'var(--radius-md)',
                          padding: '10px 12px', fontSize: 11, color: 'rgba(239,68,68,0.8)',
                          flexShrink: 0,
                        }}
                      >
                        {errorMsg}
                        <div style={{ fontSize: 9, color: '#475569', marginTop: 4, fontFamily: 'monospace' }}>
                          Ensure backend is running at localhost:8000 with /chat/* routes mounted.
                        </div>
                      </m.div>
                    )}
                  </AnimatePresence>

                  {/* ── Results ───────────────────────────────────────────── */}
                  <div ref={resultsRef} style={{ padding: '0 12px 12px' }}>
                    <SearchResults
                      result={result}
                      source={selectedTrack}
                      onTrackClick={onTrackSelect}
                      onFindSimilar={handleFindSimilarInChat}
                      onCandidateSelect={handleCandidateSelect}
                      playingId={playingId}
                      onPlay={handlePlay}
                    />
                  </div>

                  {/* ── History ───────────────────────────────────────────── */}
                  <AnimatePresence>
                    {history.length > 1 && !isBusy && (
                      <m.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        style={{
                          borderTop: '1px solid rgba(124,58,237,0.08)',
                          padding: '8px 12px', flexShrink: 0,
                        }}
                      >
                        <div style={{
                          fontSize: 8, color: 'rgba(124,58,237,0.35)', marginBottom: 6,
                          letterSpacing: '0.25em', fontFamily: 'monospace',
                        }}>
                          RECENT
                        </div>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                          {history.slice(1).map((h, i) => (
                            <m.button
                              key={i}
                              onClick={() => { setQuery(h.query); setResult(h.result); setStatus('done'); }}
                              whileHover={{ scale: 1.05, color: '#7c3aed', borderColor: 'rgba(124,58,237,0.3)' }}
                              whileTap={{ scale: 0.95 }}
                              style={{
                                background: 'var(--bg-card)', border: '1px solid rgba(124,58,237,0.1)',
                                borderRadius: 'var(--radius-pill)',
                                color: '#2d3748', fontSize: 9,
                                padding: '3px 10px', cursor: 'pointer',
                                fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.03em',
                              }}
                            >{h.query}</m.button>
                          ))}
                        </div>
                      </m.div>
                    )}
                  </AnimatePresence>
                </m.div>
              )}
            </AnimatePresence>
          </m.div>
        )}
      </AnimatePresence>
      {/* Tag editor modal */}
      <AnimatePresence>
        {editingTrack && (
          <TagEditor
            track={editingTrack}
            onClose={() => setEditingTrack(null)}
            onSave={(updated) => {
              // Update the track in current results with new tags
              if (result?.tracks) {
                setResult(prev => ({
                  ...prev,
                  tracks: prev.tracks.map(t =>
                    t.trackid === editingTrack.trackid
                      ? { ...t, semantic_tags: updated.semantic_tags, vibe_descriptors: updated.vibe_descriptors, energy: updated.energy }
                      : t
                  ),
                }));
              }
            }}
            availableTags={availableTags}
            availableVibes={availableVibes}
          />
        )}
      </AnimatePresence>
    </>
  );
});

export default DJChatbox;
