// src/components/LiveMode.jsx
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { apiClient } from '../api/apiClient';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const SUPABASE_COVERS_BASE = 'https://cvermotfxamubejfnoje.supabase.co/storage/v1/object/public/album-covers/';

// ── Shared helpers ─────────────────────────────────────────────────────────
function makeSupabaseCoverUrl(artist, title) {
  if (!artist || !title) return null;
  let safe = `${artist}_${title}`.toLowerCase();
  safe = safe.split('').map(c => /[a-z0-9\-_]/.test(c) ? c : '_').join('');
  safe = safe.split('_').filter(Boolean).join('_').slice(0, 150);
  return `${SUPABASE_COVERS_BASE}${safe}.jpg`;
}

function getAudioSrc(track) {
  // track may be from allNodes (has audioUrl) or from API results (has audio_url)
  return track.audioUrl || track.audio_url
    || `${API_BASE}/tracks/${track.id || track.trackid}/audio`;
}

// ── Trajectory calculation ─────────────────────────────────────────────────
function getTrajectoryVector(setList) {
  if (setList.length < 2) return null;
  const recent = setList.slice(-Math.min(4, setList.length));
  const vectors = [];
  for (let i = 1; i < recent.length; i++) {
    vectors.push({
      x: (recent[i].x || 0) - (recent[i - 1].x || 0),
      y: (recent[i].y || 0) - (recent[i - 1].y || 0),
      z: (recent[i].z || 0) - (recent[i - 1].z || 0),
    });
  }
  const avg = vectors.reduce((acc, v) => ({
    x: acc.x + v.x / vectors.length,
    y: acc.y + v.y / vectors.length,
    z: acc.z + v.z / vectors.length,
  }), { x: 0, y: 0, z: 0 });
  const mag = Math.sqrt(avg.x ** 2 + avg.y ** 2 + avg.z ** 2);
  if (mag < 0.001) return null;
  return { x: avg.x / mag, y: avg.y / mag, z: avg.z / mag };
}

// Returns { starters, onTrajectory, energyUp, stepDown } based on anchor track
function suggestDirectional(anchor, setList, allNodes) {
  const played = new Set(setList.map(t => String(t.id)));
  const candidates = allNodes.filter(n => !played.has(String(n.id)) && String(n.id) !== String(anchor?.id));

  // No anchor yet — random starters
  if (!anchor) {
    return {
      starters:      candidates.sort(() => Math.random() - 0.5).slice(0, 8),
      onTrajectory:  [],
      energyUp:      [],
      stepDown:      [],
    };
  }

  const dir = setList.length >= 2 ? getTrajectoryVector(setList) : null;

  const scored = candidates.map(n => {
    const dx = (n.x || 0) - (anchor.x || 0);
    const dy = (n.y || 0) - (anchor.y || 0);
    const dz = (n.z || 0) - (anchor.z || 0);
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
    const dot  = dir ? (dx * dir.x + dy * dir.y + dz * dir.z) / dist : 0;
    const dirScore  = dir ? (dot + 1) / 2 : 0.5;
    const proxScore = 1 / (1 + dist * 0.0008);
    const bpmDiff   = Math.abs((n.bpm || 130) - (anchor.bpm || 130));
    const bpmScore  = Math.max(0, 1 - bpmDiff / 15);
    const keyScore  = keyCompatibility(n.key, anchor.key);
    const _score    = dirScore * 0.3 + proxScore * 0.15 + bpmScore * 0.25 + keyScore * 0.3;
    return { ...n, _score, _dist: dist, _dot: dot };
  });

  const anchorEnergy = parseFloat(anchor.energy ?? 0.5);

  const onTrajectory = scored
    .filter(n => n._score > 0.3)
    .sort((a, b) => b._score - a._score)
    .slice(0, 5)
    .map(n => ({ ...n, _reason: getDirectionReason(n, anchor) }));

  const energyUp = candidates
    .filter(n => parseFloat(n.energy ?? 0.5) > anchorEnergy + 0.08)
    .filter(n => keyCompatibility(n.key, anchor.key) >= 0.3)
    .sort((a, b) => {
      const sa = keyCompatibility(a.key, anchor.key) + (1 - Math.abs((a.bpm||130)-(anchor.bpm||130))/15);
      const sb = keyCompatibility(b.key, anchor.key) + (1 - Math.abs((b.bpm||130)-(anchor.bpm||130))/15);
      return sb - sa;
    })
    .slice(0, 3)
    .map(n => ({ ...n, _reason: getDirectionReason(n, anchor) }));

  const stepDown = candidates
    .filter(n => parseFloat(n.energy ?? 0.5) < anchorEnergy - 0.08)
    .filter(n => keyCompatibility(n.key, anchor.key) >= 0.3)
    .sort((a, b) => {
      const sa = keyCompatibility(a.key, anchor.key) + (1 - Math.abs((a.bpm||130)-(anchor.bpm||130))/15);
      const sb = keyCompatibility(b.key, anchor.key) + (1 - Math.abs((b.bpm||130)-(anchor.bpm||130))/15);
      return sb - sa;
    })
    .slice(0, 3)
    .map(n => ({ ...n, _reason: getDirectionReason(n, anchor) }));

  return { starters: [], onTrajectory, energyUp, stepDown };
}

function getDirectionReason(n, anchor) {
  const parts = [];
  // Energy comparison
  const nE = parseFloat(n.energy ?? 0.5);
  const aE = parseFloat(anchor.energy ?? 0.5);
  if (nE > aE + 0.1) parts.push('higher energy');
  else if (nE < aE - 0.1) parts.push('lower energy');
  // Genre from semantic tags
  const tags = Array.isArray(n.semanticTags) ? n.semanticTags : [];
  if (tags.length > 0) parts.push(tags[0]);
  // Key
  if (n.key) parts.push(n.key);
  // BPM
  if (n.bpm) {
    const bpmDiff = (n.bpm || 130) - (anchor.bpm || 130);
    if (Math.abs(bpmDiff) < 3) parts.push(`${Math.round(n.bpm)} BPM`);
    else parts.push(`${bpmDiff > 0 ? '+' : ''}${Math.round(bpmDiff)} BPM`);
  }
  return parts.join(' · ') || 'similar vibe';
}

// ── Camelot key compatibility ──────────────────────────────────────────────
const CAMELOT_MAP = {
  'Abm':'1A','G#m':'1A','Ebm':'2A','D#m':'2A','Bbm':'3A','A#m':'3A',
  'Fm':'4A','Cm':'5A','Gm':'6A','Dm':'7A','Am':'8A','Em':'9A',
  'Bm':'10A','F#m':'11A','Gbm':'11A','C#m':'12A','Dbm':'12A',
  'B':'1B','Gb':'2B','F#':'2B','Db':'3B','C#':'3B',
  'Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B',
  'F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B',
};

function parseCamelot(key) {
  if (!key) return null;
  const k = key.trim();
  const cm = k.match(/^(\d{1,2})([ABab])$/);
  if (cm) return { num: parseInt(cm[1]), letter: cm[2].toUpperCase() };
  const code = CAMELOT_MAP[k];
  if (code) { const m = code.match(/^(\d+)([AB])$/); return { num: parseInt(m[1]), letter: m[2] }; }
  return null;
}

function keyCompatibility(key1, key2) {
  const c1 = parseCamelot(key1), c2 = parseCamelot(key2);
  if (!c1 || !c2) return 0.5;
  if (c1.num === c2.num && c1.letter === c2.letter) return 1.0;
  if (c1.num === c2.num) return 0.8; // relative major/minor
  const diff = Math.min(Math.abs(c1.num - c2.num), 12 - Math.abs(c1.num - c2.num));
  if (c1.letter === c2.letter && diff === 1) return 0.85;
  if (c1.letter === c2.letter && diff === 2) return 0.5;
  if (c1.letter !== c2.letter && diff <= 1) return 0.7;
  return 0.15;
}

// ── Audio fingerprint helpers ─────────────────────────────────────────────
const NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];

function detectBPMFromFlux(flux, frameRate) {
  const minLag = Math.round(frameRate * 60 / 180);
  const maxLag = Math.round(frameRate * 60 / 70);
  let bestLag = minLag, bestCorr = -Infinity;
  const n = flux.length;
  for (let lag = minLag; lag <= maxLag && lag < n / 2; lag++) {
    let corr = 0, count = 0;
    for (let i = 0; i < n - lag; i++) { corr += flux[i] * flux[i + lag]; count++; }
    corr /= count || 1;
    if (corr > bestCorr) { bestCorr = corr; bestLag = lag; }
  }
  return Math.round(60 * frameRate / bestLag);
}

function estimateKeyFromChroma(chroma) {
  const maxC = Math.max(...chroma);
  if (maxC <= 0) return null;
  let maxIdx = 0;
  for (let i = 1; i < 12; i++) { if (chroma[i] > chroma[maxIdx]) maxIdx = i; }
  const maj = chroma[maxIdx] + chroma[(maxIdx + 4) % 12] + chroma[(maxIdx + 7) % 12];
  const min = chroma[maxIdx] + chroma[(maxIdx + 3) % 12] + chroma[(maxIdx + 7) % 12];
  return NOTE_NAMES[maxIdx] + (min > maj ? 'm' : '');
}

// ── SVG Icons ──────────────────────────────────────────────────────────────
const IconMic = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <rect x="9" y="2" width="6" height="11" rx="3" />
    <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
    <line x1="12" y1="19" x2="12" y2="23" />
    <line x1="8" y1="23" x2="16" y2="23" />
  </svg>
);
const IconPlus = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
    <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
  </svg>
);
const IconX = () => (
  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
    <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
  </svg>
);
const IconSend = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13" /><polygon points="22,2 15,22 11,13 2,9" />
  </svg>
);
const IconVector = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <path d="M5 12h14" /><polyline points="12,5 19,12 12,19" />
  </svg>
);
const IconPlay = () => (
  <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor">
    <polygon points="5,3 19,12 5,21" />
  </svg>
);
const IconPause = () => (
  <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor">
    <rect x="6" y="4" width="4" height="16" rx="1" /><rect x="14" y="4" width="4" height="16" rx="1" />
  </svg>
);
const IconSimilar = () => (
  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
    <circle cx="11" cy="11" r="7" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>
);

// ── Shared album art renderer ──────────────────────────────────────────────
function TrackArt({ track, size = 44 }) {
  const initial = track.albumArt || makeSupabaseCoverUrl(track.artist, track.name || track.title);
  const [artUrl, setArtUrl] = useState(initial);
  const [artErr, setArtErr] = useState(false);

  useEffect(() => {
    const url = track.albumArt || makeSupabaseCoverUrl(track.artist, track.name || track.title);
    setArtUrl(url);
    setArtErr(false);
  }, [track.id, track.trackid]);

  const handleErr = () => {
    if (artUrl?.includes('supabase')) {
      const name  = track.name || track.title || '';
      const term  = encodeURIComponent(`${track.artist} ${name}`);
      fetch(`https://itunes.apple.com/search?term=${term}&entity=song&limit=1`)
        .then(r => r.json())
        .then(data => {
          const raw = data.results?.[0]?.artworkUrl100;
          if (raw) setArtUrl(raw.replace('100x100bb', '300x300bb'));
          else setArtErr(true);
        })
        .catch(() => setArtErr(true));
    } else setArtErr(true);
  };

  if (artUrl && !artErr) {
    return (
      <img src={artUrl} alt={track.name || track.title} onError={handleErr}
        style={{ width: size, height: size, objectFit: 'cover', display: 'block', flexShrink: 0 }} />
    );
  }

  let hash = 0;
  const str = ((track.name || track.title || '') + (track.artist || ''));
  for (let i = 0; i < str.length; i++) { hash = ((hash << 5) - hash) + str.charCodeAt(i); hash |= 0; }
  const hue      = 200 + (Math.abs(hash) % 100);
  const initials = (str || '?').replace(/[^a-zA-Z0-9]/g, '').slice(0, 2).toUpperCase() || '??';

  return (
    <div style={{
      width: size, height: size, flexShrink: 0,
      background: `linear-gradient(135deg, hsl(${hue},50%,8%), hsl(${hue},70%,5%))`,
      border: `1px solid hsla(${hue},70%,50%,0.3)`,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: "'JetBrains Mono', monospace",
      fontSize: Math.round(size * 0.28), fontWeight: 700,
      color: `hsla(${hue},80%,70%,0.9)`,
    }}>
      {initials}
    </div>
  );
}

// ── Set list track card (horizontal scroll) ────────────────────────────────
function SetTrackArt({ track, index, isLatest, isPlaying, isAnchor, onRemove, onPlay, onSelectAnchor }) {
  const [hovered, setHovered] = useState(false);

  return (
    <div style={{ position: 'relative', flexShrink: 0 }}>
      {/* Track number */}
      <div style={{
        position: 'absolute', top: -16, left: 0, right: 0, textAlign: 'center',
        fontSize: 8, color: isAnchor ? '#a855f7' : '#2d3748',
        fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.1em',
        transition: '150ms ease',
      }}>
        {String(index + 1).padStart(2, '0')}
      </div>

      {/* Art + play overlay */}
      <div
        style={{
          width: 80, height: 80, position: 'relative', overflow: 'hidden',
          border: isAnchor  ? '2px solid #a855f7'
                : isLatest  ? '2px solid #00d4ff'
                : isPlaying ? '2px solid rgba(168,85,247,0.6)'
                : '1px solid rgba(124,58,237,0.25)',
          boxShadow: isAnchor  ? '0 0 18px rgba(168,85,247,0.35)'
                   : isLatest  ? '0 0 16px rgba(0,212,255,0.3)'
                   : 'none',
          cursor: 'pointer', transition: '200ms ease',
        }}
        onClick={() => onPlay(track)}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        <TrackArt track={track} size={80} />

        {(hovered || isPlaying) && (
          <div style={{
            position: 'absolute', inset: 0, background: 'rgba(5,5,7,0.55)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <div style={{ color: isPlaying ? '#a855f7' : '#e2e8f0' }}>
              {isPlaying ? <IconPause /> : <IconPlay />}
            </div>
          </div>
        )}

        {(isLatest || isPlaying || isAnchor) && (
          <div style={{
            position: 'absolute', bottom: 0, left: 0, right: 0,
            background: isAnchor  ? 'rgba(168,85,247,0.2)'
                      : isPlaying ? 'rgba(168,85,247,0.18)'
                      : 'rgba(0,212,255,0.15)',
            borderTop: `1px solid ${isAnchor ? 'rgba(168,85,247,0.6)' : isPlaying ? 'rgba(168,85,247,0.5)' : 'rgba(0,212,255,0.4)'}`,
            fontSize: 7,
            color: isAnchor ? '#a855f7' : isPlaying ? '#a855f7' : '#00d4ff',
            textAlign: 'center', padding: '2px 0',
            letterSpacing: '0.12em', fontFamily: "'JetBrains Mono', monospace",
          }}>
            {isAnchor ? 'EXPLORE' : isPlaying ? '▶ PLAYING' : 'LATEST'}
          </div>
        )}
      </div>

      {/* Remove */}
      <button
        onClick={() => onRemove(index)}
        aria-label="Remove from set"
        style={{
          position: 'absolute', top: -6, right: -6, width: 18, height: 18,
          background: 'rgba(5,5,7,0.9)', border: '1px solid rgba(239,68,68,0.4)',
          color: 'rgba(239,68,68,0.6)', cursor: 'pointer',
          display: 'flex', alignItems: 'center', justifyContent: 'center', transition: '150ms ease',
        }}
        onMouseEnter={e => { e.currentTarget.style.background = 'rgba(239,68,68,0.15)'; e.currentTarget.style.color = '#ef4444'; }}
        onMouseLeave={e => { e.currentTarget.style.background = 'rgba(5,5,7,0.9)'; e.currentTarget.style.color = 'rgba(239,68,68,0.6)'; }}
      >
        <IconX />
      </button>

      {/* Track info — click to explore from here */}
      <div
        onClick={() => onSelectAnchor(track)}
        title="Explore what comes after this track"
        style={{ marginTop: 6, width: 80, overflow: 'hidden', cursor: 'pointer' }}
      >
        <div style={{ fontSize: 9, color: isAnchor ? '#c084fc' : '#94a3b8', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', fontWeight: 600, transition: '150ms ease' }}>
          {track.name}
        </div>
        <div style={{ fontSize: 8, color: '#475569', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {track.artist}
        </div>
        {track.bpm && (
          <div style={{ fontSize: 8, color: '#2d3748', fontFamily: "'JetBrains Mono', monospace" }}>
            {Math.round(track.bpm)} BPM
          </div>
        )}
      </div>
    </div>
  );
}

// ── Suggestion / search result row ─────────────────────────────────────────
function TrackRow({ track, isPlaying, isAnchor, onAdd, onPlay, onFindSimilar, onSelectAnchor, isStarter = false }) {
  const name = track.name || track.title || 'Unknown';
  return (
    <div
      style={{
        display: 'flex', gap: 10, alignItems: 'center',
        padding: '9px 14px',
        background: isAnchor ? 'rgba(124,58,237,0.08)' : 'rgba(8,8,20,0.5)',
        border: `1px solid ${isAnchor ? 'rgba(168,85,247,0.4)' : 'rgba(124,58,237,0.12)'}`,
        marginBottom: 3,
        transition: '150ms ease', cursor: 'default',
      }}
      onMouseEnter={e => { e.currentTarget.style.borderColor = isAnchor ? 'rgba(168,85,247,0.6)' : 'rgba(124,58,237,0.35)'; e.currentTarget.style.background = isAnchor ? 'rgba(124,58,237,0.12)' : 'rgba(12,12,28,0.7)'; }}
      onMouseLeave={e => { e.currentTarget.style.borderColor = isAnchor ? 'rgba(168,85,247,0.4)' : 'rgba(124,58,237,0.12)'; e.currentTarget.style.background = isAnchor ? 'rgba(124,58,237,0.08)' : 'rgba(8,8,20,0.5)'; }}
    >
      {/* Art with play overlay */}
      <div
        style={{ position: 'relative', flexShrink: 0, cursor: 'pointer' }}
        onClick={() => onPlay(track)}
      >
        <TrackArt track={{ ...track, name }} size={44} />
        <div style={{
          position: 'absolute', inset: 0,
          background: isPlaying ? 'rgba(5,5,7,0.5)' : 'rgba(5,5,7,0)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          transition: '150ms ease',
        }}
          onMouseEnter={e => { e.currentTarget.style.background = 'rgba(5,5,7,0.5)'; }}
          onMouseLeave={e => { e.currentTarget.style.background = isPlaying ? 'rgba(5,5,7,0.5)' : 'rgba(5,5,7,0)'; }}
        >
          <div style={{ color: isPlaying ? '#a855f7' : '#e2e8f0', opacity: isPlaying ? 1 : 0 }}
            onMouseEnter={e => { e.currentTarget.style.opacity = '1'; }}
          >
            {isPlaying ? <IconPause /> : <IconPlay />}
          </div>
        </div>
      </div>

      {/* Info — click to explore from this track */}
      <div
        style={{ flex: 1, minWidth: 0, cursor: onSelectAnchor ? 'pointer' : 'default' }}
        onClick={() => onSelectAnchor?.(track)}
        title={onSelectAnchor ? 'Explore what comes after this track' : undefined}
      >
        <div style={{ fontSize: 12, fontWeight: 700, color: isAnchor ? '#c084fc' : '#e2e8f0', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', transition: '150ms ease' }}>
          {name}
        </div>
        <div style={{ fontSize: 10, color: '#94a3b8', marginBottom: 2 }}>{track.artist}</div>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
          {track.bpm && (
            <span style={{ fontSize: 9, color: '#00d4ff', fontFamily: "'JetBrains Mono', monospace" }}>
              {Math.round(track.bpm)} BPM
            </span>
          )}
          {track.key && (
            <span style={{ fontSize: 9, color: '#a855f7', fontFamily: "'JetBrains Mono', monospace" }}>
              {track.key}
            </span>
          )}
          {track._reason && (
            <span style={{ fontSize: 8, color: '#2d3748', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.04em' }}>
              · {track._reason}
            </span>
          )}
          {isStarter && (
            <span style={{ fontSize: 8, color: 'rgba(124,58,237,0.45)', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.06em' }}>
              STARTER
            </span>
          )}
        </div>
      </div>

      {/* Buttons */}
      <div style={{ display: 'flex', gap: 5, flexShrink: 0 }}>
        <button
          onClick={() => onFindSimilar(track)}
          aria-label={`Find similar to ${name}`}
          title="Find similar"
          style={{
            width: 28, height: 28,
            background: 'rgba(124,58,237,0.06)',
            border: '1px solid rgba(124,58,237,0.25)',
            color: 'rgba(168,85,247,0.7)',
            cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
            transition: '150ms ease',
          }}
          onMouseEnter={e => { e.currentTarget.style.background = 'rgba(124,58,237,0.15)'; e.currentTarget.style.color = '#a855f7'; }}
          onMouseLeave={e => { e.currentTarget.style.background = 'rgba(124,58,237,0.06)'; e.currentTarget.style.color = 'rgba(168,85,247,0.7)'; }}
        >
          <IconSimilar />
        </button>
        <button
          onClick={() => onAdd(track)}
          aria-label={`Add ${name} to set`}
          title="Add to set"
          style={{
            width: 28, height: 28,
            background: 'rgba(0,212,255,0.06)',
            border: '1px solid rgba(0,212,255,0.3)',
            color: '#00d4ff', cursor: 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            transition: '150ms ease',
          }}
          onMouseEnter={e => { e.currentTarget.style.background = 'rgba(0,212,255,0.15)'; }}
          onMouseLeave={e => { e.currentTarget.style.background = 'rgba(0,212,255,0.06)'; }}
        >
          <IconPlus />
        </button>
      </div>
    </div>
  );
}

// ── Direction section label ────────────────────────────────────────────────
function SectionBlock({ label, color, children }) {
  return (
    <div style={{ marginBottom: 4 }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8,
        padding: '6px 16px',
      }}>
        <div style={{ flex: 1, height: 1, background: `linear-gradient(90deg, ${color}, transparent)` }} />
        <span style={{ fontSize: 8, color, fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.18em', flexShrink: 0 }}>
          {label}
        </span>
        <div style={{ flex: 1, height: 1, background: `linear-gradient(270deg, ${color}, transparent)` }} />
      </div>
      <div style={{ padding: '0 16px' }}>{children}</div>
    </div>
  );
}

// ── Mic level meter ───────────────────────────────────────────────────────
function MicLevel({ analyser, active }) {
  const [level, setLevel] = useState(0);
  const animRef = useRef();

  useEffect(() => {
    if (!active || !analyser) { setLevel(0); return; }
    const data = new Uint8Array(analyser.frequencyBinCount);
    const tick = () => {
      analyser.getByteTimeDomainData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i++) sum += Math.abs(data[i] - 128);
      setLevel(Math.min(100, (sum / data.length) * 4));
      animRef.current = requestAnimationFrame(tick);
    };
    animRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animRef.current);
  }, [active, analyser]);

  if (!active) return null;
  return (
    <div style={{ display: 'flex', gap: 2, alignItems: 'flex-end', height: 20, padding: '0 4px' }}>
      {Array.from({ length: 12 }, (_, i) => {
        const lit = level > (i / 12) * 100;
        return (
          <div key={i} style={{
            width: 3, height: 6 + i * 1.2,
            background: lit ? (i > 9 ? '#ef4444' : i > 7 ? '#f59e0b' : '#00d4ff') : 'rgba(255,255,255,0.08)',
            transition: '80ms ease',
          }} />
        );
      })}
    </div>
  );
}

// ── Main LiveMode ──────────────────────────────────────────────────────────
export default function LiveMode({ setList, setSetList, allNodes }) {
  const [listenState, setListenState]     = useState('idle'); // idle | listening | matched
  const [detectedBPM, setDetectedBPM]     = useState(null);
  const [detectedKey, setDetectedKey]     = useState(null);
  const [matchCandidates, setMatchCandidates] = useState([]);
  const [matchedTrack, setMatchedTrack]   = useState(null);
  const [query, setQuery]                 = useState('');
  const [searchStatus, setSearchStatus]   = useState('idle');
  const [searchResults, setSearchResults] = useState([]);
  const [searchError, setSearchError]     = useState('');
  const [playingId, setPlayingId]         = useState(null);
  const [anchorTrack, setAnchorTrack]     = useState(null); // track to explore from

  const micStreamRef        = useRef(null);
  const audioCtxRef         = useRef(null);
  const analyserRef         = useRef(null);
  const sourceRef           = useRef(null); // store MediaStreamSource for proper cleanup
  const prevSpectrumRef     = useRef(null);
  const fluxHistoryRef      = useRef([]);
  const chromaAccumRef      = useRef(new Float64Array(12));
  const spectralCentroidRef = useRef({ sum: 0, count: 0 }); // for brightness matching
  const rmsEnergyRef        = useRef({ sum: 0, count: 0 });  // for energy matching
  const analysisTimerRef    = useRef(null);
  const consecutiveRef      = useRef({ id: null, count: 0 });
  const setListRef       = useRef(null);
  const audioRef         = useRef(new Audio());
  const inputRef         = useRef(null);

  // Cleanup audio + listening on unmount
  useEffect(() => {
    const a = audioRef.current;
    a.onended = () => setPlayingId(null);
    return () => {
      a.pause(); a.src = '';
      clearTimeout(analysisTimerRef.current);
      sourceRef.current?.disconnect();
      sourceRef.current = null;
      micStreamRef.current?.getTracks().forEach(t => t.stop());
      audioCtxRef.current?.close().catch(() => {});
    };
  }, []);

  // Scroll set list to end on new track
  useEffect(() => {
    if (setListRef.current) setListRef.current.scrollLeft = setListRef.current.scrollWidth;
  }, [setList.length]);

  // Default anchor = last set track; or whatever the user clicked
  const effectiveAnchor = anchorTrack ?? (setList.length > 0 ? setList[setList.length - 1] : null);
  const { starters, onTrajectory, energyUp, stepDown } = suggestDirectional(effectiveAnchor, setList, allNodes);
  const showStarters = setList.length === 0 && !anchorTrack;

  const selectAnchor = useCallback((track) => {
    const id = String(track.id || track.trackid);
    setAnchorTrack(prev => (prev && String(prev.id || prev.trackid) === id) ? null : track);
  }, []);

  // ── Audio ─────────────────────────────────────────────────────────────────
  const togglePlay = useCallback((track) => {
    const id = String(track.id || track.trackid);
    const a  = audioRef.current;

    if (playingId === id) {
      a.pause();
      setPlayingId(null);
    } else {
      a.pause();
      a.src = getAudioSrc(track);
      a.play().catch(() => {});
      setPlayingId(id);
    }
  }, [playingId]);

  // ── Add / remove ──────────────────────────────────────────────────────────
  const addTrack = useCallback((track) => {
    // Normalise: API results use title/trackid, allNodes use name/id
    const node = allNodes.find(n => String(n.id) === String(track.id || track.trackid))
      || { ...track, id: track.id || track.trackid, name: track.name || track.title };
    setSetList(prev => {
      if (prev.some(t => String(t.id) === String(node.id))) return prev;
      return [...prev, node];
    });
  }, [setSetList, allNodes]);

  const removeTrack = useCallback((index) => {
    setSetList(prev => prev.filter((_, i) => i !== index));
  }, [setSetList]);

  // ── Find similar ─────────────────────────────────────────────────────────
  const findSimilar = useCallback(async (track) => {
    const name = track.name || track.title || '';
    setQuery(`similar to ${track.artist} – ${name}`);
    setSearchStatus('searching');
    setSearchResults([]);
    setSearchError('');
    try {
      const { params } = await apiClient.post('/chat/interpret', {
        query: `tracks similar to "${name}" by ${track.artist}`,
      });
      const result = await apiClient.post('/chat/search', { params });
      setSearchResults(result.tracks || []);
      setSearchStatus('done');
    } catch (err) {
      setSearchError(err.message || 'Search failed');
      setSearchStatus('error');
    }
  }, []);

  // ── LLM search ───────────────────────────────────────────────────────────
  const runSearch = useCallback(async (q) => {
    const text = (q || query).trim();
    if (!text) return;
    setSearchStatus('searching'); setSearchResults([]); setSearchError('');
    try {
      const { params } = await apiClient.post('/chat/interpret', { query: text });
      const result     = await apiClient.post('/chat/search', { params });
      setSearchResults(result.tracks || []);
      setSearchStatus('done');
    } catch (err) {
      setSearchError(err.message || 'Search failed');
      setSearchStatus('error');
    }
  }, [query]);

  // ── Listen (audio fingerprinting) ────────────────────────────────────────
  const stopListening = useCallback(() => {
    clearTimeout(analysisTimerRef.current);
    // Disconnect source node first — this is what releases the browser mic indicator
    sourceRef.current?.disconnect();
    sourceRef.current = null;
    micStreamRef.current?.getTracks().forEach(t => t.stop());
    micStreamRef.current = null;
    audioCtxRef.current?.close().catch(() => {});
    audioCtxRef.current  = null;
    analyserRef.current  = null;
    prevSpectrumRef.current = null;
    fluxHistoryRef.current  = [];
    chromaAccumRef.current  = new Float64Array(12);
    spectralCentroidRef.current = { sum: 0, count: 0 };
    rmsEnergyRef.current        = { sum: 0, count: 0 };
    setListenState('idle');
  }, []);

  const performMatch = useCallback(() => {
    const flux = fluxHistoryRef.current;
    if (flux.length < 40) return; // need enough data

    // BPM via autocorrelation
    const frameRate = 20; // 50ms per frame = 20fps
    const bpm = detectBPMFromFlux(flux, frameRate);
    const clampedBPM = Math.max(70, Math.min(200, bpm));
    setDetectedBPM(clampedBPM);

    // Key from accumulated chroma
    const key = estimateKeyFromChroma(chromaAccumRef.current);
    if (key) setDetectedKey(key);

    // Match against library
    const played = new Set(setList.map(t => String(t.id)));
    const candidates = allNodes
      .filter(n => n.bpm && !played.has(String(n.id)))
      .map(n => {
        const bpmDiff = Math.abs(n.bpm - clampedBPM);
        const bpmS = Math.max(0, 1 - bpmDiff / 6);
        const keyS = key ? keyCompatibility(n.key, key) : 0.5;
        return { ...n, _matchScore: bpmS * 0.6 + keyS * 0.4 };
      })
      .filter(n => n._matchScore > 0.35)
      .sort((a, b) => b._matchScore - a._matchScore)
      .slice(0, 5);

    setMatchCandidates(candidates);

    // Auto-add if same top match for 3 consecutive cycles with high confidence
    if (candidates.length > 0 && candidates[0]._matchScore > 0.7) {
      const topId = String(candidates[0].id);
      if (topId === consecutiveRef.current.id) {
        consecutiveRef.current.count++;
        if (consecutiveRef.current.count >= 3) {
          addTrack(candidates[0]);
          setMatchedTrack(candidates[0]);
          consecutiveRef.current = { id: null, count: 0 };
          // Keep listening for next track
          setTimeout(() => setMatchedTrack(null), 3000);
        }
      } else {
        consecutiveRef.current = { id: topId, count: 1 };
      }
    } else {
      consecutiveRef.current = { id: null, count: 0 };
    }

    // Reset accumulators for next window
    fluxHistoryRef.current = [];
    chromaAccumRef.current = new Float64Array(12);
  }, [allNodes, setList, addTrack]);

  const startListening = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      micStreamRef.current = stream;
      const ctx = new AudioContext();
      audioCtxRef.current = ctx;
      const source = ctx.createMediaStreamSource(stream);
      sourceRef.current = source; // store so stopListening can disconnect it
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 4096;
      source.connect(analyser);
      analyserRef.current = analyser;

      prevSpectrumRef.current = null;
      fluxHistoryRef.current  = [];
      chromaAccumRef.current  = new Float64Array(12);
      consecutiveRef.current  = { id: null, count: 0 };

      const FRAME_MS = 50;
      let frameCount = 0;
      const framesPerAnalysis = Math.round(6000 / FRAME_MS); // ~120 frames = 6 seconds

      const analyze = () => {
        if (!analyserRef.current) return;

        // Spectral flux for BPM
        const freqData = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(freqData);
        if (prevSpectrumRef.current) {
          let flux = 0;
          for (let i = 0; i < freqData.length; i++) {
            const d = freqData[i] - prevSpectrumRef.current[i];
            if (d > 0) flux += d;
          }
          fluxHistoryRef.current.push(flux);
        }
        prevSpectrumRef.current = Array.from(freqData);

        // Accumulate chroma for key
        const floatData = new Float32Array(analyser.frequencyBinCount);
        analyser.getFloatFrequencyData(floatData);
        const sr = ctx.sampleRate;
        for (let i = 1; i < floatData.length; i++) {
          const freq = i * sr / analyser.fftSize;
          if (freq < 80 || freq > 4000) continue;
          const power = Math.pow(10, floatData[i] / 20);
          if (power <= 0 || !isFinite(power)) continue;
          const midi = 12 * Math.log2(freq / 440) + 69;
          const bin = ((Math.round(midi) % 12) + 12) % 12;
          chromaAccumRef.current[bin] += power;
        }

        frameCount++;
        if (frameCount % framesPerAnalysis === 0 && fluxHistoryRef.current.length > 50) {
          performMatch();
        }

        analysisTimerRef.current = setTimeout(analyze, FRAME_MS);
      };

      analyze();
      setListenState('listening');
      setDetectedBPM(null);
      setDetectedKey(null);
      setMatchCandidates([]);
      setMatchedTrack(null);
    } catch { setListenState('idle'); }
  }, [performMatch]);

  const isBusy         = searchStatus === 'searching';
  const sessionMinutes = setList.length > 0 ? Math.round(setList.length * 6.5) : 0;

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', height: '100%',
      background: '#050507', color: '#e2e8f0',
      fontFamily: "'Inter', system-ui, sans-serif", overflow: 'hidden',
    }}>

      {/* ── Header ────────────────────────────────────────────────────────── */}
      <div style={{
        padding: '14px 24px',
        borderBottom: '1px solid rgba(124,58,237,0.15)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        background: 'rgba(8,8,20,0.6)', flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
          <div>
            <div style={{ fontSize: 8, letterSpacing: '0.3em', color: 'rgba(124,58,237,0.6)', fontFamily: "'JetBrains Mono', monospace", marginBottom: 3 }}>
              LIVE SESSION
            </div>
            <div style={{ fontSize: 18, fontWeight: 700, color: '#e2e8f0' }}>SET TRACKER</div>
          </div>

          {setList.length > 0 && (
            <div style={{ display: 'flex', gap: 20 }}>
              <div>
                <div style={{ fontSize: 8, color: '#475569', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.2em', marginBottom: 2 }}>TRACKS</div>
                <div style={{ fontSize: 20, fontWeight: 700, color: '#00d4ff', fontFamily: "'JetBrains Mono', monospace" }}>{String(setList.length).padStart(2, '0')}</div>
              </div>
              <div>
                <div style={{ fontSize: 8, color: '#475569', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.2em', marginBottom: 2 }}>EST. DURATION</div>
                <div style={{ fontSize: 20, fontWeight: 700, color: '#a855f7', fontFamily: "'JetBrains Mono', monospace" }}>~{sessionMinutes}m</div>
              </div>
              {setList[setList.length - 1]?.bpm && (
                <div>
                  <div style={{ fontSize: 8, color: '#475569', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.2em', marginBottom: 2 }}>CURRENT BPM</div>
                  <div style={{ fontSize: 20, fontWeight: 700, color: '#00d4ff', fontFamily: "'JetBrains Mono', monospace" }}>{Math.round(setList[setList.length - 1].bpm)}</div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Listen toggle */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <MicLevel analyser={analyserRef.current} active={listenState === 'listening'} />
          {listenState === 'listening' && detectedBPM && (
            <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
              <span style={{ fontSize: 9, color: '#00d4ff', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.08em' }}>
                {detectedBPM} BPM
              </span>
              {detectedKey && (
                <span style={{ fontSize: 9, color: '#a855f7', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.08em' }}>
                  {detectedKey}
                </span>
              )}
            </div>
          )}
          {matchedTrack && (
            <span style={{ fontSize: 9, color: '#22c55e', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.08em', animation: 'blink 1s ease-in-out 3' }}>
              ADDED: {matchedTrack.name || matchedTrack.title}
            </span>
          )}
          <button
            onClick={listenState === 'listening' ? stopListening : startListening}
            style={{
              display: 'flex', alignItems: 'center', gap: 8, padding: '8px 16px',
              background: listenState === 'listening' ? 'rgba(239,68,68,0.1)' : 'rgba(124,58,237,0.08)',
              border: `1px solid ${listenState === 'listening' ? 'rgba(239,68,68,0.5)' : 'rgba(124,58,237,0.4)'}`,
              color: listenState === 'listening' ? '#ef4444' : '#a855f7',
              cursor: 'pointer', fontSize: 10, fontWeight: 700, letterSpacing: '0.12em',
              fontFamily: "'Inter', system-ui, sans-serif", transition: '150ms ease',
            }}
            onMouseEnter={e => { e.currentTarget.style.background = listenState === 'listening' ? 'rgba(239,68,68,0.18)' : 'rgba(124,58,237,0.15)'; }}
            onMouseLeave={e => { e.currentTarget.style.background = listenState === 'listening' ? 'rgba(239,68,68,0.1)' : 'rgba(124,58,237,0.08)'; }}
          >
            <IconMic />
            {listenState === 'listening' ? 'LISTENING' : 'LISTEN'}
          </button>
        </div>
      </div>

      {/* ── Set list ──────────────────────────────────────────────────────── */}
      <div style={{
        borderBottom: '1px solid rgba(124,58,237,0.12)',
        background: 'rgba(5,5,7,0.8)', flexShrink: 0,
      }}>
        {setList.length === 0 ? (
          <div style={{
            padding: '20px 24px', color: '#2d3748', fontSize: 10,
            fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.15em',
          }}>
            ADD A TRACK BELOW TO START YOUR SET
          </div>
        ) : (
          <div
            ref={setListRef}
            style={{
              display: 'flex', gap: 20, padding: '24px 24px 16px',
              overflowX: 'auto', overflowY: 'visible',
              scrollbarWidth: 'thin', scrollbarColor: 'rgba(124,58,237,0.3) transparent',
            }}
          >
            {setList.map((track, i) => (
              <SetTrackArt
                key={`${track.id}-${i}`}
                track={track}
                index={i}
                isLatest={i === setList.length - 1}
                isPlaying={playingId === String(track.id)}
                isAnchor={anchorTrack && String(anchorTrack.id || anchorTrack.trackid) === String(track.id)}
                onRemove={removeTrack}
                onPlay={togglePlay}
                onSelectAnchor={selectAnchor}
              />
            ))}
          </div>
        )}
      </div>

      {/* ── Main content ──────────────────────────────────────────────────── */}
      <div style={{ flex: 1, display: 'flex', gap: 0, overflow: 'hidden', minHeight: 0 }}>

        {/* Left: Trajectory / Starter suggestions */}
        <div style={{
          width: 400, flexShrink: 0,
          borderRight: '1px solid rgba(124,58,237,0.12)',
          display: 'flex', flexDirection: 'column', overflow: 'hidden',
        }}>
          {/* Panel header */}
          <div style={{
            padding: '12px 16px', flexShrink: 0,
            background: 'rgba(8,8,24,0.5)',
            borderBottom: '1px solid rgba(124,58,237,0.1)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <IconVector />
              <span style={{ fontSize: 10, fontWeight: 700, color: '#94a3b8', letterSpacing: '0.12em', fontFamily: "'JetBrains Mono', monospace" }}>
                {showStarters ? 'STARTER SUGGESTIONS' : 'SET TRAJECTORY'}
              </span>
              {anchorTrack && (
                <button
                  onClick={() => setAnchorTrack(null)}
                  style={{ marginLeft: 'auto', background: 'none', border: 'none', color: '#475569', cursor: 'pointer', fontSize: 9, fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.08em', padding: '2px 6px', transition: '150ms ease' }}
                  onMouseEnter={e => { e.currentTarget.style.color = '#94a3b8'; }}
                  onMouseLeave={e => { e.currentTarget.style.color = '#475569'; }}
                >
                  RESET ×
                </button>
              )}
            </div>
            {anchorTrack && (
              <div style={{ marginTop: 5, fontSize: 9, color: '#a855f7', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.06em', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                AFTER: {anchorTrack.name || anchorTrack.title} · {anchorTrack.artist}
              </div>
            )}
          </div>

          <div style={{ flex: 1, overflowY: 'auto', padding: '10px 0' }}>
            {/* Listen match candidates */}
            {matchCandidates.length > 0 && (
              <div style={{ padding: '0 16px 12px', borderBottom: '1px solid rgba(124,58,237,0.1)', marginBottom: 8 }}>
                <div style={{ fontSize: 8, color: '#f59e0b', letterSpacing: '0.2em', fontFamily: "'JetBrains Mono', monospace", marginBottom: 8 }}>
                  LISTENING — POSSIBLE MATCHES
                </div>
                {matchCandidates.map(t => (
                  <TrackRow key={t.id} track={t} isPlaying={playingId === String(t.id)}
                    onAdd={addTrack} onPlay={togglePlay} onFindSimilar={findSimilar} onSelectAnchor={selectAnchor} />
                ))}
              </div>
            )}

            {/* Starters when empty */}
            {showStarters && starters.length > 0 && (
              <div style={{ padding: '0 16px' }}>
                {starters.map(t => (
                  <TrackRow key={t.id} track={t} isPlaying={playingId === String(t.id)}
                    isAnchor={anchorTrack && String(anchorTrack.id) === String(t.id)}
                    onAdd={addTrack} onPlay={togglePlay} onFindSimilar={findSimilar}
                    onSelectAnchor={selectAnchor} isStarter />
                ))}
              </div>
            )}

            {/* Directional sections */}
            {!showStarters && (
              <div>
                {onTrajectory.length > 0 && (
                  <SectionBlock label="ON TRAJECTORY" color="rgba(0,212,255,0.5)">
                    {onTrajectory.map(t => (
                      <TrackRow key={t.id} track={t} isPlaying={playingId === String(t.id)}
                        isAnchor={anchorTrack && String(anchorTrack.id) === String(t.id)}
                        onAdd={addTrack} onPlay={togglePlay} onFindSimilar={findSimilar} onSelectAnchor={selectAnchor} />
                    ))}
                  </SectionBlock>
                )}

                {energyUp.length > 0 && (
                  <SectionBlock label="ENERGY UP ↑" color="rgba(239,68,68,0.5)">
                    {energyUp.map(t => (
                      <TrackRow key={t.id} track={t} isPlaying={playingId === String(t.id)}
                        isAnchor={anchorTrack && String(anchorTrack.id) === String(t.id)}
                        onAdd={addTrack} onPlay={togglePlay} onFindSimilar={findSimilar} onSelectAnchor={selectAnchor} />
                    ))}
                  </SectionBlock>
                )}

                {stepDown.length > 0 && (
                  <SectionBlock label="STEP DOWN ↓" color="rgba(14,165,233,0.5)">
                    {stepDown.map(t => (
                      <TrackRow key={t.id} track={t} isPlaying={playingId === String(t.id)}
                        isAnchor={anchorTrack && String(anchorTrack.id) === String(t.id)}
                        onAdd={addTrack} onPlay={togglePlay} onFindSimilar={findSimilar} onSelectAnchor={selectAnchor} />
                    ))}
                  </SectionBlock>
                )}

                {onTrajectory.length === 0 && energyUp.length === 0 && stepDown.length === 0 && matchCandidates.length === 0 && (
                  <div style={{ padding: '16px', color: '#2d3748', fontSize: 10, fontFamily: "'JetBrains Mono', monospace", lineHeight: 1.8, letterSpacing: '0.08em' }}>
                    <div>ADD 2+ TRACKS TO ACTIVATE</div>
                    <div>VECTOR-BASED SUGGESTIONS.</div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Right: Search */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <div style={{
            padding: '12px 16px', flexShrink: 0,
            background: 'rgba(8,8,24,0.5)',
            borderBottom: '1px solid rgba(124,58,237,0.1)',
            display: 'flex', alignItems: 'center', gap: 8,
          }}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="rgba(124,58,237,0.6)" strokeWidth="2.5" strokeLinecap="round">
              <circle cx="11" cy="11" r="7" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
            <span style={{ fontSize: 10, fontWeight: 700, color: '#94a3b8', letterSpacing: '0.12em', fontFamily: "'JetBrains Mono', monospace" }}>
              SEARCH LIBRARY
            </span>
          </div>

          {/* Search input */}
          <div style={{ padding: '14px 16px', borderBottom: '1px solid rgba(124,58,237,0.08)', flexShrink: 0 }}>
            <div style={{ display: 'flex', gap: 8 }}>
              <input
                ref={inputRef}
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') runSearch(); }}
                disabled={isBusy}
                placeholder='e.g. "dark minimal 140" or "something to close with"'
                style={{
                  flex: 1, background: 'rgba(8,8,20,0.8)',
                  border: '1px solid rgba(124,58,237,0.18)',
                  color: '#e2e8f0', fontSize: 13, padding: '10px 12px',
                  fontFamily: "'JetBrains Mono', monospace",
                  outline: 'none', caretColor: '#7c3aed',
                  opacity: isBusy ? 0.5 : 1, borderRadius: 0,
                }}
                onFocus={e => { e.target.style.borderColor = 'rgba(124,58,237,0.55)'; }}
                onBlur={e => { e.target.style.borderColor = 'rgba(124,58,237,0.18)'; }}
              />
              <button
                onClick={() => runSearch()}
                disabled={isBusy || !query.trim()}
                style={{
                  padding: '0 16px', borderRadius: 0,
                  background: isBusy || !query.trim() ? 'rgba(8,8,20,0.5)' : 'linear-gradient(135deg, rgba(124,58,237,0.5), rgba(0,212,255,0.3))',
                  border: `1px solid ${isBusy || !query.trim() ? 'rgba(124,58,237,0.1)' : 'rgba(124,58,237,0.45)'}`,
                  color: isBusy || !query.trim() ? '#2d3748' : '#e2e8f0',
                  cursor: isBusy || !query.trim() ? 'default' : 'pointer',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  transition: '150ms ease',
                }}
              >
                {isBusy
                  ? <div style={{ width: 14, height: 14, border: '1.5px solid rgba(124,58,237,0.2)', borderTop: '1.5px solid #7c3aed', animation: 'spin 0.8s linear infinite' }} />
                  : <IconSend />}
              </button>
            </div>
          </div>

          {searchStatus === 'error' && (
            <div style={{ margin: '10px 16px', padding: '10px 12px', background: 'rgba(16,5,5,0.8)', border: '1px solid rgba(239,68,68,0.25)', fontSize: 11, color: 'rgba(239,68,68,0.8)', flexShrink: 0 }}>
              {searchError}
            </div>
          )}

          {/* Results */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '10px 0' }}>
            {searchStatus === 'searching' && (
              <div style={{ padding: '16px', fontSize: 10, color: 'rgba(124,58,237,0.6)', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.08em', display: 'flex', gap: 8, alignItems: 'center' }}>
                <div style={{ width: 12, height: 12, border: '1.5px solid rgba(124,58,237,0.15)', borderTop: '1.5px solid #7c3aed', animation: 'spin 0.8s linear infinite' }} />
                SEARCHING LIBRARY...
              </div>
            )}

            {searchResults.length > 0 && (
              <div style={{ padding: '0 16px' }}>
                <div style={{ fontSize: 8, color: '#475569', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.2em', marginBottom: 10 }}>
                  {searchResults.length} RESULTS
                </div>
                {searchResults.map((t, i) => (
                  <TrackRow
                    key={t.trackid || i}
                    track={{ ...t, id: t.trackid, name: t.title }}
                    isPlaying={playingId === String(t.trackid)}
                    onAdd={addTrack}
                    onPlay={togglePlay}
                    onFindSimilar={findSimilar}
                  />
                ))}
              </div>
            )}

            {searchStatus === 'done' && searchResults.length === 0 && (
              <div style={{ padding: '16px', fontSize: 10, color: '#2d3748', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.08em' }}>
                NO RESULTS FOUND
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
