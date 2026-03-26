// src/components/LiveMode.jsx
import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { m, AnimatePresence } from 'framer-motion';
import { apiClient } from '../api/apiClient';
import GlassPanel from './ui/GlassPanel';
import TagEditor from './TagEditor';
import CrateBuilder from './CrateBuilder';
import Toast from './ui/Toast';
import { makeSupabaseCoverUrl } from '../utils/coverUrl';
import { useAuthStore } from '../stores/authStore';
import { isFileSystemAccessSupported, pickMusicFolder } from '../utils/localAudio';
import { IconMic, IconPlus, IconClose, IconSend, IconVector, IconPlay, IconPause, IconSearch, IconEdit } from './icons';

const API_BASE = import.meta.env.VITE_API_URL
  || (import.meta.env.DEV ? 'http://localhost:8000' : 'https://djmate.onrender.com');

// ── Shared helpers ─────────────────────────────────────────────────────────
function getAudioSrc(track) {
  return track.audioUrl || track.audio_url
    || `${API_BASE}/tracks/${track.id || track.trackid}/audio`;
}

// ── 7D Trajectory calculation ──────────────────────────────────────────────
function normBpm(bpm) { return Math.max(0, Math.min(1, ((bpm || 130) - 60) / 140)); }

function camelotToNum(key) {
  const parsed = parseCamelot(key);
  if (!parsed) return 0.5;
  return ((parsed.num - 1) / 11) + (parsed.letter === 'B' ? 0.5 / 12 : 0);
}

function vibeHash(tags) {
  if (!Array.isArray(tags) || tags.length === 0) return 0.5;
  const sum = tags.join('').split('').reduce((s, c) => s + c.charCodeAt(0), 0);
  return (sum % 100) / 100;
}

function trackTo7D(track) {
  return {
    x: track.x || 0,
    y: track.y || 0,
    z: track.z || 0,
    bpm_norm: normBpm(track.bpm),
    energy: parseFloat(track.energy ?? 5),
    key_num: camelotToNum(track.key),
    vibe_hash: vibeHash(track.semanticTags || track.semantic_tags),
  };
}

function getTrajectoryVector(setList) {
  if (setList.length < 2) return null;
  const recent = setList.slice(-Math.min(4, setList.length));
  const dims = ['x', 'y', 'z', 'bpm_norm', 'energy', 'key_num', 'vibe_hash'];
  const vectors = [];
  for (let i = 1; i < recent.length; i++) {
    const a = trackTo7D(recent[i - 1]);
    const b = trackTo7D(recent[i]);
    const delta = {};
    dims.forEach(d => { delta[d] = b[d] - a[d]; });
    vectors.push(delta);
  }
  const avg = {};
  dims.forEach(d => {
    avg[d] = vectors.reduce((s, v) => s + v[d], 0) / vectors.length;
  });
  const mag = Math.sqrt(dims.reduce((s, d) => s + avg[d] ** 2, 0));
  if (mag < 0.001) return null;
  const norm = {};
  dims.forEach(d => { norm[d] = avg[d] / mag; });
  norm._raw = avg;
  return norm;
}

function getSetDirectionSummary(setList) {
  if (setList.length < 2) return null;
  const recent = setList.slice(-Math.min(4, setList.length));
  const deltas = { energy: 0, bpm_norm: 0, key_num: 0, vibe_hash: 0 };
  const dims = Object.keys(deltas);
  for (let i = 1; i < recent.length; i++) {
    const a = trackTo7D(recent[i - 1]);
    const b = trackTo7D(recent[i]);
    dims.forEach(d => { deltas[d] += b[d] - a[d]; });
  }
  const n = recent.length - 1;
  dims.forEach(d => { deltas[d] /= n; });

  const parts = [];
  if (deltas.energy > 0.5) parts.push('Building energy');
  else if (deltas.energy < -0.5) parts.push('Dropping down');
  else parts.push('Maintaining groove');

  if (Math.abs(deltas.bpm_norm) > 0.03) {
    parts.push(deltas.bpm_norm > 0 ? 'Tempo rising' : 'Tempo falling');
  }
  if (Math.abs(deltas.vibe_hash) > 0.08) parts.push('Shifting genre');
  if (Math.abs(deltas.key_num) > 0.05) parts.push('Key drifting');

  return parts.join(' / ');
}

function suggestDirectional(anchor, setList, allNodes) {
  const played = new Set(setList.map(t => String(t.id)));
  const candidates = allNodes.filter(n => !played.has(String(n.id)) && String(n.id) !== String(anchor?.id));

  if (!anchor) {
    return {
      starters:      candidates.sort(() => Math.random() - 0.5).slice(0, 8),
      onTrajectory:  [],
      energyUp:      [],
      stepDown:      [],
    };
  }

  const dir = setList.length >= 2 ? getTrajectoryVector(setList) : null;
  const anchorVec = trackTo7D(anchor);

  // Compute energy delta trend from recent tracks
  let energyDelta = 0;
  if (setList.length >= 2) {
    const tail = setList.slice(-Math.min(4, setList.length));
    for (let i = 1; i < tail.length; i++) {
      energyDelta += parseFloat(tail[i].energy ?? 5) - parseFloat(tail[i - 1].energy ?? 5);
    }
    energyDelta /= (tail.length - 1);
  }

  const scored = candidates.map(n => {
    const nVec = trackTo7D(n);

    // Spatial score (15%) — proximity in x/y/z
    const dx = nVec.x - anchorVec.x;
    const dy = nVec.y - anchorVec.y;
    const dz = nVec.z - anchorVec.z;
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
    const dot  = dir ? (dx * dir.x + dy * dir.y + dz * dir.z) / dist : 0;
    const dirScore  = dir ? (dot + 1) / 2 : 0.5;
    const proxScore = 1 / (1 + dist * 0.0008);
    const spatialScore = dirScore * 0.6 + proxScore * 0.4;

    // BPM score (20%) — penalize >8 BPM difference hard
    const bpmDiff = Math.abs((n.bpm || 130) - (anchor.bpm || 130));
    const bpmScore = bpmDiff <= 8 ? 1 - (bpmDiff / 8) * 0.3 : Math.max(0, 1 - bpmDiff / 15);

    // Energy score (25%) — reward continuing the energy delta trend
    const nEnergy = parseFloat(n.energy ?? 5);
    const aEnergy = parseFloat(anchor.energy ?? 5);
    const candidateDelta = nEnergy - aEnergy;
    const trendAlignment = energyDelta !== 0
      ? (Math.sign(candidateDelta) === Math.sign(energyDelta) ? 0.3 : -0.1)
      : 0;
    const energyProximity = 1 - Math.min(1, Math.abs(candidateDelta) / 5);
    const energyScore = Math.max(0, Math.min(1, energyProximity + trendAlignment));

    // Key score (20%) — Camelot compatibility
    const keyScore = keyCompatibility(n.key, anchor.key);

    // Vibe score (20%) — semantic_tags overlap
    const anchorTags = Array.isArray(anchor.semanticTags) ? anchor.semanticTags : [];
    const nTags = Array.isArray(n.semanticTags) ? n.semanticTags : [];
    let vibeScore = 0.5;
    if (anchorTags.length > 0 && nTags.length > 0) {
      const aSet = new Set(anchorTags.map(t => t.toLowerCase()));
      const overlap = nTags.filter(t => aSet.has(t.toLowerCase())).length;
      vibeScore = overlap / Math.max(anchorTags.length, nTags.length);
    }

    const _score = spatialScore * 0.15 + bpmScore * 0.20 + energyScore * 0.25 + keyScore * 0.20 + vibeScore * 0.20;
    return { ...n, _score, _dist: dist, _dot: dot };
  });

  const anchorEnergy = parseFloat(anchor.energy ?? 5);

  const onTrajectory = scored
    .filter(n => n._score > 0.3)
    .sort((a, b) => b._score - a._score)
    .slice(0, 5)
    .map(n => ({ ...n, _reason: getDirectionReason(n, anchor) }));

  const energyUp = candidates
    .filter(n => parseFloat(n.energy ?? 5) > anchorEnergy + 1)
    .filter(n => keyCompatibility(n.key, anchor.key) >= 0.3)
    .sort((a, b) => {
      const sa = keyCompatibility(a.key, anchor.key) + (1 - Math.abs((a.bpm||130)-(anchor.bpm||130))/15);
      const sb = keyCompatibility(b.key, anchor.key) + (1 - Math.abs((b.bpm||130)-(anchor.bpm||130))/15);
      return sb - sa;
    })
    .slice(0, 3)
    .map(n => ({ ...n, _reason: getDirectionReason(n, anchor) }));

  const stepDown = candidates
    .filter(n => parseFloat(n.energy ?? 5) < anchorEnergy - 1)
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
  const nE = parseFloat(n.energy ?? 5);
  const aE = parseFloat(anchor.energy ?? 5);
  if (nE > aE + 1) parts.push('higher energy');
  else if (nE < aE - 1) parts.push('lower energy');
  const tags = Array.isArray(n.semanticTags) ? n.semanticTags : [];
  if (tags.length > 0) parts.push(tags[0]);
  if (n.key) parts.push(n.key);
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
  if (code) { const m2 = code.match(/^(\d+)([AB])$/); return { num: parseInt(m2[1]), letter: m2[2] }; }
  return null;
}

function keyCompatibility(key1, key2) {
  const c1 = parseCamelot(key1), c2 = parseCamelot(key2);
  if (!c1 || !c2) return 0.5;
  if (c1.num === c2.num && c1.letter === c2.letter) return 1.0;
  if (c1.num === c2.num) return 0.8;
  const diff = Math.min(Math.abs(c1.num - c2.num), 12 - Math.abs(c1.num - c2.num));
  if (c1.letter === c2.letter && diff === 1) return 0.85;
  if (c1.letter === c2.letter && diff === 2) return 0.5;
  if (c1.letter !== c2.letter && diff <= 1) return 0.7;
  return 0.15;
}

// ── MFCC fingerprint helpers ──────────────────────────────────────────────
const NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];

function hzToMel(f)  { return 2595 * Math.log10(1 + f / 700); }
function melToHz(m2) { return 700 * (Math.pow(10, m2 / 2595) - 1); }

function buildMelFilterbank(sr, fftSize, nMels = 40, fmin = 20, fmax = 8000) {
  const nBins  = Math.floor(fftSize / 2) + 1;
  const melMin = hzToMel(fmin);
  const melMax = hzToMel(fmax);
  const melPts = Array.from({ length: nMels + 2 }, (_, i) =>
    melMin + (melMax - melMin) * i / (nMels + 1));
  const hzPts  = melPts.map(melToHz);
  const binPts = hzPts.map(hz => Math.min(nBins - 1, Math.floor(hz * fftSize / sr)));

  return Array.from({ length: nMels }, (_, m2) => {
    const row = new Float32Array(nBins);
    const lo = binPts[m2], mid = binPts[m2 + 1], hi = binPts[m2 + 2];
    for (let k = lo; k < mid; k++) row[k] = (k - lo) / Math.max(1, mid - lo);
    for (let k = mid; k < hi;  k++) row[k] = (hi - k) / Math.max(1, hi - mid);
    return row;
  });
}

function computeMFCC(floatFreqData, filterbank, nMFCC = 13) {
  const nMels = filterbank.length;
  const logMel = filterbank.map(row => {
    let e = 0;
    for (let k = 0; k < row.length; k++) {
      const linearAmp = Math.pow(10, floatFreqData[k] / 20);
      e += row[k] * linearAmp * linearAmp;
    }
    return Math.log(Math.max(e, 1e-10));
  });
  return Array.from({ length: nMFCC }, (_, n) => {
    let s = 0;
    for (let m2 = 0; m2 < nMels; m2++)
      s += logMel[m2] * Math.cos(Math.PI * n * (m2 + 0.5) / nMels);
    return s * (n === 0 ? Math.sqrt(1 / nMels) : Math.sqrt(2 / nMels));
  });
}

function cosine(a, b) {
  let dot = 0, ma = 0, mb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; ma += a[i] * a[i]; mb += b[i] * b[i]; }
  return (ma > 0 && mb > 0) ? dot / (Math.sqrt(ma) * Math.sqrt(mb)) : 0;
}

function normaliseMFCC(v) {
  const mean = v.reduce((s, x) => s + x, 0) / v.length;
  const c    = v.map(x => x - mean);
  const norm = Math.sqrt(c.reduce((s, x) => s + x * x, 0));
  return norm > 1e-10 ? c.map(x => x / norm) : c;
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

// Icons imported from ./icons
// IconX alias for LiveMode (was 11px close icon)
const IconX = (props) => <IconClose size={11} {...props} />;
// IconSimilar alias for LiveMode (was 11px search icon)
const IconSimilar = (props) => <IconSearch size={11} {...props} />;

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
        style={{ width: size, height: size, objectFit: 'cover', display: 'block', flexShrink: 0, borderRadius: 'var(--radius-sm)' }} />
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
      borderRadius: 'var(--radius-sm)',
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
  return (
    <m.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ type: 'spring', damping: 22, stiffness: 300, delay: index * 0.03 }}
      style={{ position: 'relative', flexShrink: 0 }}
    >
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
      <m.div
        whileHover={{ scale: 1.05 }}
        style={{
          width: 80, height: 80, position: 'relative', overflow: 'hidden',
          borderRadius: 'var(--radius-md)',
          border: isAnchor  ? '2px solid #a855f7'
                : isLatest  ? '2px solid #00d4ff'
                : isPlaying ? '2px solid rgba(168,85,247,0.6)'
                : '1px solid var(--glass-border)',
          boxShadow: isAnchor  ? '0 0 18px rgba(168,85,247,0.35)'
                   : isLatest  ? '0 0 16px rgba(0,212,255,0.3)'
                   : 'none',
          cursor: 'pointer', transition: 'border-color 200ms ease, box-shadow 200ms ease',
        }}
        onClick={() => onPlay(track)}
      >
        <TrackArt track={track} size={80} />

        <m.div
          initial={{ opacity: 0 }}
          whileHover={{ opacity: 1 }}
          style={{
            position: 'absolute', inset: 0, background: 'rgba(5,5,7,0.55)',
            borderRadius: 'var(--radius-md)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            opacity: isPlaying ? 1 : 0,
          }}
        >
          <div style={{ color: isPlaying ? '#a855f7' : '#e2e8f0' }}>
            {isPlaying ? <IconPause /> : <IconPlay />}
          </div>
        </m.div>

        {(isLatest || isPlaying || isAnchor) && (
          <div style={{
            position: 'absolute', bottom: 0, left: 0, right: 0,
            borderRadius: '0 0 var(--radius-md) var(--radius-md)',
            background: isAnchor  ? 'rgba(168,85,247,0.2)'
                      : isPlaying ? 'rgba(168,85,247,0.18)'
                      : 'rgba(0,212,255,0.15)',
            borderTop: `1px solid ${isAnchor ? 'rgba(168,85,247,0.6)' : isPlaying ? 'rgba(168,85,247,0.5)' : 'rgba(0,212,255,0.4)'}`,
            fontSize: 7,
            color: isAnchor ? '#a855f7' : isPlaying ? '#a855f7' : '#00d4ff',
            textAlign: 'center', padding: '2px 0',
            letterSpacing: '0.12em', fontFamily: "'JetBrains Mono', monospace",
          }}>
            {isAnchor ? 'EXPLORE' : isPlaying ? 'PLAYING' : 'LATEST'}
          </div>
        )}
      </m.div>

      {/* Remove */}
      <m.button
        onClick={() => onRemove(index)}
        aria-label="Remove from set"
        whileHover={{ scale: 1.1, background: 'rgba(239,68,68,0.15)', color: '#ef4444' }}
        whileTap={{ scale: 0.9 }}
        style={{
          position: 'absolute', top: -6, right: -6, width: 18, height: 18,
          borderRadius: '50%',
          background: 'rgba(5,5,7,0.9)', border: '1px solid rgba(239,68,68,0.4)',
          color: 'rgba(239,68,68,0.6)', cursor: 'pointer',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}
      >
        <IconX />
      </m.button>

      {/* Track info */}
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
    </m.div>
  );
}

// ── Suggestion / search result row ─────────────────────────────────────────
const TrackRow = React.memo(function TrackRow({ track, isPlaying, isAnchor, onAdd, onPlay, onFindSimilar, onEdit, onSelectAnchor, isStarter = false, index = 0 }) {
  const name = track.name || track.title || 'Unknown';
  return (
    <m.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: 'spring', damping: 26, stiffness: 300, delay: index * 0.03 }}
      whileHover={{ backgroundColor: isAnchor ? 'rgba(124,58,237,0.12)' : 'rgba(16,16,36,0.7)' }}
      style={{
        display: 'flex', gap: 10, alignItems: 'center',
        padding: '9px 14px',
        borderRadius: 'var(--radius-md)',
        background: isAnchor ? 'rgba(124,58,237,0.08)' : 'var(--bg-card)',
        border: `1px solid ${isAnchor ? 'rgba(168,85,247,0.4)' : 'var(--glass-border)'}`,
        marginBottom: 4,
        cursor: 'default',
      }}
    >
      {/* Art with play overlay */}
      <div
        style={{ position: 'relative', flexShrink: 0, cursor: 'pointer' }}
        onClick={() => onPlay(track)}
      >
        <TrackArt track={{ ...track, name }} size={44} />
        <m.div
          style={{
            position: 'absolute', inset: 0,
            borderRadius: 'var(--radius-sm)',
            background: isPlaying ? 'rgba(5,5,7,0.5)' : 'rgba(5,5,7,0)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}
          whileHover={{ background: 'rgba(5,5,7,0.5)' }}
        >
          <div style={{ color: isPlaying ? '#a855f7' : '#e2e8f0', opacity: isPlaying ? 1 : 0, transition: '120ms ease' }}>
            {isPlaying ? <IconPause /> : <IconPlay />}
          </div>
        </m.div>
      </div>

      {/* Info */}
      <div style={{ flex: 1, minWidth: 0, cursor: 'default' }}>
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
            <span style={{
              fontSize: 8, color: 'rgba(124,58,237,0.6)',
              fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.06em',
              background: 'rgba(124,58,237,0.08)', borderRadius: 'var(--radius-pill)',
              padding: '1px 6px',
            }}>
              STARTER
            </span>
          )}
        </div>
      </div>

      {/* Buttons */}
      <div style={{ display: 'flex', gap: 5, flexShrink: 0 }}>
        {onEdit && (
          <m.button
            onClick={() => onEdit(track)}
            aria-label={`Edit tags for ${name}`}
            title="Edit tags"
            whileHover={{ scale: 1.1, background: 'rgba(124,58,237,0.15)', color: '#c084fc' }}
            whileTap={{ scale: 0.9 }}
            style={{
              width: 28, height: 28, borderRadius: 'var(--radius-sm)',
              background: 'rgba(124,58,237,0.04)',
              border: '1px solid rgba(124,58,237,0.18)',
              color: 'rgba(168,85,247,0.5)',
              cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}
          >
            <IconEdit />
          </m.button>
        )}
        <m.button
          onClick={() => onFindSimilar(track)}
          aria-label={`Find similar to ${name}`}
          title="Find similar"
          whileHover={{ scale: 1.1, background: 'rgba(124,58,237,0.15)', color: '#a855f7' }}
          whileTap={{ scale: 0.9 }}
          style={{
            width: 28, height: 28, borderRadius: 'var(--radius-sm)',
            background: 'rgba(124,58,237,0.06)',
            border: '1px solid rgba(124,58,237,0.25)',
            color: 'rgba(168,85,247,0.7)',
            cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}
        >
          <IconSimilar />
        </m.button>
        <m.button
          onClick={() => onAdd(track)}
          aria-label={`Add ${name} to set`}
          title="Add to set"
          whileHover={{ scale: 1.1, background: 'rgba(0,212,255,0.15)' }}
          whileTap={{ scale: 0.9 }}
          style={{
            width: 28, height: 28, borderRadius: 'var(--radius-sm)',
            background: 'rgba(0,212,255,0.06)',
            border: '1px solid rgba(0,212,255,0.3)',
            color: '#00d4ff', cursor: 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}
        >
          <IconPlus />
        </m.button>
      </div>
    </m.div>
  );
});

// ── Direction section label ────────────────────────────────────────────────
function SectionBlock({ label, color, children }) {
  return (
    <div style={{ marginBottom: 4 }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8,
        padding: '6px 16px',
      }}>
        <div style={{ flex: 1, height: 1, background: `linear-gradient(90deg, ${color}, transparent)`, borderRadius: 'var(--radius-pill)' }} />
        <span style={{
          fontSize: 8, color, fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.18em', flexShrink: 0,
          background: 'rgba(8,8,20,0.4)', borderRadius: 'var(--radius-pill)', padding: '2px 10px',
        }}>
          {label}
        </span>
        <div style={{ flex: 1, height: 1, background: `linear-gradient(270deg, ${color}, transparent)`, borderRadius: 'var(--radius-pill)' }} />
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
            borderRadius: 'var(--radius-xs)',
            background: lit ? (i > 9 ? '#a855f7' : i > 7 ? '#7c3aed' : '#00d4ff') : 'rgba(255,255,255,0.08)',
            transition: '80ms ease',
          }} />
        );
      })}
    </div>
  );
}

// ── Main LiveMode ──────────────────────────────────────────────────────────
export default function LiveMode({ setList, setSetList, allNodes }) {
  const [listenState, setListenState]     = useState('idle');
  const [detectedBPM, setDetectedBPM]     = useState(null);
  const [detectedKey, setDetectedKey]     = useState(null);
  const [matchCandidates, setMatchCandidates] = useState([]);
  const [matchedTrack, setMatchedTrack]   = useState(null);
  const [query, setQuery]                 = useState('');
  const [searchStatus, setSearchStatus]   = useState('idle');
  const [searchResults, setSearchResults] = useState([]);
  const [searchError, setSearchError]     = useState('');
  const [playingId, setPlayingId]         = useState(null);
  const [anchorTrack, setAnchorTrack]     = useState(null);
  const [editingTrack, setEditingTrack]   = useState(null);
  const [availableTags, setAvailableTags] = useState([]);
  const [availableVibes, setAvailableVibes] = useState([]);
  const [liveSubMode, setLiveSubMode]     = useState('tracker');
  const [uploadToast, setUploadToast]     = useState({ visible: false, message: '' });

  const profile = useAuthStore(s => s.profile);
  const setImportedLibrary = useAuthStore(s => s.setImportedLibrary);

  const micStreamRef     = useRef(null);
  const audioCtxRef      = useRef(null);
  const analyserRef      = useRef(null);
  const sourceRef        = useRef(null);
  const melFBRef         = useRef(null);
  const mfccAccumRef     = useRef(null);
  const mfccFrameCountRef= useRef(0);
  const chromaAccumRef   = useRef(new Float64Array(12));
  const prevSpectrumRef  = useRef(null);
  const analysisTimerRef = useRef(null);
  const consecutiveRef   = useRef({ id: null, count: 0 });
  const lockoutUntilRef  = useRef(0);
  const lastAddedIdRef   = useRef(null);
  const setListRef       = useRef(null);
  const audioRef         = useRef(new Audio());
  const inputRef         = useRef(null);

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
      mfccAccumRef.current  = null;
      melFBRef.current      = null;
    };
  }, []);

  useEffect(() => {
    if (setListRef.current) setListRef.current.scrollLeft = setListRef.current.scrollWidth;
  }, [setList.length]);

  // Fetch available tags for tag editor autocomplete
  useEffect(() => {
    apiClient.get('/tags/available').then(data => {
      setAvailableTags(data.semantic_tags || []);
      setAvailableVibes(data.vibes || []);
    }).catch(() => {});
  }, []);

  const effectiveAnchor = anchorTrack ?? (setList.length > 0 ? setList[setList.length - 1] : null);
  const showStarters = setList.length === 0 && !anchorTrack;

  const anchorId   = effectiveAnchor?.id ?? effectiveAnchor?.trackid ?? null;
  const setListKey = setList.map(t => t.id).join(',');
  const { starters, onTrajectory, energyUp, stepDown } = useMemo(
    () => suggestDirectional(effectiveAnchor, setList, allNodes),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [anchorId, setListKey, allNodes.length],
  );

  const directionSummary = useMemo(
    () => getSetDirectionSummary(setList),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [setListKey],
  );

  const selectAnchor = useCallback((track) => {
    const id = String(track.id || track.trackid);
    setAnchorTrack(prev => (prev && String(prev.id || prev.trackid) === id) ? null : track);
  }, []);

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

  const addTrack = useCallback((track) => {
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
    sourceRef.current?.disconnect();
    sourceRef.current  = null;
    micStreamRef.current?.getTracks().forEach(t => t.stop());
    micStreamRef.current = null;
    audioCtxRef.current?.close().catch(() => {});
    audioCtxRef.current   = null;
    analyserRef.current   = null;
    melFBRef.current      = null;
    mfccAccumRef.current  = null;
    mfccFrameCountRef.current = 0;
    chromaAccumRef.current  = new Float64Array(12);
    prevSpectrumRef.current = null;
    consecutiveRef.current  = { id: null, count: 0 };
    setListenState('idle');
    setMatchCandidates([]);
  }, []);

  const performMatch = useCallback(() => {
    const frameCount = mfccFrameCountRef.current;
    if (frameCount < 60) return;

    const key = estimateKeyFromChroma(chromaAccumRef.current);
    if (key) setDetectedKey(key);

    const accumMean = Array.from(mfccAccumRef.current).map(v => v / frameCount);
    const liveMFCC  = normaliseMFCC(accumMean);

    const played = new Set(setList.map(t => String(t.id)));
    const scoredWithFP = allNodes
      .filter(n => n.mfccFp && !played.has(String(n.id)))
      .map(n => {
        const score = cosine(liveMFCC, n.mfccFp);
        return { ...n, _matchScore: score };
      })
      .sort((a, b) => b._matchScore - a._matchScore);

    const top5 = scoredWithFP.slice(0, 5);
    setMatchCandidates(top5);

    if (top5[0]?.bpm) setDetectedBPM(Math.round(top5[0].bpm));

    if (top5.length > 0) {
      const best  = top5[0];
      const topId = String(best.id);
      const margin = top5.length > 1 ? (best._matchScore - top5[1]._matchScore) : 1;

      if (best._matchScore >= 0.90 && margin >= 0.02 && topId !== String(lastAddedIdRef.current)) {
        if (topId === consecutiveRef.current.id) {
          consecutiveRef.current.count++;
          if (consecutiveRef.current.count >= 3) {
            addTrack(best);
            setMatchedTrack(best);
            lastAddedIdRef.current  = best.id;
            consecutiveRef.current  = { id: null, count: 0 };
            setTimeout(() => setMatchedTrack(null), 4000);
          }
        } else {
          consecutiveRef.current = { id: topId, count: 1 };
        }
      } else {
        consecutiveRef.current = { id: null, count: 0 };
      }
    }

    mfccAccumRef.current   = new Float64Array(13);
    mfccFrameCountRef.current = 0;
    chromaAccumRef.current = new Float64Array(12);
  }, [allNodes, setList, addTrack]);

  const startListening = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      micStreamRef.current = stream;

      const ctx = new AudioContext();
      audioCtxRef.current = ctx;

      const source = ctx.createMediaStreamSource(stream);
      sourceRef.current = source;

      const analyser = ctx.createAnalyser();
      analyser.fftSize = 4096;
      source.connect(analyser);
      analyserRef.current = analyser;

      melFBRef.current         = buildMelFilterbank(ctx.sampleRate, analyser.fftSize);
      mfccAccumRef.current     = new Float64Array(13);
      mfccFrameCountRef.current= 0;
      chromaAccumRef.current   = new Float64Array(12);
      prevSpectrumRef.current  = null;
      consecutiveRef.current   = { id: null, count: 0 };

      const FRAME_MS         = 50;
      const ANALYSIS_WINDOW  = 15_000;
      const framesPerWindow  = Math.round(ANALYSIS_WINDOW / FRAME_MS);
      let   frameCount       = 0;

      const floatData = new Float32Array(analyser.frequencyBinCount);

      const analyze = () => {
        if (!analyserRef.current || !melFBRef.current) return;

        analyser.getFloatFrequencyData(floatData);

        const mfccFrame = computeMFCC(floatData, melFBRef.current);
        const acc = mfccAccumRef.current;
        for (let i = 0; i < 13; i++) acc[i] += mfccFrame[i];
        mfccFrameCountRef.current++;

        const sr = ctx.sampleRate;
        for (let i = 1; i < floatData.length; i++) {
          const freq = i * sr / analyser.fftSize;
          if (freq < 80 || freq > 4000) continue;
          const power = Math.pow(10, floatData[i] / 20);
          if (power <= 0 || !isFinite(power)) continue;
          const midi = 12 * Math.log2(freq / 440) + 69;
          const bin  = ((Math.round(midi) % 12) + 12) % 12;
          chromaAccumRef.current[bin] += power;
        }

        frameCount++;
        if (frameCount % framesPerWindow === 0) {
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
      paddingTop: 60,
      color: '#e2e8f0',
      fontFamily: "'Inter', system-ui, sans-serif", overflow: 'hidden',
    }}>

      {/* ── Header ────────────────────────────────────────────────────────── */}
      <GlassPanel animate={false} depth={1} style={{
        padding: '14px 24px',
        borderBottom: '1px solid rgba(124,58,237,0.12)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        flexShrink: 0, borderRadius: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: '#e2e8f0' }}>
            {liveSubMode === 'tracker' ? 'SET TRACKER' : 'CRATE BUILDER'}
          </div>

          {/* Mode toggle pill */}
          <div style={{
            display: 'flex', background: 'rgba(8,8,20,0.6)',
            borderRadius: 'var(--radius-pill)', padding: 2,
            border: '1px solid var(--glass-border)',
          }}>
            {['tracker', 'builder'].map(mode => (
              <m.button
                key={mode}
                onClick={() => setLiveSubMode(mode)}
                whileTap={{ scale: 0.95 }}
                style={{
                  padding: '4px 14px', fontSize: 9, fontWeight: 600,
                  fontFamily: "'JetBrains Mono', monospace",
                  letterSpacing: '0.08em', cursor: 'pointer',
                  borderRadius: 'var(--radius-pill)',
                  border: 'none',
                  background: liveSubMode === mode
                    ? 'linear-gradient(135deg, rgba(124,58,237,0.3), rgba(0,212,255,0.15))'
                    : 'transparent',
                  color: liveSubMode === mode ? '#e2e8f0' : '#475569',
                  transition: 'all 200ms ease',
                }}
              >
                {mode === 'tracker' ? 'TRACKER' : 'CRATES'}
              </m.button>
            ))}
          </div>

          {setList.length > 0 && (
            <div style={{ display: 'flex', gap: 20 }}>
              {[
                { label: 'TRACKS', value: String(setList.length).padStart(2, '0'), color: '#00d4ff' },
                { label: 'EST. DURATION', value: `~${sessionMinutes}m`, color: '#a855f7' },
                ...(setList[setList.length - 1]?.bpm ? [{ label: 'CURRENT BPM', value: Math.round(setList[setList.length - 1].bpm), color: '#00d4ff' }] : []),
              ].map(stat => (
                <div key={stat.label} style={{
                  background: 'rgba(8,8,20,0.5)', borderRadius: 'var(--radius-md)',
                  padding: '6px 12px', border: '1px solid var(--glass-border)',
                }}>
                  <div style={{ fontSize: 8, color: '#475569', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.2em', marginBottom: 2 }}>{stat.label}</div>
                  <div style={{ fontSize: 20, fontWeight: 700, color: stat.color, fontFamily: "'JetBrains Mono', monospace" }}>{stat.value}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Upload Library prompt (only if not imported yet) */}
        {!profile?.has_imported_library && (
          <m.button
            onClick={async () => {
              if (!isFileSystemAccessSupported()) {
                setUploadToast({ visible: true, message: 'Local folder access requires Chrome, Edge, or Arc' });
                return;
              }
              try { await pickMusicFolder(); } catch {}
            }}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
            style={{
              padding: '6px 14px',
              background: 'rgba(0,212,255,0.1)',
              border: '1px solid rgba(0,212,255,0.2)',
              borderRadius: 'var(--radius-sm)',
              color: '#00d4ff', fontSize: 10, fontWeight: 600,
              cursor: 'pointer', fontFamily: "'Inter', system-ui, sans-serif",
              letterSpacing: '0.06em',
            }}
          >
            Upload Library
          </m.button>
        )}

        {/* Listen toggle */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <MicLevel analyser={analyserRef.current} active={listenState === 'listening'} />

          {listenState === 'listening' && (
            <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
              {matchCandidates.length > 0 && (
                <>
                  <span style={{ fontSize: 9, color: '#94a3b8', fontFamily: "'JetBrains Mono', monospace", maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {matchCandidates[0].name}
                  </span>
                  <span style={{
                    fontSize: 9, fontFamily: "'JetBrains Mono', monospace",
                    color: matchCandidates[0]._matchScore >= 0.9 ? '#22c55e'
                         : matchCandidates[0]._matchScore >= 0.75 ? '#f59e0b'
                         : '#475569',
                  }}>
                    {Math.round(matchCandidates[0]._matchScore * 100)}%
                  </span>
                  {consecutiveRef.current.count > 0 && (
                    <span style={{ fontSize: 8, color: '#a855f7', fontFamily: "'JetBrains Mono', monospace" }}>
                      {consecutiveRef.current.count}/3
                    </span>
                  )}
                </>
              )}
              {detectedKey && (
                <span style={{ fontSize: 9, color: 'rgba(124,58,237,0.6)', fontFamily: "'JetBrains Mono', monospace" }}>
                  {detectedKey}
                </span>
              )}
            </div>
          )}

          {/* Flash on auto-add */}
          <AnimatePresence>
            {matchedTrack && (
              <m.span
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                style={{
                  fontSize: 9, color: '#22c55e', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.08em',
                  background: 'rgba(34,197,94,0.08)', borderRadius: 'var(--radius-pill)', padding: '3px 10px',
                  border: '1px solid rgba(34,197,94,0.25)',
                }}
              >
                ADDED: {matchedTrack.name || matchedTrack.title}
              </m.span>
            )}
          </AnimatePresence>

          <m.button
            onClick={listenState === 'listening' ? stopListening : startListening}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            style={{
              display: 'flex', alignItems: 'center', gap: 8, padding: '8px 18px',
              borderRadius: 'var(--radius-pill)',
              background: listenState === 'listening' ? 'rgba(239,68,68,0.1)' : 'rgba(124,58,237,0.08)',
              border: `1px solid ${listenState === 'listening' ? 'rgba(239,68,68,0.5)' : 'rgba(124,58,237,0.4)'}`,
              color: listenState === 'listening' ? '#ef4444' : '#a855f7',
              cursor: 'pointer', fontSize: 10, fontWeight: 700, letterSpacing: '0.12em',
              fontFamily: "'Inter', system-ui, sans-serif",
            }}
          >
            <IconMic />
            {listenState === 'listening' ? 'LISTENING' : 'LISTEN'}
          </m.button>
        </div>
      </GlassPanel>

      {liveSubMode === 'builder' ? (
        <CrateBuilder />
      ) : (
      <>
      {/* ── Set list ──────────────────────────────────────────────────────── */}
      <div style={{
        borderBottom: '1px solid rgba(124,58,237,0.1)',
        background: 'rgba(5,5,7,0.4)', flexShrink: 0,
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
              scrollSnapType: 'x proximity',
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
      <div style={{ flex: 1, display: 'flex', gap: 12, overflow: 'hidden', minHeight: 0, padding: '12px 12px 12px 12px' }}>

        {/* Left: Trajectory / Starter suggestions */}
        <GlassPanel animate={false} depth={1} style={{
          width: 400, flexShrink: 0,
          display: 'flex', flexDirection: 'column', overflow: 'hidden',
          borderRadius: 'var(--radius-lg)',
        }}>
          {/* Panel header */}
          <div style={{
            padding: '12px 16px', flexShrink: 0,
            background: 'rgba(8,8,24,0.4)',
            borderBottom: '1px solid rgba(124,58,237,0.08)',
            borderRadius: 'var(--radius-lg) var(--radius-lg) 0 0',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <IconVector />
              <span style={{ fontSize: 10, fontWeight: 700, color: '#94a3b8', letterSpacing: '0.12em', fontFamily: "'JetBrains Mono', monospace" }}>
                {showStarters ? 'STARTER SUGGESTIONS' : 'SET TRAJECTORY'}
              </span>
              {anchorTrack && (
                <m.button
                  onClick={() => setAnchorTrack(null)}
                  whileHover={{ color: '#94a3b8' }}
                  whileTap={{ scale: 0.95 }}
                  style={{ marginLeft: 'auto', background: 'none', border: 'none', color: '#475569', cursor: 'pointer', fontSize: 9, fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.08em', padding: '2px 6px' }}
                >
                  RESET
                </m.button>
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
                <div style={{
                  fontSize: 8, color: '#f59e0b', letterSpacing: '0.2em', fontFamily: "'JetBrains Mono', monospace", marginBottom: 8,
                  background: 'rgba(245,158,11,0.06)', borderRadius: 'var(--radius-pill)', padding: '3px 10px', display: 'inline-block',
                }}>
                  LISTENING — POSSIBLE MATCHES
                </div>
                {matchCandidates.map((t, i) => (
                  <TrackRow key={t.id} track={t} isPlaying={playingId === String(t.id)}
                    onAdd={addTrack} onPlay={togglePlay} onFindSimilar={findSimilar} onEdit={setEditingTrack} onSelectAnchor={selectAnchor} index={i} />
                ))}
              </div>
            )}

            {showStarters && starters.length > 0 && (
              <div style={{ padding: '0 16px' }}>
                {starters.map((t, i) => (
                  <TrackRow key={t.id} track={t} isPlaying={playingId === String(t.id)}
                    isAnchor={anchorTrack && String(anchorTrack.id) === String(t.id)}
                    onAdd={addTrack} onPlay={togglePlay} onFindSimilar={findSimilar}
                    onSelectAnchor={selectAnchor} isStarter index={i} />
                ))}
              </div>
            )}

            {!showStarters && (
              <div>
                {directionSummary && (
                  <div style={{
                    padding: '8px 16px',
                    fontSize: 10,
                    fontFamily: "'JetBrains Mono', monospace",
                    color: 'var(--cyan, #00d4ff)',
                    letterSpacing: '0.08em',
                    opacity: 0.8,
                    borderBottom: '1px solid rgba(0,212,255,0.1)',
                  }}>
                    {directionSummary}
                  </div>
                )}
                {onTrajectory.length > 0 && (
                  <SectionBlock label="ON TRAJECTORY" color="rgba(0,212,255,0.5)">
                    {onTrajectory.map((t, i) => (
                      <TrackRow key={t.id} track={t} isPlaying={playingId === String(t.id)}
                        isAnchor={anchorTrack && String(anchorTrack.id) === String(t.id)}
                        onAdd={addTrack} onPlay={togglePlay} onFindSimilar={findSimilar} onEdit={setEditingTrack} onSelectAnchor={selectAnchor} index={i} />
                    ))}
                  </SectionBlock>
                )}

                {energyUp.length > 0 && (
                  <SectionBlock label="ENERGY UP" color="rgba(239,68,68,0.5)">
                    {energyUp.map((t, i) => (
                      <TrackRow key={t.id} track={t} isPlaying={playingId === String(t.id)}
                        isAnchor={anchorTrack && String(anchorTrack.id) === String(t.id)}
                        onAdd={addTrack} onPlay={togglePlay} onFindSimilar={findSimilar} onEdit={setEditingTrack} onSelectAnchor={selectAnchor} index={i} />
                    ))}
                  </SectionBlock>
                )}

                {stepDown.length > 0 && (
                  <SectionBlock label="STEP DOWN" color="rgba(14,165,233,0.5)">
                    {stepDown.map((t, i) => (
                      <TrackRow key={t.id} track={t} isPlaying={playingId === String(t.id)}
                        isAnchor={anchorTrack && String(anchorTrack.id) === String(t.id)}
                        onAdd={addTrack} onPlay={togglePlay} onFindSimilar={findSimilar} onEdit={setEditingTrack} onSelectAnchor={selectAnchor} index={i} />
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
        </GlassPanel>

        {/* Right: Search */}
        <GlassPanel animate={false} depth={1} style={{
          flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden',
          borderRadius: 'var(--radius-lg)',
        }}>
          <div style={{
            padding: '12px 16px', flexShrink: 0,
            background: 'rgba(8,8,24,0.4)',
            borderBottom: '1px solid rgba(124,58,237,0.08)',
            borderRadius: 'var(--radius-lg) var(--radius-lg) 0 0',
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
          <div style={{ padding: '14px 16px', borderBottom: '1px solid rgba(124,58,237,0.06)', flexShrink: 0 }}>
            <div style={{ display: 'flex', gap: 8 }}>
              <input
                ref={inputRef}
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') runSearch(); }}
                disabled={isBusy}
                placeholder='e.g. "dark minimal 140" or "something to close with"'
                style={{
                  flex: 1, background: 'rgba(8,8,20,0.6)',
                  border: '1px solid rgba(124,58,237,0.18)',
                  borderRadius: 'var(--radius-md)',
                  color: '#e2e8f0', fontSize: 13, padding: '10px 12px',
                  fontFamily: "'JetBrains Mono', monospace",
                  outline: 'none', caretColor: '#7c3aed',
                  opacity: isBusy ? 0.5 : 1, transition: '200ms ease',
                }}
                onFocus={e => { e.target.style.borderColor = 'rgba(124,58,237,0.55)'; e.target.style.boxShadow = '0 0 0 3px rgba(124,58,237,0.08)'; }}
                onBlur={e => { e.target.style.borderColor = 'rgba(124,58,237,0.18)'; e.target.style.boxShadow = 'none'; }}
              />
              <m.button
                onClick={() => runSearch()}
                disabled={isBusy || !query.trim()}
                whileHover={(isBusy || !query.trim()) ? {} : { scale: 1.05 }}
                whileTap={(isBusy || !query.trim()) ? {} : { scale: 0.95 }}
                style={{
                  padding: '0 16px', borderRadius: 'var(--radius-sm)',
                  background: isBusy || !query.trim() ? 'rgba(8,8,20,0.5)' : 'linear-gradient(135deg, rgba(124,58,237,0.5), rgba(0,212,255,0.3))',
                  border: `1px solid ${isBusy || !query.trim() ? 'rgba(124,58,237,0.1)' : 'rgba(124,58,237,0.45)'}`,
                  color: isBusy || !query.trim() ? '#2d3748' : '#e2e8f0',
                  cursor: isBusy || !query.trim() ? 'default' : 'pointer',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}
              >
                {isBusy
                  ? <div style={{ width: 14, height: 14, border: '1.5px solid rgba(124,58,237,0.2)', borderTop: '1.5px solid #7c3aed', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
                  : <IconSend />}
              </m.button>
            </div>
          </div>

          <AnimatePresence>
            {searchStatus === 'error' && (
              <m.div
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                style={{ margin: '10px 16px', padding: '10px 12px', background: 'rgba(16,5,5,0.8)', border: '1px solid rgba(239,68,68,0.25)', borderRadius: 'var(--radius-md)', fontSize: 11, color: 'rgba(239,68,68,0.8)', flexShrink: 0 }}
              >
                {searchError}
              </m.div>
            )}
          </AnimatePresence>

          {/* Results */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '10px 0' }}>
            <AnimatePresence>
              {searchStatus === 'searching' && (
                <m.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  style={{ padding: '16px', fontSize: 10, color: 'rgba(124,58,237,0.6)', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.08em', display: 'flex', gap: 8, alignItems: 'center' }}
                >
                  <div style={{ width: 12, height: 12, border: '1.5px solid rgba(124,58,237,0.15)', borderTop: '1.5px solid #7c3aed', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
                  SEARCHING LIBRARY...
                </m.div>
              )}
            </AnimatePresence>

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
                    onEdit={setEditingTrack}
                    index={i}
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
        </GlassPanel>
      </div>

      {/* Tag editor modal */}
      <AnimatePresence>
        {editingTrack && (
          <TagEditor
            track={editingTrack}
            onClose={() => setEditingTrack(null)}
            onSave={() => {}}
            availableTags={availableTags}
            availableVibes={availableVibes}
          />
        )}
      </AnimatePresence>
      </>
      )}

      <Toast
        message={uploadToast.message}
        visible={uploadToast.visible}
        onDismiss={() => setUploadToast({ visible: false, message: '' })}
      />
    </div>
  );
}
