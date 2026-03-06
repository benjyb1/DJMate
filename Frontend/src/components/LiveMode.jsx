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

function suggestNextTracks(setList, allNodes) {
  const played = new Set(setList.map(t => String(t.id)));
  const candidates = allNodes.filter(n => !played.has(String(n.id)));
  if (candidates.length === 0) return [];

  if (setList.length === 0) {
    return candidates.sort(() => Math.random() - 0.5).slice(0, 8);
  }

  const last = setList[setList.length - 1];

  if (setList.length === 1) {
    // BPM-proximity only
    return candidates
      .map(n => ({ ...n, _score: -Math.abs((n.bpm || 130) - (last.bpm || 130)) }))
      .sort((a, b) => b._score - a._score)
      .slice(0, 8);
  }

  const dir = getTrajectoryVector(setList);

  return candidates
    .map(n => {
      const dx = (n.x || 0) - (last.x || 0);
      const dy = (n.y || 0) - (last.y || 0);
      const dz = (n.z || 0) - (last.z || 0);
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;

      // Alignment with trajectory direction (0→1)
      const dot = dir ? (dx * dir.x + dy * dir.y + dz * dir.z) / dist : 0;
      const dirScore = (dot + 1) / 2; // normalise -1..1 → 0..1

      // Proximity in embedding space
      const proxScore = 1 / (1 + dist * 0.0008);

      // BPM continuity (±10 BPM preferred)
      const bpmDiff = Math.abs((n.bpm || 130) - (last.bpm || 130));
      const bpmScore = Math.max(0, 1 - bpmDiff / 25);

      const _score = dirScore * 0.5 + proxScore * 0.3 + bpmScore * 0.2;
      return { ...n, _score, _reason: getReason(n, last, dir, dist, dot) };
    })
    .sort((a, b) => b._score - a._score)
    .slice(0, 8);
}

function getReason(n, last, dir, dist, dot) {
  if (!n.bpm) return 'similar vibe';
  const bpmDiff = (n.bpm || 130) - (last.bpm || 130);
  if (Math.abs(bpmDiff) < 3) return `${Math.round(n.bpm)} BPM · on trajectory`;
  if (bpmDiff > 0) return `+${Math.round(bpmDiff)} BPM · ahead`;
  return `${Math.round(bpmDiff)} BPM · step down`;
}

// ── SVG Icons ──────────────────────────────────────────────────────────────
const IconMic = ({ active }) => (
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

// ── Mini album art card ────────────────────────────────────────────────────
function SetTrackArt({ track, index, isLatest, onRemove }) {
  const [artUrl, setArtUrl] = useState(track.albumArt || makeSupabaseCoverUrl(track.artist, track.name));
  const [artErr, setArtErr] = useState(false);

  const handleErr = () => {
    if (artUrl && artUrl.includes('supabase')) {
      // Fall through to iTunes
      const term = encodeURIComponent(`${track.artist} ${track.name}`);
      fetch(`https://itunes.apple.com/search?term=${term}&entity=song&limit=1`)
        .then(r => r.json())
        .then(data => {
          const raw = data.results?.[0]?.artworkUrl100;
          if (raw) setArtUrl(raw.replace('100x100bb', '300x300bb'));
          else setArtErr(true);
        })
        .catch(() => setArtErr(true));
    } else {
      setArtErr(true);
    }
  };

  // Generative fallback
  let hash = 0;
  const str = (track.name || '') + (track.artist || '');
  for (let i = 0; i < str.length; i++) { hash = ((hash << 5) - hash) + str.charCodeAt(i); hash |= 0; }
  const hue = 200 + (Math.abs(hash) % 100);
  const initials = (track.name || '?').replace(/[^a-zA-Z0-9]/g, '').slice(0, 2).toUpperCase() || '??';

  return (
    <div style={{ position: 'relative', flexShrink: 0 }}>
      {/* Track number */}
      <div style={{
        position: 'absolute', top: -16, left: 0, right: 0, textAlign: 'center',
        fontSize: 8, color: '#2d3748', fontFamily: "'JetBrains Mono', monospace",
        letterSpacing: '0.1em',
      }}>
        {String(index + 1).padStart(2, '0')}
      </div>

      {/* Art */}
      <div style={{
        width: 80, height: 80,
        border: isLatest ? '2px solid #00d4ff' : '1px solid rgba(124,58,237,0.25)',
        boxShadow: isLatest ? '0 0 16px rgba(0,212,255,0.3)' : 'none',
        position: 'relative', overflow: 'hidden',
        transition: '200ms ease',
      }}>
        {artUrl && !artErr ? (
          <img src={artUrl} alt={track.name} onError={handleErr}
            style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />
        ) : (
          <div style={{
            width: '100%', height: '100%',
            background: `linear-gradient(135deg, hsl(${hue},50%,8%), hsl(${hue},70%,5%))`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontFamily: "'JetBrains Mono', monospace", fontSize: 18, fontWeight: 700,
            color: `hsla(${hue},80%,70%,0.9)`,
          }}>
            {initials}
          </div>
        )}

        {/* Latest indicator */}
        {isLatest && (
          <div style={{
            position: 'absolute', bottom: 0, left: 0, right: 0,
            background: 'rgba(0,212,255,0.15)',
            borderTop: '1px solid rgba(0,212,255,0.4)',
            fontSize: 7, color: '#00d4ff', textAlign: 'center',
            padding: '2px 0', letterSpacing: '0.15em',
            fontFamily: "'JetBrains Mono', monospace",
          }}>PLAYING</div>
        )}
      </div>

      {/* Remove button */}
      <button
        onClick={() => onRemove(index)}
        aria-label="Remove from set"
        style={{
          position: 'absolute', top: -6, right: -6,
          width: 18, height: 18,
          background: 'rgba(5,5,7,0.9)',
          border: '1px solid rgba(239,68,68,0.4)',
          color: 'rgba(239,68,68,0.6)',
          cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
          transition: '150ms ease',
        }}
        onMouseEnter={e => { e.currentTarget.style.background = 'rgba(239,68,68,0.15)'; e.currentTarget.style.color = '#ef4444'; }}
        onMouseLeave={e => { e.currentTarget.style.background = 'rgba(5,5,7,0.9)'; e.currentTarget.style.color = 'rgba(239,68,68,0.6)'; }}
      >
        <IconX />
      </button>

      {/* Track info tooltip */}
      <div style={{
        marginTop: 6, width: 80, overflow: 'hidden',
      }}>
        <div style={{
          fontSize: 9, color: '#94a3b8', whiteSpace: 'nowrap',
          overflow: 'hidden', textOverflow: 'ellipsis', fontWeight: 600,
        }}>{track.name}</div>
        <div style={{
          fontSize: 8, color: '#475569', whiteSpace: 'nowrap',
          overflow: 'hidden', textOverflow: 'ellipsis',
        }}>{track.artist}</div>
        {track.bpm && (
          <div style={{ fontSize: 8, color: '#2d3748', fontFamily: "'JetBrains Mono', monospace" }}>
            {Math.round(track.bpm)} BPM
          </div>
        )}
      </div>
    </div>
  );
}

// ── Suggestion card ────────────────────────────────────────────────────────
function SuggestionCard({ track, onAdd }) {
  const [artUrl, setArtUrl] = useState(track.albumArt || makeSupabaseCoverUrl(track.artist, track.name));
  const [artErr, setArtErr] = useState(false);

  const handleErr = () => {
    if (artUrl?.includes('supabase')) {
      const term = encodeURIComponent(`${track.artist} ${track.name}`);
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

  let hash = 0;
  const str = (track.name || '') + (track.artist || '');
  for (let i = 0; i < str.length; i++) { hash = ((hash << 5) - hash) + str.charCodeAt(i); hash |= 0; }
  const hue = 200 + (Math.abs(hash) % 100);
  const initials = (track.name || '?').replace(/[^a-zA-Z0-9]/g, '').slice(0, 2).toUpperCase() || '??';

  return (
    <div
      style={{
        display: 'flex', gap: 10, alignItems: 'center',
        padding: '10px 14px',
        background: 'rgba(8,8,20,0.5)',
        border: '1px solid rgba(124,58,237,0.12)',
        marginBottom: 4,
        transition: '150ms ease',
        cursor: 'default',
      }}
      onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(124,58,237,0.35)'; e.currentTarget.style.background = 'rgba(12,12,28,0.7)'; }}
      onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(124,58,237,0.12)'; e.currentTarget.style.background = 'rgba(8,8,20,0.5)'; }}
    >
      {/* Art */}
      <div style={{ width: 44, height: 44, flexShrink: 0, overflow: 'hidden', border: '1px solid rgba(124,58,237,0.2)' }}>
        {artUrl && !artErr ? (
          <img src={artUrl} alt={track.name} onError={handleErr}
            style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />
        ) : (
          <div style={{
            width: '100%', height: '100%',
            background: `linear-gradient(135deg, hsl(${hue},50%,8%), hsl(${hue},70%,5%))`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontFamily: "'JetBrains Mono', monospace", fontSize: 13, fontWeight: 700,
            color: `hsla(${hue},80%,70%,0.9)`,
          }}>
            {initials}
          </div>
        )}
      </div>

      {/* Info */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: '#e2e8f0', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {track.name}
        </div>
        <div style={{ fontSize: 10, color: '#94a3b8', marginBottom: 3 }}>{track.artist}</div>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
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
        </div>
      </div>

      {/* Add button */}
      <button
        onClick={() => onAdd(track)}
        aria-label={`Add ${track.name} to set`}
        style={{
          flexShrink: 0, width: 30, height: 30,
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
        const threshold = (i / 12) * 100;
        const lit = level > threshold;
        return (
          <div key={i} style={{
            width: 3, height: 6 + i * 1.2,
            background: lit
              ? (i > 9 ? '#ef4444' : i > 7 ? '#f59e0b' : '#00d4ff')
              : 'rgba(255,255,255,0.08)',
            transition: '80ms ease',
          }} />
        );
      })}
    </div>
  );
}

// ── Main LiveMode ──────────────────────────────────────────────────────────
export default function LiveMode({ setList, setSetList, allNodes }) {
  const [isListening, setIsListening]   = useState(false);
  const [micStatus, setMicStatus]       = useState(''); // 'detecting' | 'found' | 'error'
  const [micCandidates, setMicCandidates] = useState([]);
  const [query, setQuery]               = useState('');
  const [searchStatus, setSearchStatus] = useState('idle');
  const [searchResults, setSearchResults] = useState([]);
  const [searchError, setSearchError]   = useState('');

  const micStreamRef  = useRef(null);
  const audioCtxRef   = useRef(null);
  const analyserRef   = useRef(null);
  const recorderRef   = useRef(null);
  const chunksRef     = useRef([]);
  const setListRef    = useRef(null);

  const inputRef = useRef(null);

  // Scroll set list to end when new track added
  useEffect(() => {
    if (setListRef.current) {
      setListRef.current.scrollLeft = setListRef.current.scrollWidth;
    }
  }, [setList.length]);

  // Suggestions from trajectory
  const suggestions = setList.length > 0 ? suggestNextTracks(setList, allNodes) : [];

  // ── Add / remove ──────────────────────────────────────────────────────────
  const addTrack = useCallback((track) => {
    setSetList(prev => {
      if (prev.some(t => String(t.id) === String(track.id))) return prev;
      return [...prev, track];
    });
  }, [setSetList]);

  const removeTrack = useCallback((index) => {
    setSetList(prev => prev.filter((_, i) => i !== index));
  }, [setSetList]);

  // ── Mic recognition ──────────────────────────────────────────────────────
  const startListening = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      micStreamRef.current = stream;

      const ctx = new AudioContext();
      audioCtxRef.current = ctx;

      const source  = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      analyserRef.current = analyser;

      // Record audio for BPM analysis
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      recorder.onstop = () => analyseMicAudio();
      recorderRef.current = recorder;
      recorder.start();

      setIsListening(true);
      setMicStatus('detecting');
      setMicCandidates([]);

      // Stop after 12 seconds
      setTimeout(() => {
        if (recorderRef.current?.state === 'recording') {
          recorderRef.current.stop();
        }
      }, 12000);

    } catch (err) {
      setMicStatus('error');
      console.error('Mic error:', err);
    }
  }, []);

  const stopListening = useCallback(() => {
    recorderRef.current?.stop();
    micStreamRef.current?.getTracks().forEach(t => t.stop());
    audioCtxRef.current?.close();
    micStreamRef.current = null;
    audioCtxRef.current = null;
    analyserRef.current = null;
    setIsListening(false);
    setMicStatus('');
  }, []);

  const analyseMicAudio = useCallback(() => {
    // Simple BPM estimation via onset/energy detection
    // We decode the recorded audio offline and look for energy peaks
    const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
    const reader = new FileReader();
    reader.onloadend = async () => {
      try {
        const ctx = new OfflineAudioContext(1, 44100 * 12, 44100);
        const buf = await ctx.decodeAudioData(reader.result);
        const data = buf.getChannelData(0);

        // Simple energy-based onset detection
        const windowSize = Math.floor(44100 * 0.02); // 20ms windows
        const energies = [];
        for (let i = 0; i < data.length - windowSize; i += windowSize) {
          let e = 0;
          for (let j = i; j < i + windowSize; j++) e += data[j] * data[j];
          energies.push(e / windowSize);
        }

        // Find peaks (energy > local mean * threshold)
        const mean = energies.reduce((a, b) => a + b, 0) / energies.length;
        const peaks = [];
        for (let i = 1; i < energies.length - 1; i++) {
          if (energies[i] > energies[i - 1] && energies[i] > energies[i + 1] && energies[i] > mean * 2) {
            if (peaks.length === 0 || i - peaks[peaks.length - 1] > 10) {
              peaks.push(i);
            }
          }
        }

        // Estimate BPM from peak intervals
        if (peaks.length < 3) {
          setMicStatus('no-detect');
          stopListening();
          return;
        }
        const intervals = peaks.slice(1).map((p, i) => p - peaks[i]);
        const medianInterval = intervals.sort((a, b) => a - b)[Math.floor(intervals.length / 2)];
        const bpm = Math.round(60 / (medianInterval * 0.02)); // convert windows to seconds

        // Find library tracks with matching BPM
        const bpmClamped = Math.max(80, Math.min(200, bpm));
        const matches = allNodes
          .filter(n => n.bpm && Math.abs(n.bpm - bpmClamped) < 8)
          .sort((a, b) => Math.abs(a.bpm - bpmClamped) - Math.abs(b.bpm - bpmClamped))
          .slice(0, 5);

        setMicCandidates(matches.length > 0 ? matches : []);
        setMicStatus(matches.length > 0 ? 'found' : 'no-detect');
      } catch (e) {
        setMicStatus('error');
      }
      stopListening();
    };
    reader.readAsArrayBuffer(blob);
  }, [allNodes, stopListening]);

  // ── LLM search ───────────────────────────────────────────────────────────
  const runSearch = useCallback(async (q) => {
    const text = (q || query).trim();
    if (!text) return;
    setSearchStatus('searching'); setSearchResults([]); setSearchError('');
    try {
      const { params } = await apiClient.post('/chat/interpret', { query: text });
      const result = await apiClient.post('/chat/search', { params });
      setSearchResults(result.tracks || []);
      setSearchStatus('done');
    } catch (err) {
      setSearchError(err.message || 'Search failed');
      setSearchStatus('error');
    }
  }, [query]);

  const isBusy = searchStatus === 'searching';

  // ── Format session duration ───────────────────────────────────────────────
  const sessionMinutes = setList.length > 0 ? Math.round(setList.length * 6.5) : 0;

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', height: '100%',
      background: '#050507', color: '#e2e8f0',
      fontFamily: "'Inter', system-ui, sans-serif",
      overflow: 'hidden',
    }}>
      {/* ── Set header ────────────────────────────────────────────────── */}
      <div style={{
        padding: '14px 24px',
        borderBottom: '1px solid rgba(124,58,237,0.15)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        background: 'rgba(8,8,20,0.6)', flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
          <div>
            <div style={{ fontSize: 8, letterSpacing: '0.3em', color: 'rgba(124,58,237,0.6)', fontFamily: "'JetBrains Mono', monospace", marginBottom: 3 }}>
              LIVE SESSION
            </div>
            <div style={{ fontSize: 18, fontWeight: 700, color: '#e2e8f0' }}>
              SET TRACKER
            </div>
          </div>

          {setList.length > 0 && (
            <div style={{ display: 'flex', gap: 20, alignItems: 'center' }}>
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

        {/* Mic toggle */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <MicLevel analyser={analyserRef.current} active={isListening} />

          {micStatus === 'detecting' && (
            <span style={{ fontSize: 9, color: '#f59e0b', fontFamily: "'JetBrains Mono', monospace', letterSpacing: '0.1em'", animation: 'blink 1s ease-in-out infinite' }}>
              ANALYSING...
            </span>
          )}
          {micStatus === 'no-detect' && (
            <span style={{ fontSize: 9, color: '#475569', fontFamily: "'JetBrains Mono', monospace" }}>NO MATCH</span>
          )}

          <button
            onClick={isListening ? stopListening : startListening}
            aria-label={isListening ? 'Stop listening' : 'Start BPM detection'}
            style={{
              display: 'flex', alignItems: 'center', gap: 8,
              padding: '8px 16px',
              background: isListening ? 'rgba(239,68,68,0.1)' : 'rgba(124,58,237,0.08)',
              border: `1px solid ${isListening ? 'rgba(239,68,68,0.5)' : 'rgba(124,58,237,0.4)'}`,
              color: isListening ? '#ef4444' : '#a855f7',
              cursor: 'pointer', fontSize: 10, fontWeight: 700, letterSpacing: '0.12em',
              fontFamily: "'Inter', system-ui, sans-serif",
              transition: '150ms ease',
            }}
            onMouseEnter={e => { e.currentTarget.style.background = isListening ? 'rgba(239,68,68,0.18)' : 'rgba(124,58,237,0.15)'; }}
            onMouseLeave={e => { e.currentTarget.style.background = isListening ? 'rgba(239,68,68,0.1)' : 'rgba(124,58,237,0.08)'; }}
          >
            <IconMic active={isListening} />
            {isListening ? 'STOP LISTENING' : 'BPM DETECT'}
          </button>
        </div>
      </div>

      {/* ── Set list (horizontal scroll) ──────────────────────────────── */}
      <div style={{
        borderBottom: '1px solid rgba(124,58,237,0.12)',
        background: 'rgba(5,5,7,0.8)',
        flexShrink: 0,
      }}>
        {setList.length === 0 ? (
          <div style={{
            padding: '32px 24px', textAlign: 'center',
            color: '#2d3748', fontSize: 11, fontFamily: "'JetBrains Mono', monospace",
            letterSpacing: '0.15em',
          }}>
            ADD TRACKS BELOW TO START TRACKING YOUR SET
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
                onRemove={removeTrack}
              />
            ))}
          </div>
        )}
      </div>

      {/* ── Main content: Suggestions + Search ───────────────────────── */}
      <div style={{ flex: 1, display: 'flex', gap: 0, overflow: 'hidden', minHeight: 0 }}>

        {/* Left: Trajectory suggestions */}
        <div style={{
          width: 420, flexShrink: 0,
          borderRight: '1px solid rgba(124,58,237,0.12)',
          display: 'flex', flexDirection: 'column',
          overflow: 'hidden',
        }}>
          <div style={{
            padding: '12px 16px', flexShrink: 0,
            background: 'rgba(8,8,24,0.5)',
            borderBottom: '1px solid rgba(124,58,237,0.1)',
            display: 'flex', alignItems: 'center', gap: 8,
          }}>
            <IconVector />
            <span style={{ fontSize: 10, fontWeight: 700, color: '#94a3b8', letterSpacing: '0.12em', fontFamily: "'JetBrains Mono', monospace" }}>
              SET TRAJECTORY
            </span>
            {setList.length >= 2 && (
              <span style={{ fontSize: 8, color: '#2d3748', fontFamily: "'JetBrains Mono', monospace", marginLeft: 4 }}>
                · VECTOR PROJECTION ACTIVE
              </span>
            )}
          </div>

          <div style={{ flex: 1, overflowY: 'auto', padding: '12px 0' }}>
            {/* Mic candidates */}
            {micCandidates.length > 0 && (
              <div style={{ padding: '0 16px 12px', borderBottom: '1px solid rgba(124,58,237,0.1)', marginBottom: 12 }}>
                <div style={{ fontSize: 8, color: '#f59e0b', letterSpacing: '0.2em', fontFamily: "'JetBrains Mono', monospace", marginBottom: 8 }}>
                  BPM MATCHES — SELECT PLAYING TRACK
                </div>
                {micCandidates.map(t => (
                  <SuggestionCard key={t.id} track={t} onAdd={addTrack} />
                ))}
              </div>
            )}

            {suggestions.length === 0 && micCandidates.length === 0 && (
              <div style={{
                padding: '24px 16px', color: '#2d3748', fontSize: 10,
                fontFamily: "'JetBrains Mono', monospace", lineHeight: 1.8, letterSpacing: '0.08em',
              }}>
                <div style={{ marginBottom: 8 }}>NO TRAJECTORY YET.</div>
                <div>ADD 2+ TRACKS TO ACTIVATE</div>
                <div>VECTOR-BASED SUGGESTIONS.</div>
              </div>
            )}

            <div style={{ padding: '0 16px' }}>
              {suggestions.map(t => (
                <SuggestionCard key={t.id} track={t} onAdd={addTrack} />
              ))}
            </div>
          </div>
        </div>

        {/* Right: LLM search */}
        <div style={{
          flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden',
        }}>
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

          {/* Input */}
          <div style={{ padding: '14px 16px', borderBottom: '1px solid rgba(124,58,237,0.08)', flexShrink: 0 }}>
            <div style={{ display: 'flex', gap: 8 }}>
              <input
                ref={inputRef}
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') runSearch(); }}
                disabled={isBusy}
                placeholder='Search your library — e.g. "dark minimal 140" or "something to close with"'
                style={{
                  flex: 1,
                  background: 'rgba(8,8,20,0.8)',
                  border: '1px solid rgba(124,58,237,0.18)',
                  color: '#e2e8f0', fontSize: 13,
                  padding: '10px 12px',
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
                  padding: '0 16px',
                  background: isBusy || !query.trim() ? 'rgba(8,8,20,0.5)' : 'linear-gradient(135deg, rgba(124,58,237,0.5), rgba(0,212,255,0.3))',
                  border: `1px solid ${isBusy || !query.trim() ? 'rgba(124,58,237,0.1)' : 'rgba(124,58,237,0.45)'}`,
                  color: isBusy || !query.trim() ? '#2d3748' : '#e2e8f0',
                  cursor: isBusy || !query.trim() ? 'default' : 'pointer',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  transition: '150ms ease', borderRadius: 0,
                }}
              >
                {isBusy
                  ? <div style={{ width: 14, height: 14, border: '1.5px solid rgba(124,58,237,0.2)', borderTop: '1.5px solid #7c3aed', animation: 'spin 0.8s linear infinite' }} />
                  : <IconSend />}
              </button>
            </div>
          </div>

          {/* Error */}
          {searchStatus === 'error' && (
            <div style={{ margin: '10px 16px', padding: '10px 12px', background: 'rgba(16,5,5,0.8)', border: '1px solid rgba(239,68,68,0.25)', fontSize: 11, color: 'rgba(239,68,68,0.8)', flexShrink: 0 }}>
              {searchError}
            </div>
          )}

          {/* Results */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '12px 0' }}>
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
                  <div
                    key={t.trackid || i}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 10,
                      padding: '8px 10px', marginBottom: 3,
                      background: 'rgba(8,8,20,0.5)',
                      border: '1px solid rgba(124,58,237,0.1)',
                      transition: '150ms ease',
                    }}
                    onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(124,58,237,0.3)'; e.currentTarget.style.background = 'rgba(12,12,28,0.7)'; }}
                    onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(124,58,237,0.1)'; e.currentTarget.style.background = 'rgba(8,8,20,0.5)'; }}
                  >
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 12, fontWeight: 700, color: '#e2e8f0', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {t.title || 'Unknown'}
                      </div>
                      <div style={{ fontSize: 10, color: '#94a3b8' }}>{t.artist}</div>
                      <div style={{ fontSize: 9, color: '#475569', fontFamily: "'JetBrains Mono', monospace', letterSpacing: '0.05em'" }}>
                        {t.bpm ? `${Math.round(t.bpm)} BPM` : ''}{t.key ? ` · ${t.key}` : ''}
                      </div>
                    </div>
                    <button
                      onClick={() => {
                        const node = allNodes.find(n => String(n.id) === String(t.trackid));
                        if (node) addTrack(node);
                      }}
                      style={{
                        flexShrink: 0, width: 28, height: 28,
                        background: 'rgba(0,212,255,0.06)',
                        border: '1px solid rgba(0,212,255,0.3)',
                        color: '#00d4ff', cursor: 'pointer',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        transition: '150ms ease', borderRadius: 0,
                      }}
                      onMouseEnter={e => { e.currentTarget.style.background = 'rgba(0,212,255,0.15)'; }}
                      onMouseLeave={e => { e.currentTarget.style.background = 'rgba(0,212,255,0.06)'; }}
                    >
                      <IconPlus />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
