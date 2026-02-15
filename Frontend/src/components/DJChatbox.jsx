// src/components/DJChatbox.jsx
// Replicates the "Describe what you want" tab from streamSimilar.py.
// Uses apiClient (/chat/interpret → /chat/search) for all backend calls.
//
// Props:
//   selectedTrack — node object from the 3D graph (optional context)
//   trackCount    — total number of tracks loaded

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { apiClient } from '../api/apiClient';

// ── Utility: direction badge (mirrors describe_direction() in streamSimilar.py) ─
function describeDirection(source, candidate) {
  const clues = [];

  const sE = parseFloat(source?.energy  ?? 0.5);
  const cE = parseFloat(candidate?.energy ?? 0.5);
  if (cE - sE >  0.15) clues.push({ label: 'higher energy', color: '#ff4444' });
  if (cE - sE < -0.15) clues.push({ label: 'lower energy',  color: '#4488ff' });

  const sB = source?.bpm, cB = candidate?.bpm;
  if (sB && cB) {
    const diff = parseFloat(cB) - parseFloat(sB);
    if (diff >  5) clues.push({ label: `+${Math.round(diff)} BPM faster`, color: '#ffaa00' });
    if (diff < -5) clues.push({ label: `${Math.round(diff)} BPM slower`, color: '#aaaaff' });
  }

  const srcTags  = new Set(source?.semantic_tags    || []);
  const cndTags  = new Set(candidate?.semantic_tags || []);
  const srcVibes = new Set(source?.vibe_descriptors || []);
  const cndVibes = new Set(candidate?.vibe_descriptors || []);

  const newTags  = [...cndTags ].filter(t => !srcTags.has(t));
  const newVibes = [...cndVibes].filter(v => !srcVibes.has(v));

  if (newTags.length)  clues.push({ label: `more ${newTags[0]}`,  color: '#00ff88' });
  if (newVibes.length) clues.push({ label: newVibes[0],           color: '#cc88ff' });

  if (!clues.length) return { label: 'similar vibe', color: '#555555' };
  const moreClue = clues.find(c => c.label.startsWith('more '));
  return moreClue || clues[0];
}

// ── SimBar ─────────────────────────────────────────────────────────────────────
function SimBar({ score }) {
  const pct   = Math.round(score * 100);
  const color = score > 0.7 ? '#00ff88' : score > 0.4 ? '#ffaa00' : '#ff4444';
  return (
    <div style={{ marginTop: 6 }}>
      <div style={{ height: 3, borderRadius: 2, width: `${pct}%`, background: color, transition: 'width 0.4s ease' }} />
      <div style={{ fontSize: 10, color: '#555', marginTop: 2 }}>{pct}% match</div>
    </div>
  );
}

// ── TrackCard (mirrors result-card in streamSimilar.py) ────────────────────────
function TrackCard({ rank, track, score, source }) {
  const [audioError, setAudioError] = useState(false);
  const direction = source ? describeDirection(source, track) : null;

  const metaParts = [];
  if (track.bpm)    metaParts.push(`${Math.round(track.bpm)} BPM`);
  if (track.key)    metaParts.push(track.key);
  if (track.energy != null) metaParts.push(`energy ${parseFloat(track.energy).toFixed(2)}`);

  return (
    <div
      style={{
        display: 'grid', gridTemplateColumns: '1fr auto',
        gap: 10, alignItems: 'center',
        background: '#0a0a0a',
        border: '1px solid #1e1e1e',
        borderRadius: 10,
        padding: '12px 14px',
        marginBottom: 4,
        transition: 'border-color 0.15s',
      }}
      onMouseEnter={e => e.currentTarget.style.borderColor = '#333'}
      onMouseLeave={e => e.currentTarget.style.borderColor = '#1e1e1e'}
    >
      {/* ── Left: info ─────────────────────────────────────────────────── */}
      <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
        <div style={{ fontSize: 26, color: '#222', fontWeight: 900, lineHeight: 1, minWidth: 28, fontFamily: 'monospace' }}>
          {rank}
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 14, fontWeight: 700, color: '#fff', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {track.title || 'Unknown'}
          </div>
          <div style={{ fontSize: 11, color: '#888', marginTop: 2 }}>
            {track.artist || 'Unknown'}
          </div>
          {metaParts.length > 0 && (
            <div style={{ fontSize: 10, color: '#555', marginTop: 5 }}>
              {metaParts.join(' · ')}
            </div>
          )}

          {/* Tags */}
          {track.semantic_tags?.length > 0 && (
            <div style={{ marginTop: 5, display: 'flex', flexWrap: 'wrap', gap: 3 }}>
              {track.semantic_tags.slice(0, 6).map(tag => (
                <span key={tag} style={{
                  background: '#0d0d2a', color: '#5555bb',
                  borderRadius: 10, padding: '2px 7px', fontSize: 10,
                  border: '1px solid #1a1a3a',
                }}>{tag}</span>
              ))}
            </div>
          )}

          {/* Badges */}
          <div style={{ display: 'flex', gap: 5, marginTop: 5, flexWrap: 'wrap', alignItems: 'center' }}>
            {track.inferred && (
              <span style={{
                background: '#111100', color: '#aaaa00', border: '1px solid #333300',
                borderRadius: 8, padding: '2px 7px', fontSize: 10,
              }}>🔮 inferred</span>
            )}
            {direction && (
              <span style={{
                background: `${direction.color}18`, color: direction.color,
                border: `1px solid ${direction.color}55`,
                borderRadius: 10, padding: '2px 8px', fontSize: 10, fontWeight: 600,
              }}>↗ {direction.label}</span>
            )}
          </div>

          <SimBar score={score} />
        </div>
      </div>

      {/* ── Right: audio ───────────────────────────────────────────────── */}
      <div style={{ minWidth: 140 }}>
        {track.filepath && !audioError ? (
          <audio
            controls
            src={track.filepath}
            onError={() => setAudioError(true)}
            style={{ width: 140, height: 32, accentColor: '#00ffff' }}
          />
        ) : (
          <div style={{ fontSize: 10, color: '#333', textAlign: 'center' }}>
            {track.filepath ? '⚠ file not found' : '—'}
          </div>
        )}
      </div>
    </div>
  );
}

// ── SearchResults ──────────────────────────────────────────────────────────────
function SearchResults({ result, source }) {
  if (!result) return null;

  return (
    <div style={{ marginTop: 12 }}>

      {/* Relaxation notice */}
      {result.relaxation_step > 0 && (
        <div style={{
          background: '#110a00', border: '1px solid #443300',
          borderRadius: 8, padding: '7px 12px', marginBottom: 10,
          fontSize: 11, color: '#cc9900',
        }}>
          ⚡ Widened search ({result.relaxation_label})
        </div>
      )}
      {result.inferred_count > 0 && (
        <div style={{
          background: '#110a00', border: '1px solid #443300',
          borderRadius: 8, padding: '7px 12px', marginBottom: 10,
          fontSize: 11, color: '#cc9900',
        }}>
          🔮 {result.inferred_count} track(s) inferred by sound similarity
        </div>
      )}

      {/* Summary line */}
      <div style={{ fontSize: 11, color: '#666', marginBottom: 10 }}>
        <span style={{ color: '#aaa', fontWeight: 600 }}>{result.tracks.length} tracks</span>
        {result.reasoning && (
          <span> · <em style={{ color: '#888' }}>{result.reasoning}</em></span>
        )}
        {result.confidence > 0 && (
          <span> · confidence <span style={{ color: '#00ffcc' }}>{Math.round(result.confidence * 100)}%</span></span>
        )}
        {result.model_used && (
          <span> · <code style={{ background: '#0a0a0a', padding: '1px 5px', borderRadius: 3, fontSize: 10, color: '#555' }}>{result.model_used}</code></span>
        )}
      </div>

      {/* Track cards */}
      {result.tracks.map((track, i) => (
        <TrackCard
          key={track.trackid || i}
          rank={i + 1}
          track={track}
          score={track.relevance_score ?? 0.5}
          source={source}
        />
      ))}
    </div>
  );
}

// ── Suggested quick prompts ────────────────────────────────────────────────────
const QUICK_PROMPTS = [
  'fast kicking house',
  'dark minimal techno 140',
  'soulful deep opening set',
  'peak time euphoric',
];

// ── Main DJChatbox ─────────────────────────────────────────────────────────────
export default function DJChatbox({ selectedTrack, trackCount }) {
  const [isOpen,      setIsOpen]      = useState(false);
  const [isMinimised, setIsMinimised] = useState(false);
  const [query,       setQuery]       = useState('');
  const [status,      setStatus]      = useState('idle'); // idle | interpreting | searching | done | error
  const [result,      setResult]      = useState(null);
  const [errorMsg,    setErrorMsg]    = useState('');
  const [history,     setHistory]     = useState([]);
  const inputRef   = useRef(null);
  const resultsRef = useRef(null);

  // Focus input when opened
  useEffect(() => {
    if (isOpen && !isMinimised) setTimeout(() => inputRef.current?.focus(), 120);
  }, [isOpen, isMinimised]);

  // Scroll results into view after they load
  useEffect(() => {
    if (result) resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [result]);

  // When user clicks a graph node, pre-fill a "similar to" query
  useEffect(() => {
    if (selectedTrack && isOpen) {
      setQuery(`find something similar to ${selectedTrack.name}`);
      inputRef.current?.focus();
    }
  }, [selectedTrack]); // eslint-disable-line react-hooks/exhaustive-deps

  const runSearch = useCallback(async (queryText) => {
    const q = (queryText || query).trim();
    if (!q) return;

    setStatus('interpreting');
    setResult(null);
    setErrorMsg('');

    try {
      // Step 1 — interpret
      const { params } = await apiClient.post('/chat/interpret', { query: q });

      setStatus('searching');

      // Step 2 — search
      const searchResult = await apiClient.post('/chat/search', { params });

      setResult(searchResult);
      setHistory(prev => [{ query: q, result: searchResult }, ...prev.slice(0, 9)]);
      setStatus('done');

    } catch (err) {
      setErrorMsg(err.message || 'Unknown error');
      setStatus('error');
    }
  }, [query]);

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      runSearch();
    }
  };

  const isBusy = status === 'interpreting' || status === 'searching';

  return (
    <>
      {/* ── Pulse beacon ──────────────────────────────────────────────────── */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          title="Open DJMate Search"
          style={{
            position: 'fixed', bottom: 24, left: 24, zIndex: 1000,
            width: 52, height: 52, borderRadius: '50%',
            background: 'radial-gradient(circle at 35% 35%, #002222, #000)',
            border: '1px solid #00ffff88',
            cursor: 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 22,
            animation: 'djpulse 2.5s ease-in-out infinite',
            boxShadow: '0 0 20px #00ffff33',
          }}
        >
          🎛️
        </button>
      )}

      {/* ── Chat panel ────────────────────────────────────────────────────── */}
      {isOpen && (
        <div style={{
          position: 'fixed', bottom: 24, left: 24, zIndex: 1001,
          width: 380,
          maxHeight: isMinimised ? 46 : '80vh',
          display: 'flex', flexDirection: 'column',
          background: 'rgba(3, 6, 8, 0.97)',
          border: '1px solid #00ffff33',
          borderRadius: 14,
          backdropFilter: 'blur(24px)',
          boxShadow: '0 0 80px #00ffff0d, 0 32px 80px rgba(0,0,0,0.85)',
          overflow: 'hidden',
          fontFamily: "'IBM Plex Mono', 'Fira Code', 'Courier New', monospace",
          transition: 'max-height 0.25s cubic-bezier(0.4,0,0.2,1)',
        }}>

          {/* ── Header ──────────────────────────────────────────────────── */}
          <div
            onClick={() => setIsMinimised(v => !v)}
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '10px 14px', flexShrink: 0,
              background: 'linear-gradient(90deg, #001414, #000a0a)',
              borderBottom: isMinimised ? 'none' : '1px solid #001a1a',
              cursor: 'pointer', userSelect: 'none',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div style={{
                width: 7, height: 7, borderRadius: '50%',
                background: isBusy ? '#ffaa00' : '#00ff88',
                boxShadow: isBusy ? '0 0 8px #ffaa00' : '0 0 8px #00ff88',
                animation: isBusy ? 'blink 0.7s ease-in-out infinite' : 'none',
              }} />
              <span style={{ fontSize: 11, color: '#00ffff', fontWeight: 700, letterSpacing: '0.1em' }}>
                DJMATE · SEMANTIC SEARCH
              </span>
            </div>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <span style={{ fontSize: 10, color: '#00ffff44' }}>{isMinimised ? '▲' : '▼'}</span>
              <span
                style={{ fontSize: 14, color: '#ff444455', cursor: 'pointer', padding: '0 2px' }}
                onClick={e => { e.stopPropagation(); setIsOpen(false); }}
              >✕</span>
            </div>
          </div>

          {!isMinimised && (
            <div style={{
              flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column',
              scrollbarWidth: 'thin', scrollbarColor: '#00ffff18 transparent',
            }}>

              {/* ── Input area ──────────────────────────────────────────── */}
              <div style={{ padding: '12px 14px', borderBottom: '1px solid #001010', flexShrink: 0 }}>
                <div style={{ fontSize: 10, color: '#00ffff55', marginBottom: 6, letterSpacing: '0.08em' }}>
                  DESCRIBE WHAT YOU WANT TO PLAY NEXT
                </div>
                <div style={{ display: 'flex', gap: 8 }}>
                  <textarea
                    ref={inputRef}
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    onKeyDown={handleKey}
                    disabled={isBusy}
                    placeholder={'e.g. "fast kicking house", "dark minimal give me 8"'}
                    rows={2}
                    style={{
                      flex: 1,
                      background: '#050f0f',
                      border: '1px solid #00ffff22',
                      borderRadius: 8,
                      color: '#aae8e8',
                      fontSize: 12,
                      padding: '8px 10px',
                      fontFamily: 'inherit',
                      resize: 'none',
                      outline: 'none',
                      lineHeight: 1.5,
                      caretColor: '#00ffff',
                      opacity: isBusy ? 0.5 : 1,
                    }}
                    onFocus={e => e.target.style.borderColor = '#00ffff66'}
                    onBlur={e => e.target.style.borderColor = '#00ffff22'}
                  />
                  <button
                    onClick={() => runSearch()}
                    disabled={isBusy || !query.trim()}
                    style={{
                      padding: '0 14px',
                      background: (isBusy || !query.trim()) ? '#001010' : 'linear-gradient(135deg, #003322, #001a11)',
                      border: `1px solid ${(isBusy || !query.trim()) ? '#001a1a' : '#00ff8855'}`,
                      borderRadius: 8,
                      color: (isBusy || !query.trim()) ? '#002a2a' : '#00ff88',
                      cursor: (isBusy || !query.trim()) ? 'default' : 'pointer',
                      fontSize: 18,
                      fontFamily: 'inherit',
                      transition: 'all 0.15s',
                      alignSelf: 'stretch',
                    }}
                  >
                    {isBusy ? '…' : '↑'}
                  </button>
                </div>

                {/* Quick prompts */}
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 8 }}>
                  {QUICK_PROMPTS.map(p => (
                    <button
                      key={p}
                      disabled={isBusy}
                      onClick={() => { setQuery(p); inputRef.current?.focus(); }}
                      style={{
                        background: '#001212', border: '1px solid #00ffff22',
                        borderRadius: 10, color: '#00ffff66', fontSize: 10,
                        padding: '3px 9px', cursor: 'pointer', fontFamily: 'inherit',
                        letterSpacing: '0.04em',
                      }}
                      onMouseEnter={e => { e.target.style.borderColor = '#00ffff66'; e.target.style.color = '#00ffff'; }}
                      onMouseLeave={e => { e.target.style.borderColor = '#00ffff22'; e.target.style.color = '#00ffff66'; }}
                    >{p}</button>
                  ))}
                </div>
              </div>

              {/* ── Status / spinner ────────────────────────────────────── */}
              {isBusy && (
                <div style={{
                  padding: '14px 18px',
                  display: 'flex', alignItems: 'center', gap: 10,
                  color: '#00ffff88', fontSize: 11,
                }}>
                  <div style={{
                    width: 14, height: 14, borderRadius: '50%',
                    border: '2px solid #001a1a', borderTop: '2px solid #00ffff',
                    animation: 'spin 0.8s linear infinite',
                  }} />
                  {status === 'interpreting' ? 'Interpreting your request…' : 'Finding tracks…'}
                </div>
              )}

              {/* ── Error ───────────────────────────────────────────────── */}
              {status === 'error' && (
                <div style={{
                  margin: '10px 14px', padding: '10px 12px',
                  background: '#110000', border: '1px solid #440000',
                  borderRadius: 8, fontSize: 11, color: '#ff6666',
                }}>
                  ⚠ {errorMsg}
                  <div style={{ fontSize: 10, color: '#883333', marginTop: 4 }}>
                    Make sure the backend is running at localhost:8000 and /chat/* routes are mounted.
                  </div>
                </div>
              )}

              {/* ── Results ─────────────────────────────────────────────── */}
              <div ref={resultsRef} style={{ padding: '0 14px 14px' }}>
                <SearchResults result={result} source={selectedTrack} />
              </div>

              {/* ── Search history ───────────────────────────────────────── */}
              {history.length > 1 && !isBusy && (
                <div style={{
                  borderTop: '1px solid #001010', padding: '10px 14px', flexShrink: 0,
                }}>
                  <div style={{ fontSize: 10, color: '#00ffff33', marginBottom: 6, letterSpacing: '0.08em' }}>
                    RECENT SEARCHES
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {history.slice(1).map((h, i) => (
                      <button
                        key={i}
                        onClick={() => { setQuery(h.query); setResult(h.result); setStatus('done'); }}
                        style={{
                          background: '#001010', border: '1px solid #001e1e',
                          borderRadius: 8, color: '#005555', fontSize: 10,
                          padding: '3px 9px', cursor: 'pointer', fontFamily: 'inherit',
                        }}
                        onMouseEnter={e => { e.target.style.color = '#00aaaa'; e.target.style.borderColor = '#003333'; }}
                        onMouseLeave={e => { e.target.style.color = '#005555'; e.target.style.borderColor = '#001e1e'; }}
                      >{h.query}</button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      <style>{`
        @keyframes djpulse {
          0%, 100% { box-shadow: 0 0 20px #00ffff33; }
          50%       { box-shadow: 0 0 36px #00ffff66, 0 0 70px #00ffff22; }
        }
        @keyframes blink {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.15; }
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </>
  );
}