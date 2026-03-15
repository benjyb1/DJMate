// src/components/PlaylistOrganiser.jsx
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { m, AnimatePresence } from 'framer-motion';
import { apiClient } from '../api/apiClient';
import GlassPanel from './ui/GlassPanel';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ── SVG Icons ─────────────────────────────────────────────────────────────
const IconImport = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="7 10 12 15 17 10" />
    <line x1="12" y1="15" x2="12" y2="3" />
  </svg>
);

const IconExport = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="17 8 12 3 7 8" />
    <line x1="12" y1="3" x2="12" y2="15" />
  </svg>
);

const IconPlus = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true">
    <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
  </svg>
);

const IconTrash = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <polyline points="3 6 5 6 21 6" />
    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
  </svg>
);

const IconSuggest = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <circle cx="12" cy="12" r="10" />
    <line x1="12" y1="16" x2="12" y2="12" />
    <line x1="12" y1="8" x2="12.01" y2="8" />
  </svg>
);

const IconPlaylist = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
    <line x1="8" y1="6" x2="21" y2="6" /><line x1="8" y1="12" x2="21" y2="12" /><line x1="8" y1="18" x2="21" y2="18" />
    <line x1="3" y1="6" x2="3.01" y2="6" /><line x1="3" y1="12" x2="3.01" y2="12" /><line x1="3" y1="18" x2="3.01" y2="18" />
  </svg>
);

const IconCheck = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <polyline points="20 6 9 17 4 12" />
  </svg>
);

// ── Helpers ───────────────────────────────────────────────────────────────
function formatDate(dateStr) {
  if (!dateStr) return '';
  const d = new Date(dateStr);
  return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
}

// ── Animation variants ───────────────────────────────────────────────────
const listItemVariants = {
  hidden: { opacity: 0, x: -12 },
  visible: (i) => ({
    opacity: 1, x: 0,
    transition: { delay: i * 0.03, type: 'spring', damping: 25, stiffness: 300 },
  }),
  exit: { opacity: 0, x: -12 },
};

const trackRowVariants = {
  hidden: { opacity: 0, y: 8 },
  visible: (i) => ({
    opacity: 1, y: 0,
    transition: { delay: i * 0.02, type: 'spring', damping: 25, stiffness: 300 },
  }),
  exit: { opacity: 0, y: -8, transition: { duration: 0.15 } },
};

// ── Component ─────────────────────────────────────────────────────────────
export default function PlaylistOrganiser() {
  const fileInputRef = useRef(null);

  const [playlists, setPlaylists] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [selectedPlaylist, setSelectedPlaylist] = useState(null);
  const [suggestions, setSuggestions] = useState([]);
  const [isImporting, setIsImporting] = useState(false);
  const [isSuggesting, setIsSuggesting] = useState(false);
  const [loadError, setLoadError] = useState(null);

  // ── Fetch playlists on mount ────────────────────────────────────────────
  const fetchPlaylists = useCallback(async () => {
    try {
      const data = await apiClient.get('/playlists');
      setPlaylists(Array.isArray(data) ? data : data.playlists || []);
      setLoadError(null);
    } catch (err) {
      setLoadError(err.message);
    }
  }, []);

  useEffect(() => {
    fetchPlaylists();
  }, [fetchPlaylists]);

  // ── Fetch selected playlist detail ──────────────────────────────────────
  useEffect(() => {
    if (!selectedId) {
      setSelectedPlaylist(null);
      setSuggestions([]);
      return;
    }
    (async () => {
      try {
        const data = await apiClient.get(`/playlists/${selectedId}`);
        setSelectedPlaylist(data);
        setSuggestions([]);
      } catch (err) {
        console.error('Failed to load playlist:', err);
      }
    })();
  }, [selectedId]);

  // ── Import Rekordbox XML ────────────────────────────────────────────────
  const handleImport = useCallback(async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setIsImporting(true);
    try {
      const text = await file.text();
      await apiClient.post('/playlists/import', { xml_content: text, filename: file.name });
      await fetchPlaylists();
    } catch (err) {
      console.error('Import failed:', err);
    } finally {
      setIsImporting(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }, [fetchPlaylists]);

  // ── Remove track from playlist ──────────────────────────────────────────
  const handleRemoveTrack = useCallback(async (trackId) => {
    if (!selectedId) return;
    try {
      await apiClient.delete(`/playlists/${selectedId}/tracks/${trackId}`);
      setSelectedPlaylist(prev => {
        if (!prev) return prev;
        return {
          ...prev,
          tracks: prev.tracks.filter(t => (t.trackid || t.id) !== trackId),
        };
      });
    } catch (err) {
      console.error('Failed to remove track:', err);
    }
  }, [selectedId]);

  // ── Suggest tracks ──────────────────────────────────────────────────────
  const handleSuggest = useCallback(async () => {
    if (!selectedId) return;
    setIsSuggesting(true);
    try {
      const data = await apiClient.get(`/playlists/${selectedId}/suggest`);
      setSuggestions(Array.isArray(data) ? data : data.suggestions || []);
    } catch (err) {
      console.error('Suggest failed:', err);
    } finally {
      setIsSuggesting(false);
    }
  }, [selectedId]);

  // ── Add suggestion to playlist ──────────────────────────────────────────
  const handleAddSuggestion = useCallback(async (trackId) => {
    if (!selectedId) return;
    try {
      await apiClient.post(`/playlists/${selectedId}/tracks`, { track_id: trackId });
      // Refresh
      const data = await apiClient.get(`/playlists/${selectedId}`);
      setSelectedPlaylist(data);
      setSuggestions(prev => prev.filter(s => (s.trackid || s.id) !== trackId));
    } catch (err) {
      console.error('Failed to add track:', err);
    }
  }, [selectedId]);

  // ── Export XML ──────────────────────────────────────────────────────────
  const handleExport = useCallback(async () => {
    if (!selectedId) return;
    try {
      const url = `${API_BASE}/playlists/${selectedId}/export`;
      const res = await fetch(url);
      if (!res.ok) throw new Error('Export failed');
      const blob = await res.blob();
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `${selectedPlaylist?.name || 'playlist'}.xml`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(a.href);
    } catch (err) {
      console.error('Export failed:', err);
    }
  }, [selectedId, selectedPlaylist]);

  const tracks = selectedPlaylist?.tracks || [];

  return (
    <div style={{
      display: 'flex', height: '100%', width: '100%',
      padding: '80px 24px 24px',
      gap: 20,
      fontFamily: "'Inter', system-ui, sans-serif",
    }}>

      {/* ═══════════ SIDEBAR ═══════════ */}
      <GlassPanel
        depth={2}
        style={{
          width: 280, minWidth: 280,
          borderRadius: 'var(--radius-xl)',
          display: 'flex', flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        {/* Gradient accent bar */}
        <div style={{
          height: 2,
          background: 'linear-gradient(90deg, #7c3aed, #00d4ff, rgba(0,212,255,0.1))',
        }} />

        {/* Header */}
        <div style={{ padding: '18px 20px 14px', borderBottom: '1px solid var(--border-subtle)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div style={{ color: '#a855f7' }}><IconPlaylist /></div>
              <span style={{
                fontSize: 10, fontWeight: 700, letterSpacing: '0.2em',
                color: '#e2e8f0',
              }}>
                PLAYLISTS
              </span>
            </div>
            <span style={{
              fontSize: 9, color: '#475569',
              fontFamily: "'JetBrains Mono', monospace",
              letterSpacing: '0.08em',
            }}>
              {playlists.length}
            </span>
          </div>
        </div>

        {/* Import button */}
        <div style={{ padding: '12px 16px 8px' }}>
          <input
            ref={fileInputRef}
            type="file"
            accept=".xml"
            onChange={handleImport}
            style={{ display: 'none' }}
          />
          <m.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => fileInputRef.current?.click()}
            disabled={isImporting}
            style={{
              width: '100%',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
              padding: '10px 0',
              background: 'rgba(124,58,237,0.08)',
              border: '1px solid rgba(124,58,237,0.3)',
              borderRadius: 'var(--radius-sm)',
              color: isImporting ? '#475569' : '#a855f7',
              cursor: isImporting ? 'wait' : 'pointer',
              fontSize: 10, fontWeight: 700,
              letterSpacing: '0.14em',
              fontFamily: "'Inter', system-ui, sans-serif",
            }}
          >
            <IconImport />
            {isImporting ? 'IMPORTING...' : 'IMPORT REKORDBOX XML'}
          </m.button>
        </div>

        {/* Playlist list */}
        <div style={{ flex: 1, overflow: 'auto', padding: '4px 8px 12px' }}>
          {loadError && (
            <div style={{
              padding: '12px 14px', margin: '4px 0',
              fontSize: 10, color: '#94a3b8',
              fontFamily: "'JetBrains Mono', monospace",
              textAlign: 'center',
            }}>
              {loadError}
            </div>
          )}
          {!loadError && playlists.length === 0 && (
            <div style={{
              padding: '32px 14px',
              fontSize: 11, color: '#475569',
              textAlign: 'center', lineHeight: 1.6,
            }}>
              No playlists yet. Import a Rekordbox XML to get started.
            </div>
          )}
          <AnimatePresence>
            {playlists.map((pl, i) => {
              const id = pl.id || pl.playlist_id;
              const isActive = selectedId === id;
              return (
                <m.div
                  key={id}
                  custom={i}
                  variants={listItemVariants}
                  initial="hidden"
                  animate="visible"
                  exit="exit"
                  onClick={() => setSelectedId(id)}
                  style={{
                    padding: '12px 14px',
                    margin: '2px 0',
                    borderRadius: 'var(--radius-sm)',
                    cursor: 'pointer',
                    background: isActive ? 'var(--gradient-accent-soft)' : 'transparent',
                    border: isActive
                      ? '1px solid rgba(124,58,237,0.25)'
                      : '1px solid transparent',
                    transition: 'background 150ms ease, border-color 150ms ease',
                  }}
                  whileHover={{
                    background: isActive
                      ? undefined
                      : 'rgba(124,58,237,0.04)',
                  }}
                >
                  <div style={{
                    fontSize: 12, fontWeight: 600,
                    color: isActive ? '#e2e8f0' : '#94a3b8',
                    marginBottom: 4,
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }}>
                    {pl.name}
                  </div>
                  <div style={{
                    display: 'flex', alignItems: 'center', gap: 8,
                    fontSize: 9, color: '#475569',
                    fontFamily: "'JetBrains Mono', monospace",
                    letterSpacing: '0.06em',
                  }}>
                    <span>{pl.track_count ?? '—'} tracks</span>
                    {pl.created_at && (
                      <>
                        <span style={{ color: 'var(--border-panel)' }}>|</span>
                        <span>{formatDate(pl.created_at)}</span>
                      </>
                    )}
                  </div>
                </m.div>
              );
            })}
          </AnimatePresence>
        </div>
      </GlassPanel>

      {/* ═══════════ MAIN AREA ═══════════ */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <AnimatePresence mode="wait">
          {!selectedId ? (
            <m.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              style={{
                flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}
            >
              <div style={{ textAlign: 'center' }}>
                <div style={{ color: 'rgba(124,58,237,0.2)', marginBottom: 16 }}>
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" aria-hidden="true">
                    <line x1="8" y1="6" x2="21" y2="6" /><line x1="8" y1="12" x2="21" y2="12" /><line x1="8" y1="18" x2="21" y2="18" />
                    <line x1="3" y1="6" x2="3.01" y2="6" /><line x1="3" y1="12" x2="3.01" y2="12" /><line x1="3" y1="18" x2="3.01" y2="18" />
                  </svg>
                </div>
                <div style={{
                  fontSize: 13, color: '#475569', letterSpacing: '0.04em',
                }}>
                  Select a playlist to view tracks
                </div>
              </div>
            </m.div>
          ) : (
            <m.div
              key={selectedId}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ type: 'spring', damping: 25, stiffness: 300 }}
              style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}
            >
              <GlassPanel
                depth={2}
                animate={false}
                style={{
                  flex: 1, display: 'flex', flexDirection: 'column',
                  borderRadius: 'var(--radius-xl)',
                  overflow: 'hidden', minHeight: 0,
                }}
              >
                {/* Gradient accent bar */}
                <div style={{
                  height: 2,
                  background: 'linear-gradient(90deg, #7c3aed, #00d4ff, rgba(0,212,255,0.1))',
                }} />

                {/* Header */}
                <div style={{
                  padding: '18px 24px 14px',
                  borderBottom: '1px solid var(--border-subtle)',
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  flexShrink: 0,
                }}>
                  <div>
                    <div style={{
                      fontSize: 18, fontWeight: 700, color: '#e2e8f0',
                      marginBottom: 4,
                    }}>
                      {selectedPlaylist?.name || 'Loading...'}
                    </div>
                    <div style={{
                      fontSize: 10, color: '#475569',
                      fontFamily: "'JetBrains Mono', monospace",
                      letterSpacing: '0.1em',
                    }}>
                      {tracks.length} TRACKS
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: 8 }}>
                    <m.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.97 }}
                      onClick={handleSuggest}
                      disabled={isSuggesting}
                      style={{
                        display: 'flex', alignItems: 'center', gap: 6,
                        padding: '8px 16px',
                        background: 'rgba(0,212,255,0.06)',
                        border: '1px solid rgba(0,212,255,0.3)',
                        borderRadius: 'var(--radius-sm)',
                        color: isSuggesting ? '#475569' : '#00d4ff',
                        cursor: isSuggesting ? 'wait' : 'pointer',
                        fontSize: 10, fontWeight: 700,
                        letterSpacing: '0.12em',
                        fontFamily: "'Inter', system-ui, sans-serif",
                      }}
                    >
                      <IconSuggest />
                      {isSuggesting ? 'ANALYSING...' : 'SUGGEST TRACKS'}
                    </m.button>

                    <m.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.97 }}
                      onClick={handleExport}
                      style={{
                        display: 'flex', alignItems: 'center', gap: 6,
                        padding: '8px 16px',
                        background: 'rgba(124,58,237,0.06)',
                        border: '1px solid rgba(124,58,237,0.3)',
                        borderRadius: 'var(--radius-sm)',
                        color: '#a855f7',
                        cursor: 'pointer',
                        fontSize: 10, fontWeight: 700,
                        letterSpacing: '0.12em',
                        fontFamily: "'Inter', system-ui, sans-serif",
                      }}
                    >
                      <IconExport />
                      EXPORT XML
                    </m.button>
                  </div>
                </div>

                {/* Track list */}
                <div style={{ flex: 1, overflow: 'auto', padding: '8px 12px' }}>
                  {/* Column headers */}
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: '36px 1fr 1fr 72px 56px 36px',
                    gap: 8,
                    padding: '8px 12px',
                    fontSize: 8, fontWeight: 600,
                    letterSpacing: '0.2em',
                    color: '#475569',
                    fontFamily: "'JetBrains Mono', monospace",
                    borderBottom: '1px solid var(--border-subtle)',
                    marginBottom: 4,
                  }}>
                    <span>#</span>
                    <span>TITLE</span>
                    <span>ARTIST</span>
                    <span>BPM</span>
                    <span>KEY</span>
                    <span />
                  </div>

                  {tracks.length === 0 && (
                    <div style={{
                      padding: '40px 0', textAlign: 'center',
                      fontSize: 11, color: '#475569',
                    }}>
                      No tracks in this playlist
                    </div>
                  )}

                  <AnimatePresence>
                    {tracks.map((track, i) => {
                      const tid = track.trackid || track.id;
                      return (
                        <m.div
                          key={tid}
                          custom={i}
                          variants={trackRowVariants}
                          initial="hidden"
                          animate="visible"
                          exit="exit"
                          layout
                          style={{
                            display: 'grid',
                            gridTemplateColumns: '36px 1fr 1fr 72px 56px 36px',
                            gap: 8,
                            padding: '10px 12px',
                            borderRadius: 'var(--radius-sm)',
                            alignItems: 'center',
                            transition: 'background 150ms ease',
                          }}
                          whileHover={{ background: 'rgba(124,58,237,0.04)' }}
                        >
                          <span style={{
                            fontSize: 10, color: '#475569',
                            fontFamily: "'JetBrains Mono', monospace",
                          }}>
                            {String(i + 1).padStart(2, '0')}
                          </span>
                          <span style={{
                            fontSize: 12, fontWeight: 600, color: '#e2e8f0',
                            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                          }}>
                            {track.title || track.name || 'Unknown'}
                          </span>
                          <span style={{
                            fontSize: 11, color: '#94a3b8',
                            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                          }}>
                            {track.artist || 'Unknown'}
                          </span>
                          <span style={{
                            fontSize: 11, color: '#00d4ff',
                            fontFamily: "'JetBrains Mono', monospace",
                          }}>
                            {track.bpm ? Math.round(track.bpm) : '—'}
                          </span>
                          <span style={{
                            fontSize: 11, color: '#a855f7',
                            fontFamily: "'JetBrains Mono', monospace",
                          }}>
                            {track.key || '—'}
                          </span>
                          <m.button
                            whileHover={{ scale: 1.15 }}
                            whileTap={{ scale: 0.9 }}
                            onClick={() => handleRemoveTrack(tid)}
                            aria-label="Remove track"
                            style={{
                              background: 'none', border: 'none',
                              color: '#475569', cursor: 'pointer',
                              padding: 4, borderRadius: 'var(--radius-sm)',
                              display: 'flex', alignItems: 'center', justifyContent: 'center',
                              transition: 'color 150ms ease',
                            }}
                            whileHover={{ color: '#ef4444' }}
                          >
                            <IconTrash />
                          </m.button>
                        </m.div>
                      );
                    })}
                  </AnimatePresence>
                </div>

                {/* ── Suggestions panel ─────────────────────────────────────── */}
                <AnimatePresence>
                  {suggestions.length > 0 && (
                    <m.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ type: 'spring', damping: 25, stiffness: 300 }}
                      style={{ overflow: 'hidden', flexShrink: 0 }}
                    >
                      <div style={{
                        borderTop: '1px solid var(--border-subtle)',
                        padding: '14px 24px 18px',
                      }}>
                        <div style={{
                          fontSize: 9, fontWeight: 700, letterSpacing: '0.2em',
                          color: '#00d4ff', marginBottom: 12,
                          fontFamily: "'JetBrains Mono', monospace",
                        }}>
                          SUGGESTED TRACKS
                        </div>

                        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                          {suggestions.map((s, i) => {
                            const sid = s.trackid || s.id;
                            return (
                              <m.div
                                key={sid}
                                custom={i}
                                variants={trackRowVariants}
                                initial="hidden"
                                animate="visible"
                                style={{
                                  display: 'flex', alignItems: 'center',
                                  justifyContent: 'space-between',
                                  padding: '8px 12px',
                                  borderRadius: 'var(--radius-sm)',
                                  background: 'rgba(0,212,255,0.03)',
                                  border: '1px solid rgba(0,212,255,0.1)',
                                }}
                              >
                                <div style={{ flex: 1, minWidth: 0 }}>
                                  <div style={{
                                    fontSize: 12, fontWeight: 600, color: '#e2e8f0',
                                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                                  }}>
                                    {s.title || s.name || 'Unknown'}
                                  </div>
                                  <div style={{
                                    fontSize: 10, color: '#94a3b8',
                                    display: 'flex', gap: 8, marginTop: 2,
                                  }}>
                                    <span>{s.artist || 'Unknown'}</span>
                                    {s.bpm && (
                                      <span style={{ color: '#00d4ff', fontFamily: "'JetBrains Mono', monospace" }}>
                                        {Math.round(s.bpm)} BPM
                                      </span>
                                    )}
                                    {s.key && (
                                      <span style={{ color: '#a855f7', fontFamily: "'JetBrains Mono', monospace" }}>
                                        {s.key}
                                      </span>
                                    )}
                                  </div>
                                </div>

                                <m.button
                                  whileHover={{ scale: 1.05 }}
                                  whileTap={{ scale: 0.95 }}
                                  onClick={() => handleAddSuggestion(sid)}
                                  style={{
                                    display: 'flex', alignItems: 'center', gap: 4,
                                    padding: '6px 12px',
                                    background: 'rgba(0,212,255,0.08)',
                                    border: '1px solid rgba(0,212,255,0.3)',
                                    borderRadius: 'var(--radius-pill)',
                                    color: '#00d4ff',
                                    cursor: 'pointer',
                                    fontSize: 9, fontWeight: 700,
                                    letterSpacing: '0.1em',
                                    fontFamily: "'Inter', system-ui, sans-serif",
                                    flexShrink: 0, marginLeft: 12,
                                  }}
                                >
                                  <IconPlus /> ADD
                                </m.button>
                              </m.div>
                            );
                          })}
                        </div>
                      </div>
                    </m.div>
                  )}
                </AnimatePresence>
              </GlassPanel>
            </m.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
