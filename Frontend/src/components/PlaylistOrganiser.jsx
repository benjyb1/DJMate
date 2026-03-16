// src/components/PlaylistOrganiser.jsx — Folder-based track organiser (card grid layout)
import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { LazyMotion, domAnimation, m, AnimatePresence } from 'framer-motion';
import { apiClient } from '../api/apiClient';
import GlassPanel from './ui/GlassPanel';
import { makeSupabaseCoverUrl } from '../utils/coverUrl';

// ── LibraryPlaylist ──────────────────────────────────────────────────────────
function LibraryPlaylist({ playlist, isSelected, onClick }) {
  return (
    <m.div
      onClick={onClick}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.97 }}
      style={{
        display: 'flex', alignItems: 'center', gap: 8,
        padding: '8px 12px', margin: '1px 0',
        borderRadius: 'var(--radius-sm)',
        cursor: 'pointer',
        background: isSelected ? 'rgba(0,212,255,0.08)' : 'transparent',
        border: isSelected ? '1px solid rgba(0,212,255,0.25)' : '1px solid transparent',
        transition: 'background 120ms ease, border-color 120ms ease',
      }}
    >
      <span style={{ color: isSelected ? '#a855f7' : '#475569', display: 'flex', flexShrink: 0 }}>
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M2 4c0-.6.4-1 1-1h3.6l1.4 1.5H13c.6 0 1 .4 1 1V12c0 .6-.4 1-1 1H3c-.6 0-1-.4-1-1V4z" stroke="currentColor" strokeWidth="1.2"/></svg>
      </span>
      <span style={{
        flex: 1, minWidth: 0, fontSize: 11, fontWeight: 600,
        color: isSelected ? '#e2e8f0' : '#94a3b8',
        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        fontFamily: 'var(--font-ui)',
      }}>
        {playlist.name}
      </span>
      <span style={{
        fontSize: 9, color: '#475569', flexShrink: 0,
        fontFamily: 'var(--font-mono)',
      }}>
        {playlist.track_count ?? 0}
      </span>
    </m.div>
  );
}

// ── WorkspaceFolder ──────────────────────────────────────────────────────────
function WorkspaceFolder({ folder, isExpanded, onToggle, onDelete, onDrop, onDragOver, trackCount }) {
  const [dragOver, setDragOver] = useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
    onDragOver?.(e);
  };
  const handleDragLeave = () => setDragOver(false);
  const handleDrop = (e) => {
    setDragOver(false);
    onDrop(e);
  };

  return (
    <m.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      transition={{ type: 'spring', damping: 25, stiffness: 300 }}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={onToggle}
      style={{
        display: 'flex', alignItems: 'center', gap: 6,
        padding: '5px 10px 5px 12px',
        borderRadius: 'var(--radius-pill)',
        cursor: 'pointer',
        background: dragOver
          ? 'rgba(0,212,255,0.12)'
          : isExpanded
            ? 'rgba(124,58,237,0.12)'
            : 'rgba(255,255,255,0.04)',
        border: dragOver
          ? '1px solid rgba(0,212,255,0.5)'
          : isExpanded
            ? '1px solid rgba(124,58,237,0.3)'
            : '1px solid var(--glass-border)',
        boxShadow: dragOver ? '0 0 12px rgba(0,212,255,0.2)' : 'none',
        transition: 'background 120ms ease, border-color 120ms ease, box-shadow 120ms ease',
      }}
    >
      <span style={{ color: isExpanded ? '#a855f7' : '#94a3b8', display: 'flex', flexShrink: 0 }}>
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none"><path d="M2 4c0-.6.4-1 1-1h3.6l1.4 1.5H13c.6 0 1 .4 1 1V12c0 .6-.4 1-1 1H3c-.6 0-1-.4-1-1V4z" stroke="currentColor" strokeWidth="1.2"/></svg>
      </span>
      <span style={{
        fontSize: 11, fontWeight: 600, color: '#e2e8f0',
        whiteSpace: 'nowrap', fontFamily: 'var(--font-ui)',
      }}>
        {folder.name}
      </span>
      <span style={{
        fontSize: 9, color: '#475569', fontFamily: 'var(--font-mono)',
        background: 'rgba(255,255,255,0.06)', borderRadius: 'var(--radius-pill)',
        padding: '1px 6px', minWidth: 18, textAlign: 'center',
      }}>
        {trackCount}
      </span>
      <m.button
        whileHover={{ scale: 1.15 }}
        whileTap={{ scale: 0.9 }}
        onClick={(e) => { e.stopPropagation(); onDelete(); }}
        aria-label="Delete folder"
        style={{
          background: 'none', border: 'none', color: '#475569',
          cursor: 'pointer', padding: 2, display: 'flex',
          alignItems: 'center', justifyContent: 'center',
          marginLeft: 2,
        }}
      >
        <svg width="10" height="10" viewBox="0 0 12 12"><path d="M2 2l8 8M10 2l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
      </m.button>
    </m.div>
  );
}

// ── TrackCard ────────────────────────────────────────────────────────────────
function TrackCard({ track, isSelected, isPlaying, onSelect, onPlay, onDragStart }) {
  const [artError, setArtError] = useState(false);
  const artUrl = makeSupabaseCoverUrl(track.artist, track.title);

  const gradientFallback = (
    <div style={{
      width: '100%', aspectRatio: '1', borderRadius: 'var(--radius-sm)',
      background: 'linear-gradient(135deg, rgba(124,58,237,0.3), rgba(0,212,255,0.2))',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontSize: 28, color: 'rgba(255,255,255,0.3)', fontFamily: 'var(--font-ui)',
    }}>
      {(track.title || '?')[0].toUpperCase()}
    </div>
  );

  return (
    <m.div
      draggable="true"
      onDragStart={(e) => {
        const tid = track.trackid || track.id;
        onDragStart(e, tid);
      }}
      onClick={onSelect}
      whileHover={{ y: -2 }}
      style={{
        width: '100%',
        background: 'rgba(255,255,255,0.03)',
        border: isSelected
          ? '1px solid rgba(124,58,237,0.5)'
          : '1px solid rgba(255,255,255,0.04)',
        borderRadius: 'var(--radius-md)',
        padding: 8,
        cursor: 'pointer',
        boxShadow: isSelected
          ? '0 0 0 2px #7c3aed, 0 0 12px rgba(124,58,237,0.3)'
          : 'var(--shadow-card)',
        transition: 'border-color 120ms ease, box-shadow 120ms ease',
        display: 'flex', flexDirection: 'column', gap: 6,
      }}
    >
      {/* Album art */}
      {artError || !artUrl ? gradientFallback : (
        <img
          src={artUrl}
          alt=""
          onError={() => setArtError(true)}
          loading="lazy"
          style={{
            width: '100%', aspectRatio: '1', objectFit: 'cover',
            borderRadius: 'var(--radius-sm)', display: 'block',
          }}
        />
      )}

      {/* Title */}
      <div style={{
        fontSize: 13, fontWeight: 600, color: '#e2e8f0',
        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        fontFamily: 'var(--font-ui)', lineHeight: 1.3,
      }}>
        {track.title || 'Unknown'}
      </div>

      {/* Artist */}
      <div style={{
        fontSize: 11, color: '#94a3b8',
        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        fontFamily: 'var(--font-ui)', lineHeight: 1.2, marginTop: -2,
      }}>
        {track.artist || 'Unknown'}
      </div>

      {/* Bottom row: play + BPM + Key */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6, marginTop: 'auto',
      }}>
        {/* Play/Pause button */}
        <m.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={(e) => { e.stopPropagation(); onPlay(); }}
          style={{
            background: 'rgba(124,58,237,0.12)', border: 'none',
            borderRadius: 'var(--radius-pill)', width: 24, height: 24,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            cursor: 'pointer', color: isPlaying ? '#00d4ff' : '#a855f7',
            flexShrink: 0,
            animation: isPlaying ? 'glowPulse 1.5s ease-in-out infinite' : 'none',
          }}
        >
          {isPlaying ? (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><rect x="3" y="2" width="2.5" height="10" rx="0.5" fill="currentColor"/><rect x="8.5" y="2" width="2.5" height="10" rx="0.5" fill="currentColor"/></svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M3 2l9 5-9 5V2z" fill="currentColor"/></svg>
          )}
        </m.button>

        {/* BPM pill */}
        {track.bpm && (
          <span style={{
            fontSize: 10, color: '#00d4ff', fontFamily: 'var(--font-mono)',
            background: 'rgba(0,212,255,0.08)', borderRadius: 'var(--radius-pill)',
            padding: '2px 6px',
          }}>
            {Math.round(track.bpm)}
          </span>
        )}

        {/* Key pill */}
        {track.key && (
          <span style={{
            fontSize: 10, color: '#a855f7', fontFamily: 'var(--font-mono)',
            background: 'rgba(124,58,237,0.08)', borderRadius: 'var(--radius-pill)',
            padding: '2px 6px', marginLeft: 'auto',
          }}>
            {track.key}
          </span>
        )}
      </div>
    </m.div>
  );
}

// ── ExpandedFolderTracks ─────────────────────────────────────────────────────
function ExpandedFolderTracks({ folderId }) {
  const [tracks, setTracks] = useState([]);

  useEffect(() => {
    (async () => {
      try {
        const data = await apiClient.get(`/playlists/${folderId}`);
        setTracks(data.tracks || []);
      } catch (err) {
        console.error(err);
      }
    })();
  }, [folderId]);

  return (
    <div style={{
      padding: '8px 20px', display: 'flex', gap: 8,
      overflowX: 'auto', background: 'rgba(124,58,237,0.05)',
    }}>
      {tracks.length === 0 && (
        <span style={{ color: '#475569', fontSize: 12, fontFamily: 'var(--font-ui)' }}>
          Empty folder -- drag tracks here
        </span>
      )}
      {tracks.map(t => (
        <div key={t.trackid || t.id} style={{
          padding: '4px 10px', background: 'rgba(255,255,255,0.04)',
          borderRadius: 'var(--radius-pill)',
          fontSize: 11, color: '#94a3b8', whiteSpace: 'nowrap',
          fontFamily: 'var(--font-mono)',
        }}>
          {t.title} -- {t.artist}
        </div>
      ))}
    </div>
  );
}

// ── PlaylistChat ─────────────────────────────────────────────────────────────
function PlaylistChat({
  chatQuery, setChatQuery, chatStatus, chatFeedback,
  suggestedTracks, onSubmit, onPlayTrack, playingTrackId,
  workspaceFolders, onAddToFolder,
}) {
  const [expanded, setExpanded] = useState(true);
  const quickPrompts = ['ORGANIZE ALL', 'SUGGEST TRACKS', 'MOVE TRACKS'];

  return (
    <div style={{
      borderTop: '1px solid var(--glass-border)',
      padding: expanded ? '14px 20px 16px' : '10px 20px',
      flexShrink: 0,
    }}>
      {/* Toggle */}
      <div
        onClick={() => setExpanded(v => !v)}
        style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          cursor: 'pointer', marginBottom: expanded ? 10 : 0,
        }}
      >
        <span style={{
          fontSize: 9, fontWeight: 700, letterSpacing: '0.2em',
          color: '#00d4ff', fontFamily: 'var(--font-mono)',
        }}>
          ORGANISER CHAT
        </span>
        <span style={{
          display: 'flex',
          transform: expanded ? 'rotate(90deg)' : 'rotate(0deg)',
          transition: 'transform 150ms ease',
          color: '#94a3b8',
        }}>
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M4 2l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
        </span>
      </div>

      <AnimatePresence>
        {expanded && (
          <m.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            style={{ overflow: 'hidden' }}
          >
            {/* Quick prompts */}
            <div style={{ display: 'flex', gap: 6, marginBottom: 10, flexWrap: 'wrap' }}>
              {quickPrompts.map(prompt => (
                <m.button
                  key={prompt}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.97 }}
                  onClick={() => onSubmit(prompt)}
                  style={{
                    padding: '5px 12px',
                    background: 'rgba(124,58,237,0.06)',
                    border: '1px solid rgba(124,58,237,0.2)',
                    borderRadius: 'var(--radius-pill)',
                    color: '#94a3b8', fontSize: 9, fontWeight: 700,
                    letterSpacing: '0.1em', cursor: 'pointer',
                    fontFamily: 'var(--font-ui)',
                  }}
                >
                  {prompt}
                </m.button>
              ))}
            </div>

            {/* Input bar */}
            <div style={{ display: 'flex', gap: 8 }}>
              <input
                value={chatQuery}
                onChange={e => setChatQuery(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && chatQuery.trim()) onSubmit(chatQuery); }}
                placeholder="e.g. Sort intro tracks by BPM ascending..."
                style={{
                  flex: 1, padding: '9px 14px',
                  background: 'rgba(8,8,20,0.5)',
                  border: '1px solid var(--glass-border)',
                  borderRadius: 'var(--radius-sm)',
                  color: '#e2e8f0', fontSize: 12,
                  fontFamily: 'var(--font-ui)',
                  outline: 'none',
                }}
              />
              <m.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.97 }}
                onClick={() => { if (chatQuery.trim()) onSubmit(chatQuery); }}
                disabled={chatStatus === 'thinking'}
                style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  width: 38, height: 38,
                  background: 'rgba(0,212,255,0.08)',
                  border: '1px solid rgba(0,212,255,0.3)',
                  borderRadius: 'var(--radius-sm)',
                  color: chatStatus === 'thinking' ? '#475569' : '#00d4ff',
                  cursor: chatStatus === 'thinking' ? 'wait' : 'pointer',
                }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
              </m.button>
            </div>

            {/* Status feedback */}
            {(chatStatus === 'thinking' || chatFeedback) && (
              <m.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                style={{
                  marginTop: 8, fontSize: 10, color: '#94a3b8',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                {chatStatus === 'thinking' ? 'Thinking...' : chatFeedback}
              </m.div>
            )}

            {/* Suggested tracks horizontal scroll */}
            {suggestedTracks && suggestedTracks.length > 0 && (
              <div style={{
                marginTop: 10, display: 'flex', gap: 8,
                overflowX: 'auto', paddingBottom: 4,
              }}>
                {suggestedTracks.map(track => {
                  const tid = track.trackid || track.id;
                  return (
                    <MiniTrackCard
                      key={tid}
                      track={track}
                      isPlaying={playingTrackId === tid}
                      onPlay={() => onPlayTrack(track)}
                    />
                  );
                })}
              </div>
            )}
          </m.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ── MiniTrackCard (for suggested tracks in chat) ─────────────────────────────
function MiniTrackCard({ track, isPlaying, onPlay }) {
  const [artError, setArtError] = useState(false);
  const artUrl = makeSupabaseCoverUrl(track.artist, track.title);

  return (
    <div style={{
      minWidth: 100, maxWidth: 100, flexShrink: 0,
      background: 'rgba(255,255,255,0.03)',
      border: '1px solid rgba(255,255,255,0.04)',
      borderRadius: 'var(--radius-sm)',
      padding: 6, cursor: 'pointer',
    }}>
      {artError || !artUrl ? (
        <div style={{
          width: '100%', aspectRatio: '1', borderRadius: 4,
          background: 'linear-gradient(135deg, rgba(124,58,237,0.3), rgba(0,212,255,0.2))',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 18, color: 'rgba(255,255,255,0.3)', fontFamily: 'var(--font-ui)',
        }}>
          {(track.title || '?')[0].toUpperCase()}
        </div>
      ) : (
        <img
          src={artUrl} alt=""
          onError={() => setArtError(true)}
          style={{ width: '100%', aspectRatio: '1', objectFit: 'cover', borderRadius: 4, display: 'block' }}
        />
      )}
      <div style={{
        fontSize: 10, fontWeight: 600, color: '#e2e8f0', marginTop: 4,
        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        fontFamily: 'var(--font-ui)',
      }}>
        {track.title || 'Unknown'}
      </div>
      <div style={{
        fontSize: 9, color: '#94a3b8', marginTop: 1,
        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        fontFamily: 'var(--font-ui)',
      }}>
        {track.artist || 'Unknown'}
      </div>
      <m.button
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        onClick={onPlay}
        style={{
          marginTop: 4, background: 'rgba(124,58,237,0.12)', border: 'none',
          borderRadius: 'var(--radius-pill)', width: 20, height: 20,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          cursor: 'pointer', color: isPlaying ? '#00d4ff' : '#a855f7',
        }}
      >
        {isPlaying ? (
          <svg width="10" height="10" viewBox="0 0 14 14" fill="none"><rect x="3" y="2" width="2.5" height="10" rx="0.5" fill="currentColor"/><rect x="8.5" y="2" width="2.5" height="10" rx="0.5" fill="currentColor"/></svg>
        ) : (
          <svg width="10" height="10" viewBox="0 0 14 14" fill="none"><path d="M3 2l9 5-9 5V2z" fill="currentColor"/></svg>
        )}
      </m.button>
    </div>
  );
}

// ── Main Component ───────────────────────────────────────────────────────────
export default function PlaylistOrganiser() {
  // Library (source)
  const [libraryPlaylists, setLibraryPlaylists] = useState([]);
  const [selectedSource, setSelectedSource] = useState('pool');
  const [sourceTracks, setSourceTracks] = useState([]);

  // Workspace (destination)
  const [workspaceFolders, setWorkspaceFolders] = useState([]);
  const [expandedFolderId, setExpandedFolderId] = useState(null);

  // Selection & drag
  const [selectedTrackIds, setSelectedTrackIds] = useState(new Set());
  const [lastSelectedIndex, setLastSelectedIndex] = useState(null);

  // Audio
  const audioRef = useRef(new Audio());
  const [playingTrackId, setPlayingTrackId] = useState(null);

  // Chat
  const [chatQuery, setChatQuery] = useState('');
  const [chatStatus, setChatStatus] = useState('idle');
  const [chatFeedback, setChatFeedback] = useState('');
  const [suggestedTracks, setSuggestedTracks] = useState([]);

  // UI
  const [searchFilter, setSearchFilter] = useState('');
  const [showNewFolderInput, setShowNewFolderInput] = useState(false);
  const [newFolderName, setNewFolderName] = useState('');
  const [loadError, setLoadError] = useState(null);

  // ── Data fetching ──────────────────────────────────────────────────────────
  const fetchLibrary = useCallback(async () => {
    try {
      const data = await apiClient.get('/playlists/tree');
      setLibraryPlaylists(Array.isArray(data) ? data : data.playlists || []);
      setLoadError(null);
    } catch (err) {
      setLoadError(err.message);
    }
  }, []);

  const fetchPool = useCallback(async () => {
    try {
      const data = await apiClient.get('/playlists/pool');
      setSourceTracks(Array.isArray(data) ? data : data.tracks || []);
    } catch (err) {
      console.error('Failed to fetch pool:', err);
    }
  }, []);

  useEffect(() => {
    fetchLibrary();
    fetchPool();
  }, [fetchLibrary, fetchPool]);

  // Audio cleanup
  useEffect(() => {
    const audio = audioRef.current;
    const handleEnded = () => setPlayingTrackId(null);
    audio.addEventListener('ended', handleEnded);
    return () => {
      audio.pause();
      audio.removeEventListener('ended', handleEnded);
    };
  }, []);

  // ── Source selection ────────────────────────────────────────────────────────
  const handleSelectSource = useCallback(async (sourceId) => {
    setSelectedSource(sourceId);
    setSelectedTrackIds(new Set());
    setLastSelectedIndex(null);
    if (sourceId === 'pool') {
      await fetchPool();
    } else {
      try {
        const data = await apiClient.get(`/playlists/${sourceId}`);
        setSourceTracks(data.tracks || []);
      } catch (err) {
        console.error(err);
      }
    }
  }, [fetchPool]);

  // ── Filter tracks ──────────────────────────────────────────────────────────
  const getDisplayedTracks = useCallback(() => {
    if (!searchFilter) return sourceTracks;
    const q = searchFilter.toLowerCase();
    return sourceTracks.filter(t =>
      (t.title || '').toLowerCase().includes(q) ||
      (t.artist || '').toLowerCase().includes(q)
    );
  }, [sourceTracks, searchFilter]);

  const displayedTracks = useMemo(() => getDisplayedTracks(), [getDisplayedTracks]);

  // ── Multi-select ───────────────────────────────────────────────────────────
  const handleTrackSelect = useCallback((trackId, index, event) => {
    setSelectedTrackIds(prev => {
      const next = new Set(prev);
      if (event.shiftKey && lastSelectedIndex !== null) {
        const start = Math.min(lastSelectedIndex, index);
        const end = Math.max(lastSelectedIndex, index);
        const displayed = getDisplayedTracks();
        for (let i = start; i <= end; i++) {
          next.add(displayed[i].trackid || displayed[i].id);
        }
      } else if (event.metaKey || event.ctrlKey) {
        if (next.has(trackId)) next.delete(trackId);
        else next.add(trackId);
      } else {
        next.clear();
        next.add(trackId);
      }
      return next;
    });
    setLastSelectedIndex(index);
  }, [lastSelectedIndex, getDisplayedTracks]);

  // ── Drag ───────────────────────────────────────────────────────────────────
  const handleDragStart = useCallback((e, trackId) => {
    let dragIds = selectedTrackIds.has(trackId)
      ? [...selectedTrackIds]
      : [trackId];

    const dragData = dragIds.map(id => {
      const track = sourceTracks.find(t => (t.trackid || t.id) === id);
      return { id, title: track?.title, artist: track?.artist };
    });

    e.dataTransfer.setData('application/json', JSON.stringify(dragData));
    e.dataTransfer.effectAllowed = 'copy';
  }, [selectedTrackIds, sourceTracks]);

  // ── Workspace folder operations ────────────────────────────────────────────
  const refreshWorkspaceFolders = useCallback(async () => {
    try {
      const data = await apiClient.get('/playlists/tree');
      const all = Array.isArray(data) ? data : data.playlists || [];
      setWorkspaceFolders(prev => {
        const existing = new Set(prev.map(f => f.id));
        const updated = all.filter(p => existing.has(p.id)).map(p => ({
          ...p,
          track_count: p.track_count || 0,
        }));
        const newOnes = all.filter(p =>
          !existing.has(p.id) && !libraryPlaylists.some(lp => lp.id === p.id)
        );
        return [...updated, ...newOnes.map(p => ({ ...p, track_count: p.track_count || 0 }))];
      });
    } catch (err) {
      console.error(err);
    }
  }, [libraryPlaylists]);

  const handleDropOnFolder = useCallback(async (folderId, e) => {
    e.preventDefault();
    try {
      const trackData = JSON.parse(e.dataTransfer.getData('application/json'));
      const trackIds = Array.isArray(trackData)
        ? trackData.map(t => t.id)
        : [trackData.id];

      await apiClient.post(`/playlists/${folderId}/add-tracks`, { track_ids: trackIds });
      await refreshWorkspaceFolders();
      setSelectedTrackIds(new Set());
    } catch (err) {
      console.error('Drop failed:', err);
    }
  }, [refreshWorkspaceFolders]);

  const handleCreateFolder = useCallback(async () => {
    if (!newFolderName.trim()) return;
    try {
      const data = await apiClient.post('/playlists/create', { name: newFolderName.trim() });
      const newFolder = {
        id: data.id || data.playlist_id,
        name: newFolderName.trim(),
        track_count: 0,
        tracks: [],
      };
      setWorkspaceFolders(prev => [...prev, newFolder]);
      setNewFolderName('');
      setShowNewFolderInput(false);
    } catch (err) {
      console.error(err);
    }
  }, [newFolderName]);

  const handleDeleteFolder = useCallback(async (folderId) => {
    try {
      await apiClient.delete(`/playlists/${folderId}`);
      setWorkspaceFolders(prev => prev.filter(f => f.id !== folderId));
      if (expandedFolderId === folderId) setExpandedFolderId(null);
      await fetchLibrary();
    } catch (err) {
      console.error('Failed to delete folder:', err);
    }
  }, [expandedFolderId, fetchLibrary]);

  // ── Audio playback ─────────────────────────────────────────────────────────
  const handlePlayTrack = useCallback((track) => {
    const trackId = track.trackid || track.id;
    if (playingTrackId === trackId) {
      audioRef.current.pause();
      setPlayingTrackId(null);
    } else {
      const audioUrl = track.filepath || track.audio_url || '';
      if (audioUrl) {
        audioRef.current.src = audioUrl;
        audioRef.current.play().catch(() => {});
        setPlayingTrackId(trackId);
      }
    }
  }, [playingTrackId]);

  // ── Chat submit ────────────────────────────────────────────────────────────
  const handleChatSubmit = useCallback(async (query) => {
    if (!query.trim()) return;
    setChatStatus('thinking');
    setChatFeedback('');
    setSuggestedTracks([]);
    try {
      const data = await apiClient.post('/playlists/organize', { query });
      setChatFeedback(data.message || 'Done');
      if (data.suggested_tracks) {
        setSuggestedTracks(data.suggested_tracks);
      }
      await refreshWorkspaceFolders();
      await fetchLibrary();
    } catch (err) {
      setChatFeedback(`Error: ${err.message}`);
    } finally {
      setChatStatus('done');
      setChatQuery('');
    }
  }, [refreshWorkspaceFolders, fetchLibrary]);

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <LazyMotion features={domAnimation}>
      <div style={{ display: 'flex', height: '100%', gap: 0 }}>

        {/* ═══════════ LEFT SIDEBAR — Library Browser ═══════════ */}
        <GlassPanel
          depth={2}
          animate={false}
          style={{
            width: 260, minWidth: 260,
            display: 'flex', flexDirection: 'column',
            borderRadius: 0,
            borderRight: '1px solid var(--glass-border)',
          }}
        >
          {/* Purple accent bar */}
          <div style={{ height: 2, background: 'linear-gradient(90deg, #7c3aed, #00d4ff)' }} />

          {/* Header */}
          <div style={{
            padding: '16px 16px 8px',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          }}>
            <span style={{
              fontSize: 11, fontFamily: 'var(--font-mono)',
              letterSpacing: 2, color: '#94a3b8',
            }}>
              LIBRARY
            </span>
            <span style={{
              fontSize: 11, fontFamily: 'var(--font-mono)', color: '#475569',
            }}>
              {libraryPlaylists.length}
            </span>
          </div>

          {/* All Tracks button */}
          <div style={{ padding: '4px 8px' }}>
            <m.button
              onClick={() => handleSelectSource('pool')}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.97 }}
              style={{
                width: '100%',
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '8px 12px',
                borderRadius: 'var(--radius-sm)',
                cursor: 'pointer',
                background: selectedSource === 'pool' ? 'rgba(0,212,255,0.08)' : 'transparent',
                border: selectedSource === 'pool'
                  ? '1px solid rgba(0,212,255,0.25)'
                  : '1px solid transparent',
                color: selectedSource === 'pool' ? '#e2e8f0' : '#94a3b8',
                fontSize: 11, fontWeight: 600,
                fontFamily: 'var(--font-ui)',
                transition: 'background 120ms ease, border-color 120ms ease',
              }}
            >
              <span style={{ color: selectedSource === 'pool' ? '#00d4ff' : '#475569', display: 'flex' }}>
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M6 12V3l7-2v9" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/><circle cx="4" cy="12" r="2" stroke="currentColor" strokeWidth="1.2"/><circle cx="11" cy="10" r="2" stroke="currentColor" strokeWidth="1.2"/></svg>
              </span>
              All Tracks
              <span style={{
                marginLeft: 'auto', fontSize: 9, color: '#475569',
                fontFamily: 'var(--font-mono)',
              }}>
                {sourceTracks.length}
              </span>
            </m.button>
          </div>

          {/* Library playlist list (scrollable) */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '4px 8px' }}>
            {loadError && (
              <div style={{
                padding: 12, fontSize: 10, color: '#94a3b8',
                fontFamily: 'var(--font-mono)', textAlign: 'center',
              }}>
                {loadError}
              </div>
            )}
            <AnimatePresence>
              {libraryPlaylists.map(pl => (
                <LibraryPlaylist
                  key={pl.id}
                  playlist={pl}
                  isSelected={selectedSource === pl.id}
                  onClick={() => handleSelectSource(pl.id)}
                />
              ))}
            </AnimatePresence>
          </div>
        </GlassPanel>

        {/* ═══════════ RIGHT MAIN AREA ═══════════ */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

          {/* WORKSPACE FOLDER BAR */}
          <div style={{
            padding: '12px 20px',
            display: 'flex', alignItems: 'center', gap: 8,
            borderBottom: '1px solid var(--glass-border)',
            background: 'rgba(8,8,20,0.4)',
            flexWrap: 'wrap',
          }}>
            <span style={{
              fontSize: 11, fontFamily: 'var(--font-mono)',
              letterSpacing: 2, color: '#94a3b8', marginRight: 8,
            }}>
              WORKSPACE
            </span>

            <AnimatePresence>
              {workspaceFolders.map(folder => (
                <WorkspaceFolder
                  key={folder.id}
                  folder={folder}
                  isExpanded={expandedFolderId === folder.id}
                  onToggle={() => setExpandedFolderId(prev => prev === folder.id ? null : folder.id)}
                  onDelete={() => handleDeleteFolder(folder.id)}
                  onDrop={(e) => handleDropOnFolder(folder.id, e)}
                  onDragOver={(e) => { e.preventDefault(); }}
                  trackCount={folder.track_count || 0}
                />
              ))}
            </AnimatePresence>

            {/* New folder button/input */}
            {showNewFolderInput ? (
              <m.div
                initial={{ width: 0, opacity: 0 }}
                animate={{ width: 'auto', opacity: 1 }}
                style={{ display: 'flex', gap: 4 }}
              >
                <input
                  value={newFolderName}
                  onChange={e => setNewFolderName(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === 'Enter') handleCreateFolder();
                    if (e.key === 'Escape') setShowNewFolderInput(false);
                  }}
                  placeholder="Folder name..."
                  autoFocus
                  style={{
                    background: 'rgba(255,255,255,0.06)',
                    border: '1px solid var(--glass-border)',
                    borderRadius: 'var(--radius-pill)',
                    padding: '4px 12px', color: '#e2e8f0',
                    fontSize: 12, fontFamily: 'var(--font-ui)',
                    outline: 'none', width: 140,
                  }}
                />
                <m.button
                  onClick={handleCreateFolder}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  style={{
                    background: '#00d4ff', color: '#0a0a14',
                    borderRadius: 'var(--radius-pill)',
                    padding: '4px 10px', fontSize: 11, fontWeight: 600,
                    border: 'none', cursor: 'pointer',
                    fontFamily: 'var(--font-ui)',
                  }}
                >
                  ADD
                </m.button>
              </m.div>
            ) : (
              <m.button
                onClick={() => setShowNewFolderInput(true)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                style={{
                  background: 'rgba(124,58,237,0.15)',
                  border: '1px solid rgba(124,58,237,0.3)',
                  borderRadius: 'var(--radius-pill)',
                  padding: '4px 12px', color: '#a855f7',
                  fontSize: 12, cursor: 'pointer',
                  display: 'flex', alignItems: 'center', gap: 4,
                  fontFamily: 'var(--font-ui)',
                }}
              >
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M6 1v10M1 6h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
                New
              </m.button>
            )}
          </div>

          {/* Expanded folder contents */}
          <AnimatePresence>
            {expandedFolderId && (
              <m.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ type: 'spring', damping: 25, stiffness: 300 }}
                style={{ overflow: 'hidden', borderBottom: '1px solid var(--glass-border)' }}
              >
                <ExpandedFolderTracks folderId={expandedFolderId} />
              </m.div>
            )}
          </AnimatePresence>

          {/* Search bar */}
          <div style={{ padding: '8px 20px', borderBottom: '1px solid rgba(124,58,237,0.08)' }}>
            <div style={{
              display: 'flex', alignItems: 'center', gap: 8,
              background: 'rgba(255,255,255,0.04)',
              borderRadius: 'var(--radius-pill)',
              padding: '6px 14px',
              border: '1px solid var(--glass-border)',
            }}>
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><circle cx="6" cy="6" r="4.5" stroke="#94a3b8" strokeWidth="1.2"/><path d="M9.5 9.5L13 13" stroke="#94a3b8" strokeWidth="1.2" strokeLinecap="round"/></svg>
              <input
                value={searchFilter}
                onChange={e => setSearchFilter(e.target.value)}
                placeholder="Search tracks..."
                style={{
                  background: 'none', border: 'none', color: '#e2e8f0',
                  fontSize: 13, fontFamily: 'var(--font-ui)',
                  outline: 'none', flex: 1,
                }}
              />
              {searchFilter && (
                <m.button
                  onClick={() => setSearchFilter('')}
                  whileHover={{ scale: 1.1 }}
                  style={{
                    background: 'none', border: 'none', color: '#475569',
                    cursor: 'pointer', padding: 0, display: 'flex',
                  }}
                >
                  <svg width="12" height="12" viewBox="0 0 12 12"><path d="M2 2l8 8M10 2l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
                </m.button>
              )}
            </div>
          </div>

          {/* TRACK CARDS GRID */}
          <div style={{ flex: 1, overflowY: 'auto', padding: 20 }}>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))',
              gap: 12,
            }}>
              {displayedTracks.map((track, index) => (
                <TrackCard
                  key={track.trackid || track.id}
                  track={track}
                  isSelected={selectedTrackIds.has(track.trackid || track.id)}
                  isPlaying={playingTrackId === (track.trackid || track.id)}
                  onSelect={(e) => handleTrackSelect(track.trackid || track.id, index, e)}
                  onPlay={() => handlePlayTrack(track)}
                  onDragStart={handleDragStart}
                />
              ))}
            </div>

            {sourceTracks.length === 0 && !loadError && (
              <div style={{ textAlign: 'center', padding: 60, color: '#475569' }}>
                <p style={{ fontSize: 14, fontFamily: 'var(--font-ui)' }}>
                  Select a playlist from the library to browse tracks
                </p>
              </div>
            )}
          </div>

          {/* CHAT PANEL (collapsible, bottom) */}
          <PlaylistChat
            chatQuery={chatQuery}
            setChatQuery={setChatQuery}
            chatStatus={chatStatus}
            chatFeedback={chatFeedback}
            suggestedTracks={suggestedTracks}
            onSubmit={handleChatSubmit}
            onPlayTrack={handlePlayTrack}
            playingTrackId={playingTrackId}
            workspaceFolders={workspaceFolders}
            onAddToFolder={handleDropOnFolder}
          />
        </div>
      </div>
    </LazyMotion>
  );
}
