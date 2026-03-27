// playlist/ActivePlaylistPanel.jsx — Top half: active playlist workspace (drop zone)
import React, { useState, useMemo, useCallback } from 'react';
import { m, AnimatePresence } from 'framer-motion';
import { makeSupabaseCoverUrl } from '../../utils/coverUrl';

// ── TrackRow (compact list row, draggable) ───────────────────────────────────
function TrackRow({ track, isSelected, isPlaying, onSelect, onPlay, onDragStart, onRemove }) {
  const [artError, setArtError] = useState(false);
  const artUrl = makeSupabaseCoverUrl(track.artist, track.title);

  return (
    <div
      draggable="true"
      onDragStart={(e) => onDragStart(e, track.trackid || track.id)}
      onClick={(e) => { onSelect(e); onPlay(); }}
      style={{
        display: 'flex', alignItems: 'center', gap: 10,
        padding: '6px 12px',
        background: isSelected ? 'rgba(124,58,237,0.08)' : 'transparent',
        border: isSelected ? '1px solid rgba(124,58,237,0.3)' : '1px solid transparent',
        borderRadius: 'var(--radius-sm)',
        cursor: 'grab',
        transition: 'background 80ms ease',
      }}
    >
      {/* Album art thumbnail */}
      <div style={{ width: 32, height: 32, flexShrink: 0, borderRadius: 4, overflow: 'hidden' }}>
        {artError || !artUrl ? (
          <div style={{
            width: 32, height: 32, borderRadius: 4,
            background: 'linear-gradient(135deg, rgba(124,58,237,0.3), rgba(0,212,255,0.2))',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 14, color: 'rgba(255,255,255,0.3)', fontFamily: 'var(--font-ui)',
          }}>
            {(track.title || '?')[0].toUpperCase()}
          </div>
        ) : (
          <img src={artUrl} alt="" onError={() => setArtError(true)} loading="lazy"
            style={{ width: 32, height: 32, objectFit: 'cover', display: 'block' }} />
        )}
      </div>

      {/* Title + Artist */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{
          fontSize: 12, fontWeight: 600, color: '#e2e8f0',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          fontFamily: 'var(--font-ui)', lineHeight: 1.3,
        }}>
          {track.title || 'Unknown'}
        </div>
        <div style={{
          fontSize: 10, color: '#64748b',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          fontFamily: 'var(--font-ui)', lineHeight: 1.2,
        }}>
          {track.artist || 'Unknown'}
        </div>
      </div>

      {/* BPM */}
      {track.bpm && (
        <span style={{
          fontSize: 10, color: '#00d4ff', fontFamily: 'var(--font-mono)',
          background: 'rgba(0,212,255,0.08)', borderRadius: 'var(--radius-pill)',
          padding: '2px 6px', flexShrink: 0,
        }}>
          {Math.round(track.bpm)}
        </span>
      )}

      {/* Key */}
      {track.key && (
        <span style={{
          fontSize: 10, color: '#a855f7', fontFamily: 'var(--font-mono)',
          background: 'rgba(124,58,237,0.08)', borderRadius: 'var(--radius-pill)',
          padding: '2px 6px', flexShrink: 0,
        }}>
          {track.key}
        </span>
      )}

      {/* Play button */}
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
          <svg width="12" height="12" viewBox="0 0 14 14" fill="none"><rect x="3" y="2" width="2.5" height="10" rx="0.5" fill="currentColor"/><rect x="8.5" y="2" width="2.5" height="10" rx="0.5" fill="currentColor"/></svg>
        ) : (
          <svg width="12" height="12" viewBox="0 0 14 14" fill="none"><path d="M3 2l9 5-9 5V2z" fill="currentColor"/></svg>
        )}
      </m.button>

      {/* Remove button */}
      {onRemove && (
        <m.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={(e) => { e.stopPropagation(); onRemove(track.trackid || track.id); }}
          style={{
            background: 'none', border: 'none', color: '#475569',
            cursor: 'pointer', padding: 0, display: 'flex', flexShrink: 0,
          }}
        >
          <svg width="10" height="10" viewBox="0 0 12 12"><path d="M2 2l8 8M10 2l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
        </m.button>
      )}
    </div>
  );
}


// ── Main Panel ───────────────────────────────────────────────────────────────
export default function ActivePlaylistPanel({
  playlist,        // { id, name } or null
  tracks,          // track[] for the active playlist
  playingTrackId,
  onPlay,
  onDropTracks,    // (trackIds[]) => void
  onRemoveTrack,   // (trackId) => void
  onCreatePlaylist,// () => void — opens new playlist flow
  onRenamePlaylist,// (id, newName) => void
  onSavePlaylist,  // (name: string) => void — saves unsaved playlist
  isUnsavedPlaylist, // boolean — true when workspace is unsaved
  searchFilter,
  onSearchChange,
}) {
  const [dragOver, setDragOver] = useState(false);
  const [selectedTrackIds, setSelectedTrackIds] = useState(new Set());
  const [lastSelectedIndex, setLastSelectedIndex] = useState(null);
  const [editingName, setEditingName] = useState(false);
  const [nameValue, setNameValue] = useState('');
  const [unsavedName, setUnsavedName] = useState('');

  // Filter tracks
  const filteredTracks = useMemo(() => {
    if (!searchFilter || !tracks) return tracks || [];
    const q = searchFilter.toLowerCase();
    return tracks.filter(t =>
      (t.title || '').toLowerCase().includes(q) ||
      (t.artist || '').toLowerCase().includes(q)
    );
  }, [tracks, searchFilter]);

  // Multi-select
  const handleTrackSelect = useCallback((trackId, index, event) => {
    setSelectedTrackIds(prev => {
      const next = new Set(prev);
      if (event.shiftKey && lastSelectedIndex !== null) {
        const start = Math.min(lastSelectedIndex, index);
        const end = Math.max(lastSelectedIndex, index);
        for (let i = start; i <= end; i++) {
          const t = filteredTracks[i];
          if (t) next.add(t.trackid || t.id);
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
  }, [lastSelectedIndex, filteredTracks]);

  // Drag start (from active playlist tracks)
  const handleDragStart = useCallback((e, trackId) => {
    const dragIds = selectedTrackIds.has(trackId) ? [...selectedTrackIds] : [trackId];
    const dragData = dragIds.map(id => {
      const track = (tracks || []).find(t => (t.trackid || t.id) === id);
      return { id, title: track?.title, artist: track?.artist };
    });
    e.dataTransfer.setData('application/json', JSON.stringify(dragData));
    e.dataTransfer.effectAllowed = 'copy';
  }, [selectedTrackIds, tracks]);

  // Drop handler
  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    if ((!playlist && !isUnsavedPlaylist) || !onDropTracks) return;
    try {
      const raw = e.dataTransfer.getData('application/json');
      if (!raw) return;
      const trackData = JSON.parse(raw);
      const trackIds = Array.isArray(trackData) ? trackData.map(t => t.id) : [trackData.id];
      onDropTracks(trackIds);
    } catch (err) {
      console.error('Drop failed:', err);
    }
  }, [playlist, isUnsavedPlaylist, onDropTracks]);

  // Rename
  const startRename = () => {
    if (!playlist) return;
    setNameValue(playlist.name || '');
    setEditingName(true);
  };
  const finishRename = () => {
    setEditingName(false);
    if (nameValue.trim() && playlist && onRenamePlaylist && nameValue.trim() !== playlist.name) {
      onRenamePlaylist(playlist.id, nameValue.trim());
    }
  };

  // ── Empty state: no playlist selected ─────────────────────────────────
  if (!playlist && !isUnsavedPlaylist) {
    return (
      <div
        onDragOver={(e) => e.preventDefault()}
        style={{
          flex: 1, display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center',
          padding: 40, minHeight: 200,
        }}
      >
        <svg width="48" height="48" viewBox="0 0 48 48" fill="none" style={{ opacity: 0.3, marginBottom: 16 }}>
          <rect x="8" y="12" width="32" height="24" rx="4" stroke="#64748b" strokeWidth="2"/>
          <path d="M24 20v8M20 24h8" stroke="#64748b" strokeWidth="2" strokeLinecap="round"/>
        </svg>
        <p style={{
          fontSize: 14, color: '#64748b', fontFamily: 'var(--font-ui)',
          textAlign: 'center', marginBottom: 16,
        }}>
          Select a playlist from the sidebar, or create a new one
        </p>
        <m.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.97 }}
          onClick={onCreatePlaylist}
          style={{
            padding: '10px 24px',
            background: 'rgba(124,58,237,0.12)',
            border: '1px solid rgba(124,58,237,0.3)',
            borderRadius: 'var(--radius-pill)',
            color: '#a855f7', fontSize: 13, fontWeight: 600,
            cursor: 'pointer', fontFamily: 'var(--font-ui)',
            display: 'flex', alignItems: 'center', gap: 8,
          }}
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M7 1v12M1 7h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
          New Playlist
        </m.button>
      </div>
    );
  }

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      style={{
        flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden',
        minHeight: 200,
        border: dragOver ? '2px solid rgba(0,212,255,0.4)' : '2px solid transparent',
        transition: 'border-color 100ms ease',
        background: dragOver ? 'rgba(0,212,255,0.03)' : 'transparent',
      }}
    >
      {/* Header bar */}
      <div style={{
        padding: '10px 20px',
        display: 'flex', alignItems: 'center', gap: 12,
        borderBottom: '1px solid var(--glass-border)',
        background: 'rgba(8,8,20,0.4)',
        flexShrink: 0,
      }}>
        <span style={{ color: '#a855f7', display: 'flex' }}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0 }}>
            <path d="M2 4c0-.6.4-1 1-1h3.6l1.4 1.5H13c.6 0 1 .4 1 1V12c0 .6-.4 1-1 1H3c-.6 0-1-.4-1-1V4z" stroke="currentColor" strokeWidth="1.1" fill="rgba(124,58,237,0.15)"/>
          </svg>
        </span>

        {/* Editable name */}
        {isUnsavedPlaylist ? (
          <input
            value={unsavedName}
            onChange={e => setUnsavedName(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && unsavedName.trim() && onSavePlaylist) onSavePlaylist(unsavedName.trim()); }}
            autoFocus
            placeholder="Playlist name..."
            style={{
              fontSize: 14, fontWeight: 700, color: '#e2e8f0',
              fontFamily: 'var(--font-ui)',
              background: 'rgba(255,255,255,0.06)',
              border: '1px solid rgba(124,58,237,0.4)',
              borderRadius: 'var(--radius-sm)',
              padding: '2px 8px', outline: 'none',
              flex: 1, minWidth: 0,
            }}
          />
        ) : editingName ? (
          <input
            value={nameValue}
            onChange={e => setNameValue(e.target.value)}
            onBlur={finishRename}
            onKeyDown={e => { if (e.key === 'Enter') finishRename(); if (e.key === 'Escape') setEditingName(false); }}
            autoFocus
            style={{
              fontSize: 14, fontWeight: 700, color: '#e2e8f0',
              fontFamily: 'var(--font-ui)',
              background: 'rgba(255,255,255,0.06)',
              border: '1px solid rgba(124,58,237,0.4)',
              borderRadius: 'var(--radius-sm)',
              padding: '2px 8px', outline: 'none',
            }}
          />
        ) : (
          <span
            onClick={startRename}
            style={{
              fontSize: 14, fontWeight: 700, color: '#e2e8f0',
              fontFamily: 'var(--font-ui)', cursor: 'text',
            }}
          >
            {playlist.name}
          </span>
        )}

        <span style={{ fontSize: 10, color: '#475569', fontFamily: 'var(--font-mono)' }}>
          {(tracks || []).length} tracks
        </span>

        {selectedTrackIds.size > 0 && (
          <span style={{
            fontSize: 10, color: '#00d4ff', fontFamily: 'var(--font-mono)',
            background: 'rgba(0,212,255,0.08)', borderRadius: 'var(--radius-pill)',
            padding: '2px 8px', marginLeft: isUnsavedPlaylist ? 0 : 'auto',
          }}>
            {selectedTrackIds.size} selected
          </span>
        )}

        {isUnsavedPlaylist && (
          <m.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
            disabled={!unsavedName.trim()}
            onClick={() => onSavePlaylist && onSavePlaylist(unsavedName.trim())}
            style={{
              marginLeft: 'auto',
              padding: '5px 14px',
              background: unsavedName.trim() ? 'rgba(124,58,237,0.2)' : 'rgba(124,58,237,0.06)',
              border: unsavedName.trim() ? '1px solid rgba(124,58,237,0.5)' : '1px solid rgba(124,58,237,0.15)',
              borderRadius: 'var(--radius-pill)',
              color: unsavedName.trim() ? '#a855f7' : '#475569',
              fontSize: 12, fontWeight: 600,
              cursor: unsavedName.trim() ? 'pointer' : 'not-allowed',
              fontFamily: 'var(--font-ui)',
              display: 'flex', alignItems: 'center', gap: 6,
              flexShrink: 0,
              transition: 'all 100ms ease',
            }}
          >
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <path d="M2 2h6l2 2v6c0 .6-.4 1-1 1H3c-.6 0-1-.4-1-1V3c0-.6.4-1 1-1z" stroke="currentColor" strokeWidth="1.1"/>
              <rect x="4" y="6" width="4" height="3" rx="0.5" stroke="currentColor" strokeWidth="0.8"/>
            </svg>
            Save Playlist
          </m.button>
        )}
      </div>

      {/* Search bar */}
      <div style={{ padding: '8px 20px', borderBottom: '1px solid rgba(124,58,237,0.08)', flexShrink: 0 }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          background: 'rgba(255,255,255,0.04)',
          borderRadius: 'var(--radius-pill)',
          padding: '6px 14px',
          border: '1px solid var(--glass-border)',
        }}>
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><circle cx="6" cy="6" r="4.5" stroke="#94a3b8" strokeWidth="1.2"/><path d="M9.5 9.5L13 13" stroke="#94a3b8" strokeWidth="1.2" strokeLinecap="round"/></svg>
          <input
            value={searchFilter || ''}
            onChange={e => onSearchChange(e.target.value)}
            placeholder="Search tracks..."
            style={{
              background: 'none', border: 'none', color: '#e2e8f0',
              fontSize: 13, fontFamily: 'var(--font-ui)',
              outline: 'none', flex: 1,
            }}
          />
          {searchFilter && (
            <m.button
              onClick={() => onSearchChange('')}
              whileHover={{ scale: 1.1 }}
              style={{ background: 'none', border: 'none', color: '#475569', cursor: 'pointer', padding: 0, display: 'flex' }}
            >
              <svg width="12" height="12" viewBox="0 0 12 12"><path d="M2 2l8 8M10 2l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
            </m.button>
          )}
        </div>
      </div>

      {/* Track list */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {filteredTracks.map((track, index) => (
          <TrackRow
            key={track.trackid || track.id}
            track={track}
            isSelected={selectedTrackIds.has(track.trackid || track.id)}
            isPlaying={playingTrackId === (track.trackid || track.id)}
            onSelect={(e) => handleTrackSelect(track.trackid || track.id, index, e)}
            onPlay={() => onPlay(track)}
            onDragStart={handleDragStart}
            onRemove={onRemoveTrack}
          />
        ))}

        {(tracks || []).length === 0 && (
          <div style={{ textAlign: 'center', padding: 40, color: '#475569' }}>
            <p style={{ fontSize: 13, fontFamily: 'var(--font-ui)' }}>
              {isUnsavedPlaylist ? 'No tracks yet' : 'This playlist is empty'}
            </p>
            <p style={{ fontSize: 11, fontFamily: 'var(--font-ui)', color: '#334155', marginTop: 4 }}>
              {isUnsavedPlaylist
                ? 'Drag tracks here from the AI results or your library'
                : 'Drag tracks here or ask the AI for suggestions'}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
