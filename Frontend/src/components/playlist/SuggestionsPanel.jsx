// playlist/SuggestionsPanel.jsx — Bottom half: LLM suggestions (two modes)
import React, { useState } from 'react';
import { m, AnimatePresence } from 'framer-motion';
import PlaylistTrackCard from './PlaylistTrackCard';

export default function SuggestionsPanel({
  suggestions,         // { mode: 'playlist'|'tracks', name?: string, tracks: [], message?: string } | null
  loading,             // boolean
  activePlaylistName,  // string | null
  onCreatePlaylist,    // (name, trackIds[]) => void — create from suggestion group
  onAddTrack,          // (trackId) => void — add single track to active playlist
  onAddAll,            // (trackIds[]) => void — add all suggestion tracks to active playlist
  playingTrackId,
  onPlay,
}) {
  const [editName, setEditName] = useState('');
  const [nameEditing, setNameEditing] = useState(false);

  const hasSuggestions = suggestions?.tracks?.length > 0;
  const isPlaylistMode = suggestions?.mode === 'playlist';
  const displayName = nameEditing ? editName : (suggestions?.name || 'Suggested Playlist');

  // ── Loading state ─────────────────────────────────────────────────────
  if (loading) {
    return (
      <div style={{
        flex: 1, display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        padding: 40, minHeight: 160,
        borderTop: '1px solid var(--glass-border)',
      }}>
        <div style={{ display: 'flex', gap: 6, marginBottom: 12 }}>
          {[0, 1, 2].map(i => (
            <m.div
              key={i}
              animate={{ opacity: [0.3, 1, 0.3] }}
              transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2 }}
              style={{
                width: 8, height: 8, borderRadius: '50%',
                background: '#7c3aed',
              }}
            />
          ))}
        </div>
        <span style={{ fontSize: 11, color: '#64748b', fontFamily: 'var(--font-ui)' }}>
          AI is searching your library...
        </span>
      </div>
    );
  }

  // ── Empty state ───────────────────────────────────────────────────────
  if (!hasSuggestions) {
    return (
      <div style={{
        flex: 1, display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        padding: 30, minHeight: 160,
        borderTop: '1px solid var(--glass-border)',
      }}>
        <svg width="36" height="36" viewBox="0 0 36 36" fill="none" style={{ opacity: 0.25, marginBottom: 12 }}>
          <circle cx="18" cy="18" r="14" stroke="#64748b" strokeWidth="1.5"/>
          <path d="M14 22c1-2 6-2 8 0M14 15h.01M22 15h.01" stroke="#64748b" strokeWidth="1.5" strokeLinecap="round"/>
        </svg>
        <p style={{ fontSize: 12, color: '#64748b', fontFamily: 'var(--font-ui)', textAlign: 'center', marginBottom: 4 }}>
          Ask the AI to suggest tracks or create a playlist
        </p>
        <p style={{ fontSize: 10, color: '#334155', fontFamily: 'var(--font-ui)', textAlign: 'center' }}>
          Use the chat bar below to get started
        </p>
      </div>
    );
  }

  // ── Suggestions content ───────────────────────────────────────────────
  return (
    <div style={{
      flex: 1, display: 'flex', flexDirection: 'column',
      overflow: 'hidden', minHeight: 160,
      borderTop: '1px solid var(--glass-border)',
    }}>
      {/* Header (for playlist mode or general header) */}
      <div style={{
        padding: '10px 20px',
        display: 'flex', alignItems: 'center', gap: 10,
        borderBottom: '1px solid rgba(124,58,237,0.08)',
        background: 'rgba(8,8,20,0.3)',
        flexShrink: 0,
      }}>
        <span style={{ color: '#00d4ff', display: 'flex' }}>
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M7 1l1.8 3.6L13 5.3l-3 2.9.7 4.1L7 10.3l-3.7 2 .7-4.1-3-2.9 4.2-.7L7 1z" stroke="currentColor" strokeWidth="1.1" strokeLinejoin="round"/>
          </svg>
        </span>

        {isPlaylistMode ? (
          <>
            {nameEditing ? (
              <input
                value={editName}
                onChange={e => setEditName(e.target.value)}
                onBlur={() => setNameEditing(false)}
                onKeyDown={e => { if (e.key === 'Enter') setNameEditing(false); }}
                autoFocus
                style={{
                  fontSize: 13, fontWeight: 700, color: '#e2e8f0',
                  fontFamily: 'var(--font-ui)',
                  background: 'rgba(255,255,255,0.06)',
                  border: '1px solid rgba(0,212,255,0.3)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '2px 8px', outline: 'none',
                }}
              />
            ) : (
              <span
                onClick={() => { setEditName(displayName); setNameEditing(true); }}
                style={{
                  fontSize: 13, fontWeight: 700, color: '#e2e8f0',
                  fontFamily: 'var(--font-ui)', cursor: 'text',
                }}
              >
                {displayName}
              </span>
            )}
            <span style={{ fontSize: 10, color: '#475569', fontFamily: 'var(--font-mono)' }}>
              {suggestions.tracks.length} tracks
            </span>

            {/* Create Playlist button */}
            <m.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.97 }}
              onClick={() => {
                const name = nameEditing ? editName : displayName;
                const trackIds = suggestions.tracks.map(t => t.trackid || t.id);
                onCreatePlaylist(name, trackIds);
              }}
              style={{
                marginLeft: 'auto',
                padding: '5px 14px',
                background: 'linear-gradient(135deg, rgba(124,58,237,0.25), rgba(0,212,255,0.15))',
                border: '1px solid rgba(124,58,237,0.4)',
                borderRadius: 'var(--radius-pill)',
                color: '#e2e8f0', fontSize: 11, fontWeight: 700,
                cursor: 'pointer', fontFamily: 'var(--font-ui)',
                display: 'flex', alignItems: 'center', gap: 6,
              }}
            >
              <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                <path d="M5 0v10M0 5h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
              Create Playlist
            </m.button>
          </>
        ) : (
          <>
            <span style={{ fontSize: 13, fontWeight: 600, color: '#e2e8f0', fontFamily: 'var(--font-ui)' }}>
              Suggestions
            </span>
            <span style={{ fontSize: 10, color: '#475569', fontFamily: 'var(--font-mono)' }}>
              {suggestions.tracks.length} tracks
            </span>
          </>
        )}

        {/* Add All button (if active playlist exists) */}
        {activePlaylistName && onAddAll && (
          <m.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => {
              const trackIds = suggestions.tracks.map(t => t.trackid || t.id);
              onAddAll(trackIds);
            }}
            style={{
              marginLeft: isPlaylistMode ? 0 : 'auto',
              padding: '5px 12px',
              background: 'rgba(0,212,255,0.08)',
              border: '1px solid rgba(0,212,255,0.25)',
              borderRadius: 'var(--radius-pill)',
              color: '#00d4ff', fontSize: 10, fontWeight: 600,
              cursor: 'pointer', fontFamily: 'var(--font-ui)',
            }}
          >
            Add All to {activePlaylistName}
          </m.button>
        )}
      </div>

      {/* Track cards */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '8px 20px 12px' }}>
        <AnimatePresence>
          {suggestions.tracks.map((track, i) => (
            <PlaylistTrackCard
              key={track.trackid || track.id}
              track={track}
              isPlaying={playingTrackId === (track.trackid || track.id)}
              onPlay={onPlay}
              onAdd={activePlaylistName ? (id) => onAddTrack(id) : null}
              activePlaylistName={activePlaylistName}
              style={{ marginBottom: 4 }}
            />
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
