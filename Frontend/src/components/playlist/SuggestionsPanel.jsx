// playlist/SuggestionsPanel.jsx — Bottom half: LLM track suggestions
import React from 'react';
import { m, AnimatePresence } from 'framer-motion';
import PlaylistTrackCard from './PlaylistTrackCard';

export default function SuggestionsPanel({
  suggestions,         // { tracks: [], message?: string } | null
  loading,             // boolean
  activePlaylistName,  // string | null
  onAddTrack,          // (trackId) => void — add single track to active playlist
  onAddAll,            // (trackIds[]) => void — move all suggestion tracks to active playlist
  playingTrackId,
  onPlay,
}) {
  const hasSuggestions = suggestions?.tracks?.length > 0;

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
        <svg width="36" height="36" viewBox="0 0 36 36" fill="none" style={{ opacity: 0.3, marginBottom: 12 }}>
          {/* Robot head */}
          <rect x="8" y="10" width="20" height="18" rx="4" stroke="#64748b" strokeWidth="1.5"/>
          {/* Antenna */}
          <line x1="18" y1="10" x2="18" y2="5" stroke="#64748b" strokeWidth="1.5" strokeLinecap="round"/>
          <circle cx="18" cy="4" r="1.5" fill="#64748b"/>
          {/* Eyes */}
          <circle cx="14" cy="18" r="2" stroke="#64748b" strokeWidth="1.5"/>
          <circle cx="22" cy="18" r="2" stroke="#64748b" strokeWidth="1.5"/>
          {/* Mouth */}
          <line x1="14" y1="24" x2="22" y2="24" stroke="#64748b" strokeWidth="1.5" strokeLinecap="round"/>
        </svg>
        <p style={{ fontSize: 12, color: '#64748b', fontFamily: 'var(--font-ui)', textAlign: 'center', marginBottom: 4 }}>
          Ask the AI to find tracks
        </p>
        <p style={{ fontSize: 10, color: '#334155', fontFamily: 'var(--font-ui)', textAlign: 'center' }}>
          Use the chat bar below to search your library
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
      {/* Header */}
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

        <span style={{ fontSize: 13, fontWeight: 600, color: '#e2e8f0', fontFamily: 'var(--font-ui)' }}>
          Suggestions
        </span>
        <span style={{ fontSize: 10, color: '#475569', fontFamily: 'var(--font-mono)' }}>
          {suggestions.tracks.length} tracks
        </span>

        {/* Move All button (if active playlist exists) */}
        {activePlaylistName && onAddAll && (
          <m.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => {
              const trackIds = suggestions.tracks.map(t => t.trackid || t.id);
              onAddAll(trackIds);
            }}
            style={{
              marginLeft: 'auto',
              padding: '5px 12px',
              background: 'rgba(0,212,255,0.08)',
              border: '1px solid rgba(0,212,255,0.25)',
              borderRadius: 'var(--radius-pill)',
              color: '#00d4ff', fontSize: 10, fontWeight: 600,
              cursor: 'pointer', fontFamily: 'var(--font-ui)',
            }}
          >
            Move All to {activePlaylistName}
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
