// playlist/PlaylistTrackCard.jsx — Card component for LLM suggestion tracks
import React, { useState, useRef } from 'react';
import { m } from 'framer-motion';
import { makeSupabaseCoverUrl } from '../../utils/coverUrl';

export default function PlaylistTrackCard({
  track,
  isPlaying,
  onPlay,
  onAdd,            // () => void — add this track to active playlist
  activePlaylistName, // string | null — name of current playlist (for button label)
  style,
}) {
  const [artError, setArtError] = useState(false);
  const trackId = track.trackid || track.id;
  const artUrl = makeSupabaseCoverUrl(trackId) || null;

  // Track mousedown position to distinguish clicks from drags
  const pointerStart = useRef(null);

  const handlePointerDown = (e) => {
    pointerStart.current = { x: e.clientX, y: e.clientY };
  };

  const handleClick = (e) => {
    if (!pointerStart.current) return;
    const dx = Math.abs(e.clientX - pointerStart.current.x);
    const dy = Math.abs(e.clientY - pointerStart.current.y);
    pointerStart.current = null;
    // Only fire play if the pointer barely moved (not a drag)
    if (dx < 6 && dy < 6) {
      onPlay(track);
    }
  };

  return (
    <m.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      draggable="true"
      onDragStart={(e) => {
        e.dataTransfer.setData('application/json', JSON.stringify([{ id: trackId, title: track.title, artist: track.artist }]));
        e.dataTransfer.effectAllowed = 'copy';
      }}
      whileHover={{ scale: 1.01 }}
      onPointerDown={handlePointerDown}
      onClick={handleClick}
      style={{
        display: 'flex', alignItems: 'center', gap: 10,
        padding: '8px 12px',
        background: 'rgba(8,8,20,0.4)',
        border: '1px solid var(--glass-border)',
        borderRadius: 'var(--radius-sm)',
        cursor: 'pointer',
        transition: 'border-color 100ms ease',
        ...style,
      }}
    >
      {/* Album art */}
      <div style={{ width: 40, height: 40, flexShrink: 0, borderRadius: 'var(--radius-sm)', overflow: 'hidden' }}>
        {artError || !artUrl ? (
          <div style={{
            width: 40, height: 40,
            background: 'linear-gradient(135deg, rgba(124,58,237,0.3), rgba(0,212,255,0.2))',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 16, color: 'rgba(255,255,255,0.3)', fontFamily: 'var(--font-ui)',
          }}>
            {(track.title || '?')[0].toUpperCase()}
          </div>
        ) : (
          <img src={artUrl} alt="" onError={() => setArtError(true)} loading="lazy"
            style={{ width: 40, height: 40, objectFit: 'cover', display: 'block' }} />
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

      {/* BPM pill */}
      {track.bpm && (
        <span style={{
          fontSize: 9, color: '#00d4ff', fontFamily: 'var(--font-mono)',
          background: 'rgba(0,212,255,0.08)', borderRadius: 'var(--radius-pill)',
          padding: '2px 6px', flexShrink: 0,
        }}>
          {Math.round(track.bpm)}
        </span>
      )}

      {/* Key pill */}
      {track.key && (
        <span style={{
          fontSize: 9, color: '#a855f7', fontFamily: 'var(--font-mono)',
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
        onClick={(e) => { e.stopPropagation(); onPlay(track); }}
        style={{
          background: 'rgba(124,58,237,0.12)', border: 'none',
          borderRadius: 'var(--radius-pill)', width: 26, height: 26,
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

      {/* Add to playlist button */}
      {onAdd && (
        <m.button
          whileHover={{ scale: 1.03 }}
          whileTap={{ scale: 0.97 }}
          onClick={(e) => { e.stopPropagation(); onAdd(trackId); }}
          style={{
            padding: '4px 10px',
            background: 'rgba(0,212,255,0.08)',
            border: '1px solid rgba(0,212,255,0.25)',
            borderRadius: 'var(--radius-pill)',
            color: '#00d4ff', fontSize: 9, fontWeight: 600,
            cursor: 'pointer', fontFamily: 'var(--font-ui)',
            flexShrink: 0, whiteSpace: 'nowrap',
          }}
        >
          + Add{activePlaylistName ? ` to ${activePlaylistName}` : ''}
        </m.button>
      )}
    </m.div>
  );
}
