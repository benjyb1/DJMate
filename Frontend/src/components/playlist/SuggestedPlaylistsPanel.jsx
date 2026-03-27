// playlist/SuggestedPlaylistsPanel.jsx — Auto-generated playlist suggestions (shown when no playlist active)
import React, { useState, useEffect, useCallback } from 'react';
import { m, AnimatePresence, LazyMotion, domAnimation } from 'framer-motion';
import { apiClient } from '../../api/apiClient';
import { makeSupabaseCoverUrl } from '../../utils/coverUrl';

// ── Icons ────────────────────────────────────────────────────────────────────
const RefreshIcon = () => (
  <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
    <path d="M11.5 2.5A6 6 0 1 0 13 7" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
    <path d="M13 2v3h-3" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const SparklesIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
    <path d="M8 1l1 3.5L12.5 5l-3.5 1L8 9.5 7 6 3.5 5l3.5-1L8 1z" stroke="currentColor" strokeWidth="1" strokeLinejoin="round"/>
    <path d="M12 9l.5 1.5 1.5.5-1.5.5-.5 1.5-.5-1.5L10 11.5l1.5-.5L12 9z" stroke="currentColor" strokeWidth="0.8" strokeLinejoin="round"/>
    <path d="M3 10l.4 1.1 1.1.4-1.1.4L3 13l-.4-1.1-1.1-.4 1.1-.4L3 10z" stroke="currentColor" strokeWidth="0.8" strokeLinejoin="round"/>
  </svg>
);

// ── Album Art Mosaic (2x2 grid of covers) ────────────────────────────────────
function ArtMosaic({ tracks }) {
  const coverTracks = tracks.slice(0, 4);
  return (
    <div style={{
      width: 72, height: 72, borderRadius: 8, overflow: 'hidden',
      display: 'grid', gridTemplateColumns: '1fr 1fr', gridTemplateRows: '1fr 1fr',
      flexShrink: 0,
    }}>
      {coverTracks.map((t, i) => (
        <MosaicTile key={t.trackid || i} track={t} />
      ))}
      {/* Fill empty slots */}
      {Array.from({ length: Math.max(0, 4 - coverTracks.length) }).map((_, i) => (
        <div key={`empty-${i}`} style={{
          background: 'linear-gradient(135deg, rgba(124,58,237,0.2), rgba(0,212,255,0.1))',
        }} />
      ))}
    </div>
  );
}

function MosaicTile({ track }) {
  const [err, setErr] = useState(false);
  const url = track.album_art_url || makeSupabaseCoverUrl(track.trackid || track.id);
  if (err || !url) {
    return (
      <div style={{
        background: 'linear-gradient(135deg, rgba(124,58,237,0.2), rgba(0,212,255,0.1))',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 10, color: 'rgba(255,255,255,0.2)',
      }}>
        {(track.title || '?')[0]}
      </div>
    );
  }
  return <img src={url} alt="" onError={() => setErr(true)} loading="lazy" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />;
}

// ── Playlist Card ────────────────────────────────────────────────────────────
function PlaylistCard({ playlist, onSelect, isExpanded }) {
  return (
    <m.div
      layout
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      whileHover={{ scale: 1.01 }}
      onClick={() => onSelect(playlist)}
      style={{
        display: 'flex', gap: 14, padding: 14,
        background: isExpanded
          ? 'rgba(124,58,237,0.08)'
          : 'rgba(255,255,255,0.02)',
        border: isExpanded
          ? '1px solid rgba(124,58,237,0.3)'
          : '1px solid rgba(255,255,255,0.04)',
        borderRadius: 12,
        cursor: 'pointer',
        transition: 'background 120ms ease, border-color 120ms ease',
      }}
    >
      <ArtMosaic tracks={playlist.tracks} />

      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 4 }}>
        <div style={{
          fontSize: 14, fontWeight: 700, color: '#e2e8f0',
          fontFamily: 'var(--font-ui)',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}>
          {playlist.name}
        </div>

        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span style={{
            fontSize: 10, color: '#64748b', fontFamily: 'var(--font-mono)',
          }}>
            {playlist.track_count} tracks
          </span>
          {playlist.bpm_range && (
            <span style={{
              fontSize: 10, color: '#475569', fontFamily: 'var(--font-mono)',
            }}>
              {playlist.bpm_range[0]}-{playlist.bpm_range[1]} BPM
            </span>
          )}
        </div>

        {/* Preview: first few artists */}
        <div style={{
          fontSize: 10, color: '#475569', fontFamily: 'var(--font-ui)',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          marginTop: 2,
        }}>
          {[...new Set(playlist.tracks.slice(0, 4).map(t => t.artist))].filter(Boolean).join(', ')}
        </div>
      </div>
    </m.div>
  );
}

// ── Expanded Track List ──────────────────────────────────────────────────────
function ExpandedPlaylist({ playlist, onBack, onCreatePlaylist, playingTrackId, onPlay, fetchTree }) {
  return (
    <m.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      style={{ display: 'flex', flexDirection: 'column', height: '100%' }}
    >
      {/* Header */}
      <div style={{
        padding: '12px 16px', display: 'flex', alignItems: 'center', gap: 10,
        borderBottom: '1px solid var(--glass-border)',
        flexShrink: 0,
      }}>
        <m.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={onBack}
          style={{
            background: 'none', border: 'none', cursor: 'pointer',
            color: '#64748b', display: 'flex', padding: 4,
          }}
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M10 3L5 8l5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </m.button>

        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 14, fontWeight: 700, color: '#e2e8f0', fontFamily: 'var(--font-ui)' }}>
            {playlist.name}
          </div>
          <div style={{ fontSize: 10, color: '#64748b', fontFamily: 'var(--font-mono)' }}>
            {playlist.tracks.length} tracks
            {playlist.bpm_range && ` \u00B7 ${playlist.bpm_range[0]}-${playlist.bpm_range[1]} BPM`}
          </div>
        </div>

        <m.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.97 }}
          onClick={async () => {
            await onCreatePlaylist(playlist.name, playlist.tracks.map(t => t.trackid));
            if (fetchTree) fetchTree();
          }}
          style={{
            padding: '6px 14px',
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
          Save
        </m.button>
      </div>

      {/* Track list */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '6px 8px' }}>
        {playlist.tracks.map((track) => (
          <ExpandedTrackRow
            key={track.trackid}
            track={track}
            isPlaying={playingTrackId === track.trackid}
            onPlay={onPlay}
          />
        ))}
      </div>
    </m.div>
  );
}

function ExpandedTrackRow({ track, isPlaying, onPlay }) {
  const [artErr, setArtErr] = useState(false);
  const artUrl = track.album_art_url || makeSupabaseCoverUrl(track.trackid || track.id);

  return (
    <m.div
      whileHover={{ background: 'rgba(255,255,255,0.03)' }}
      onClick={() => onPlay(track)}
      style={{
        display: 'flex', alignItems: 'center', gap: 10,
        padding: '6px 10px', borderRadius: 6, cursor: 'pointer',
        background: isPlaying ? 'rgba(124,58,237,0.08)' : 'transparent',
        border: isPlaying ? '1px solid rgba(124,58,237,0.2)' : '1px solid transparent',
      }}
    >
      <div style={{ width: 32, height: 32, flexShrink: 0, borderRadius: 4, overflow: 'hidden' }}>
        {artErr || !artUrl ? (
          <div style={{
            width: 32, height: 32, borderRadius: 4,
            background: 'linear-gradient(135deg, rgba(124,58,237,0.3), rgba(0,212,255,0.2))',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 14, color: 'rgba(255,255,255,0.3)',
          }}>
            {(track.title || '?')[0].toUpperCase()}
          </div>
        ) : (
          <img src={artUrl} alt="" onError={() => setArtErr(true)} loading="lazy"
            style={{ width: 32, height: 32, objectFit: 'cover' }} />
        )}
      </div>

      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{
          fontSize: 12, fontWeight: 600, color: isPlaying ? '#c4b5fd' : '#e2e8f0',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          fontFamily: 'var(--font-ui)',
        }}>
          {track.title || 'Unknown'}
        </div>
        <div style={{
          fontSize: 10, color: '#64748b',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          fontFamily: 'var(--font-ui)',
        }}>
          {track.artist || 'Unknown'}
        </div>
      </div>

      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexShrink: 0 }}>
        {track.bpm && (
          <span style={{ fontSize: 10, color: '#475569', fontFamily: 'var(--font-mono)' }}>
            {Math.round(track.bpm)}
          </span>
        )}
        {track.key && (
          <span style={{ fontSize: 10, color: '#475569', fontFamily: 'var(--font-mono)', minWidth: 40 }}>
            {track.key}
          </span>
        )}
      </div>
    </m.div>
  );
}

// ── Main Component ───────────────────────────────────────────────────────────
const PAGE_SIZE = 5;

export default function SuggestedPlaylistsPanel({ onCreatePlaylist, playingTrackId, onPlay, fetchTree }) {
  const [playlists, setPlaylists] = useState([]);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [pageOffset, setPageOffset] = useState(0);
  const [expandedPlaylist, setExpandedPlaylist] = useState(null);

  // ── Fetch existing suggestions ───────────────────────────────────────
  const fetchSuggested = useCallback(async () => {
    try {
      setLoading(true);
      const data = await apiClient.get('/suggested-playlists');
      setPlaylists(data.playlists || []);
      setPageOffset(0);
    } catch (err) {
      console.error('Failed to fetch suggested playlists:', err);
      setPlaylists([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchSuggested(); }, [fetchSuggested]);

  // ── Generate new batch ───────────────────────────────────────────────
  const handleGenerate = useCallback(async () => {
    try {
      setGenerating(true);
      const data = await apiClient.post('/suggested-playlists/generate');
      setPlaylists(data.playlists || []);
      setPageOffset(0);
      setExpandedPlaylist(null);
    } catch (err) {
      console.error('Failed to generate playlists:', err);
    } finally {
      setGenerating(false);
    }
  }, []);

  // ── Refresh (rotate page) ──────────────────────────────────────────
  const handleRefresh = useCallback(() => {
    const nextOffset = pageOffset + PAGE_SIZE;
    setPageOffset(nextOffset >= playlists.length ? 0 : nextOffset);
  }, [pageOffset, playlists.length]);

  const visiblePlaylists = playlists.slice(pageOffset, pageOffset + PAGE_SIZE);

  // ── Expanded view ──────────────────────────────────────────────────
  if (expandedPlaylist) {
    return (
      <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <AnimatePresence mode="wait">
          <ExpandedPlaylist
            key={expandedPlaylist.id}
            playlist={expandedPlaylist}
            onBack={() => setExpandedPlaylist(null)}
            onCreatePlaylist={onCreatePlaylist}
            playingTrackId={playingTrackId}
            onPlay={onPlay}
            fetchTree={fetchTree}
          />
        </AnimatePresence>
      </div>
    );
  }

  // ── Loading state ──────────────────────────────────────────────────
  if (loading) {
    return (
      <div style={{
        height: '100%', display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center', gap: 12,
      }}>
        <div style={{ display: 'flex', gap: 6 }}>
          {[0, 1, 2].map(i => (
            <m.div
              key={i}
              animate={{ opacity: [0.3, 1, 0.3] }}
              transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2 }}
              style={{ width: 8, height: 8, borderRadius: '50%', background: '#7c3aed' }}
            />
          ))}
        </div>
        <span style={{ fontSize: 11, color: '#64748b', fontFamily: 'var(--font-ui)' }}>
          Loading suggestions...
        </span>
      </div>
    );
  }

  // ── Empty / generate state ─────────────────────────────────────────
  if (playlists.length === 0) {
    return (
      <div style={{
        height: '100%', display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center', gap: 16, padding: 30,
      }}>
        <span style={{ color: 'rgba(124,58,237,0.4)', display: 'flex' }}>
          <SparklesIcon />
        </span>
        <div style={{ textAlign: 'center' }}>
          <p style={{ fontSize: 13, color: '#94a3b8', fontFamily: 'var(--font-ui)', marginBottom: 6 }}>
            No suggested playlists yet
          </p>
          <p style={{ fontSize: 10, color: '#475569', fontFamily: 'var(--font-ui)' }}>
            Generate playlists based on your library's sonic clusters
          </p>
        </div>
        <m.button
          whileHover={{ scale: 1.03 }}
          whileTap={{ scale: 0.97 }}
          onClick={handleGenerate}
          disabled={generating}
          style={{
            padding: '8px 20px',
            background: 'linear-gradient(135deg, rgba(124,58,237,0.3), rgba(0,212,255,0.2))',
            border: '1px solid rgba(124,58,237,0.4)',
            borderRadius: 'var(--radius-pill)',
            color: '#e2e8f0', fontSize: 12, fontWeight: 600,
            cursor: generating ? 'wait' : 'pointer',
            fontFamily: 'var(--font-ui)',
            opacity: generating ? 0.6 : 1,
            display: 'flex', alignItems: 'center', gap: 8,
          }}
        >
          <SparklesIcon />
          {generating ? 'Generating...' : 'Generate Playlists'}
        </m.button>
      </div>
    );
  }

  // ── Playlist cards ─────────────────────────────────────────────────
  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <div style={{
        padding: '12px 16px', display: 'flex', alignItems: 'center', gap: 10,
        borderBottom: '1px solid var(--glass-border)',
        flexShrink: 0,
      }}>
        <span style={{ color: '#7c3aed', display: 'flex' }}>
          <SparklesIcon />
        </span>
        <span style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0', fontFamily: 'var(--font-ui)', flex: 1 }}>
          Suggested for You
        </span>
        <span style={{ fontSize: 10, color: '#475569', fontFamily: 'var(--font-mono)' }}>
          {pageOffset + 1}-{Math.min(pageOffset + PAGE_SIZE, playlists.length)} of {playlists.length}
        </span>

        {/* Refresh (rotate) */}
        {playlists.length > PAGE_SIZE && (
          <m.button
            whileHover={{ scale: 1.08 }}
            whileTap={{ scale: 0.92, rotate: -180 }}
            onClick={handleRefresh}
            style={{
              background: 'rgba(255,255,255,0.04)',
              border: '1px solid rgba(255,255,255,0.08)',
              borderRadius: 6, padding: 5,
              cursor: 'pointer', color: '#94a3b8', display: 'flex',
            }}
            title="Show more"
          >
            <RefreshIcon />
          </m.button>
        )}

        {/* Regenerate */}
        <m.button
          whileHover={{ scale: 1.03 }}
          whileTap={{ scale: 0.97 }}
          onClick={handleGenerate}
          disabled={generating}
          style={{
            padding: '5px 12px',
            background: 'rgba(124,58,237,0.1)',
            border: '1px solid rgba(124,58,237,0.25)',
            borderRadius: 'var(--radius-pill)',
            color: '#a78bfa', fontSize: 10, fontWeight: 600,
            cursor: generating ? 'wait' : 'pointer',
            fontFamily: 'var(--font-ui)',
            opacity: generating ? 0.6 : 1,
          }}
        >
          {generating ? 'Generating...' : 'New Batch'}
        </m.button>
      </div>

      {/* Cards */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '10px 12px', display: 'flex', flexDirection: 'column', gap: 8 }}>
        <AnimatePresence mode="popLayout">
          {visiblePlaylists.map((pl) => (
            <PlaylistCard
              key={pl.id}
              playlist={pl}
              onSelect={setExpandedPlaylist}
            />
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
