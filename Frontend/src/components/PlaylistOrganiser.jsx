// src/components/PlaylistOrganiser.jsx — Finder-style folder tree + track browser
import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { LazyMotion, domAnimation, m, AnimatePresence } from 'framer-motion';
import { apiClient } from '../api/apiClient';
import GlassPanel from './ui/GlassPanel';
import { makeSupabaseCoverUrl } from '../utils/coverUrl';

// ── helpers ─────────────────────────────────────────────────────────────────
function buildTree(flatList) {
  const map = {};
  const roots = [];
  for (const item of flatList) {
    map[item.id] = { ...item, children: [] };
  }
  for (const item of flatList) {
    if (item.parent_id && map[item.parent_id]) {
      map[item.parent_id].children.push(map[item.id]);
    } else {
      roots.push(map[item.id]);
    }
  }
  return roots;
}

// ── SVG icons ───────────────────────────────────────────────────────────────
const ChevronIcon = ({ open }) => (
  <svg width="10" height="10" viewBox="0 0 10 10" fill="none"
    style={{ transform: open ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 120ms ease', flexShrink: 0 }}>
    <path d="M3 1.5l4 3.5-4 3.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const FolderIcon = ({ open, size = 14 }) => (
  <svg width={size} height={size} viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0 }}>
    {open
      ? <path d="M1.5 12.5V4c0-.6.4-1 1-1H6l1.5 1.5H13c.6 0 1 .4 1 1v1H4.5l-3 6V4z" stroke="currentColor" strokeWidth="1.1" fill="rgba(124,58,237,0.15)"/>
      : <path d="M2 4c0-.6.4-1 1-1h3.6l1.4 1.5H13c.6 0 1 .4 1 1V12c0 .6-.4 1-1 1H3c-.6 0-1-.4-1-1V4z" stroke="currentColor" strokeWidth="1.1"/>
    }
  </svg>
);

const TrackIcon = () => (
  <svg width="12" height="12" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0 }}>
    <path d="M6 12V3l7-2v9" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round"/>
    <circle cx="4" cy="12" r="2" stroke="currentColor" strokeWidth="1.1"/>
    <circle cx="11" cy="10" r="2" stroke="currentColor" strokeWidth="1.1"/>
  </svg>
);

const PlusIcon = () => (
  <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
    <path d="M6 1v10M1 6h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
  </svg>
);

// ── FolderTreeNode (recursive) ──────────────────────────────────────────────
function FolderTreeNode({
  node, depth, selectedId, expandedIds, onToggleExpand, onSelect,
  onDrop, folderTracks,
}) {
  const [dragOver, setDragOver] = useState(false);
  const isExpanded = expandedIds.has(node.id);
  const isSelected = selectedId === node.id;
  const hasChildren = node.children && node.children.length > 0;
  const tracks = folderTracks[node.id] || [];

  const handleDragOver = (e) => { e.preventDefault(); e.stopPropagation(); setDragOver(true); };
  const handleDragLeave = (e) => { e.stopPropagation(); setDragOver(false); };
  const handleDrop = (e) => { e.stopPropagation(); setDragOver(false); onDrop(node.id, e); };

  return (
    <div>
      {/* Folder row */}
      <div
        onClick={() => { onToggleExpand(node.id); onSelect(node.id); }}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        style={{
          display: 'flex', alignItems: 'center', gap: 4,
          padding: '4px 8px 4px ' + (8 + depth * 16) + 'px',
          cursor: 'pointer',
          background: dragOver
            ? 'rgba(0,212,255,0.12)'
            : isSelected
              ? 'rgba(124,58,237,0.1)'
              : 'transparent',
          border: dragOver ? '1px solid rgba(0,212,255,0.4)' : '1px solid transparent',
          borderRadius: 4,
          margin: '0 4px',
          transition: 'background 80ms ease, border-color 80ms ease',
          minHeight: 26,
        }}
      >
        <span style={{ color: '#64748b', display: 'flex', width: 10 }}>
          {(hasChildren || tracks.length > 0 || node.track_count > 0) && <ChevronIcon open={isExpanded} />}
        </span>
        <span style={{ color: isSelected ? '#a855f7' : '#64748b', display: 'flex' }}>
          <FolderIcon open={isExpanded} />
        </span>
        <span style={{
          flex: 1, minWidth: 0, fontSize: 12, fontWeight: isSelected ? 600 : 500,
          color: isSelected ? '#e2e8f0' : '#94a3b8',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          fontFamily: 'var(--font-ui)',
        }}>
          {node.name}
        </span>
        {(node.track_count > 0 || node.children?.length > 0) && (
          <span style={{
            fontSize: 9, color: '#475569', fontFamily: 'var(--font-mono)',
            flexShrink: 0, marginLeft: 4,
          }}>
            {node.track_count ?? 0}
          </span>
        )}
      </div>

      {/* Children (sub-folders + tracks) */}
      {isExpanded && (
        <div>
          {node.children.map(child => (
            <FolderTreeNode
              key={child.id}
              node={child}
              depth={depth + 1}
              selectedId={selectedId}
              expandedIds={expandedIds}
              onToggleExpand={onToggleExpand}
              onSelect={onSelect}
              onDrop={onDrop}
              folderTracks={folderTracks}
            />
          ))}
          {/* Inline tracks (lazy-loaded) */}
          {tracks.map(t => (
            <div
              key={t.trackid || t.id}
              draggable="true"
              onDragStart={(e) => {
                const tid = t.trackid || t.id;
                e.dataTransfer.setData('application/json', JSON.stringify([{ id: tid, title: t.title, artist: t.artist }]));
                e.dataTransfer.effectAllowed = 'copy';
              }}
              style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '3px 8px 3px ' + (8 + (depth + 1) * 16) + 'px',
                cursor: 'grab', margin: '0 4px', borderRadius: 4,
                fontSize: 11, color: '#64748b', fontFamily: 'var(--font-ui)',
              }}
            >
              <span style={{ color: '#475569', display: 'flex' }}><TrackIcon /></span>
              <span style={{
                flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}>
                {t.artist ? `${t.artist} - ${t.title}` : t.title || 'Unknown'}
              </span>
            </div>
          ))}
          {isExpanded && !hasChildren && tracks.length === 0 && node.track_count === 0 && (
            <div style={{
              padding: '4px 8px 4px ' + (8 + (depth + 1) * 16) + 'px',
              fontSize: 10, color: '#334155', fontFamily: 'var(--font-mono)', fontStyle: 'italic',
            }}>
              Empty -- drag tracks here
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── TrackRow (main area — compact list row, draggable) ──────────────────────
function TrackRow({ track, isSelected, isPlaying, onSelect, onPlay, onDragStart }) {
  const [artError, setArtError] = useState(false);
  const artUrl = makeSupabaseCoverUrl(track.artist, track.title);

  return (
    <div
      draggable="true"
      onDragStart={(e) => onDragStart(e, track.trackid || track.id)}
      onClick={onSelect}
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
    </div>
  );
}

// ── PlaylistChat ─────────────────────────────────────────────────────────────
function PlaylistChat({ chatQuery, setChatQuery, chatStatus, chatFeedback, onSubmit }) {
  const [expanded, setExpanded] = useState(true);
  const quickPrompts = ['ORGANIZE ALL', 'SUGGEST TRACKS', 'CREATE 3 SET FOLDERS'];

  return (
    <div style={{
      borderTop: '1px solid var(--glass-border)',
      padding: expanded ? '14px 20px 16px' : '10px 20px',
      flexShrink: 0,
    }}>
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
                placeholder="e.g. Make 3 folders: Intro, Peak, Cooldown..."
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
          </m.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ── Main Component ───────────────────────────────────────────────────────────
export default function PlaylistOrganiser() {
  // All playlists from DB (flat)
  const [allPlaylists, setAllPlaylists] = useState([]);
  // Track pool (all tracks)
  const [poolTracks, setPoolTracks] = useState([]);
  // Currently selected folder (shows its tracks in main area)
  const [selectedFolderId, setSelectedFolderId] = useState('pool');
  // Tracks displayed in main area
  const [displayTracks, setDisplayTracks] = useState([]);
  // Expanded folder IDs in sidebar
  const [expandedIds, setExpandedIds] = useState(new Set());
  // Lazy-loaded folder tracks (for inline sidebar display)
  const [folderTracks, setFolderTracks] = useState({});

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

  // UI
  const [searchFilter, setSearchFilter] = useState('');
  const [showNewFolderInput, setShowNewFolderInput] = useState(false);
  const [newFolderName, setNewFolderName] = useState('');
  const [loadError, setLoadError] = useState(null);

  // ── Build tree from flat list ───────────────────────────────────────────
  const tree = useMemo(() => buildTree(allPlaylists), [allPlaylists]);

  // ── Data fetching ───────────────────────────────────────────────────────
  const fetchTree = useCallback(async () => {
    try {
      const data = await apiClient.get('/playlists/tree');
      const list = Array.isArray(data) ? data : data.playlists || [];
      setAllPlaylists(list);
      setLoadError(null);
    } catch (err) {
      setLoadError(err.message);
    }
  }, []);

  const fetchPool = useCallback(async () => {
    try {
      const data = await apiClient.get('/playlists/pool');
      const tracks = Array.isArray(data) ? data : data.tracks || [];
      setPoolTracks(tracks);
      if (selectedFolderId === 'pool') setDisplayTracks(tracks);
    } catch (err) {
      console.error('Failed to fetch pool:', err);
    }
  }, [selectedFolderId]);

  useEffect(() => { fetchTree(); fetchPool(); }, [fetchTree, fetchPool]);

  // Audio cleanup
  useEffect(() => {
    const audio = audioRef.current;
    const handleEnded = () => setPlayingTrackId(null);
    audio.addEventListener('ended', handleEnded);
    return () => { audio.pause(); audio.removeEventListener('ended', handleEnded); };
  }, []);

  // ── Folder selection (loads tracks into main area) ──────────────────────
  const handleSelectFolder = useCallback(async (folderId) => {
    setSelectedFolderId(folderId);
    setSelectedTrackIds(new Set());
    setLastSelectedIndex(null);
    if (folderId === 'pool') {
      setDisplayTracks(poolTracks);
    } else {
      try {
        const data = await apiClient.get(`/playlists/${folderId}`);
        const tracks = data.tracks || [];
        setDisplayTracks(tracks);
        // Cache for sidebar inline display
        setFolderTracks(prev => ({ ...prev, [folderId]: tracks }));
      } catch (err) {
        console.error(err);
        setDisplayTracks([]);
      }
    }
  }, [poolTracks]);

  // ── Expand/collapse toggle (lazy-loads tracks) ─────────────────────────
  const handleToggleExpand = useCallback(async (folderId) => {
    setExpandedIds(prev => {
      const next = new Set(prev);
      if (next.has(folderId)) {
        next.delete(folderId);
      } else {
        next.add(folderId);
        // Lazy-load tracks if not cached
        if (!folderTracks[folderId]) {
          apiClient.get(`/playlists/${folderId}`).then(data => {
            setFolderTracks(p => ({ ...p, [folderId]: data.tracks || [] }));
          }).catch(() => {});
        }
      }
      return next;
    });
  }, [folderTracks]);

  // ── Filter tracks in main area ─────────────────────────────────────────
  const filteredTracks = useMemo(() => {
    if (!searchFilter) return displayTracks;
    const q = searchFilter.toLowerCase();
    return displayTracks.filter(t =>
      (t.title || '').toLowerCase().includes(q) ||
      (t.artist || '').toLowerCase().includes(q)
    );
  }, [displayTracks, searchFilter]);

  // ── Multi-select ───────────────────────────────────────────────────────
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

  // ── Drag start (from main area tracks) ─────────────────────────────────
  const handleDragStart = useCallback((e, trackId) => {
    const dragIds = selectedTrackIds.has(trackId) ? [...selectedTrackIds] : [trackId];
    const allTracks = [...displayTracks, ...poolTracks];
    const dragData = dragIds.map(id => {
      const track = allTracks.find(t => (t.trackid || t.id) === id);
      return { id, title: track?.title, artist: track?.artist };
    });
    e.dataTransfer.setData('application/json', JSON.stringify(dragData));
    e.dataTransfer.effectAllowed = 'copy';
  }, [selectedTrackIds, displayTracks, poolTracks]);

  // ── Drop on folder (sidebar) ──────────────────────────────────────────
  const handleDropOnFolder = useCallback(async (folderId, e) => {
    e.preventDefault();
    try {
      const raw = e.dataTransfer.getData('application/json');
      if (!raw) return;
      const trackData = JSON.parse(raw);
      const trackIds = Array.isArray(trackData) ? trackData.map(t => t.id) : [trackData.id];

      await apiClient.post(`/playlists/${folderId}/add-tracks`, { track_ids: trackIds });
      // Refresh that folder's track cache
      const data = await apiClient.get(`/playlists/${folderId}`);
      setFolderTracks(prev => ({ ...prev, [folderId]: data.tracks || [] }));
      // Refresh tree counts
      await fetchTree();
      setSelectedTrackIds(new Set());
    } catch (err) {
      console.error('Drop failed:', err);
    }
  }, [fetchTree]);

  // ── Create folder ─────────────────────────────────────────────────────
  const handleCreateFolder = useCallback(async () => {
    if (!newFolderName.trim()) return;
    try {
      await apiClient.post('/playlists/create', { name: newFolderName.trim() });
      setNewFolderName('');
      setShowNewFolderInput(false);
      await fetchTree();
    } catch (err) {
      console.error('Failed to create folder:', err);
    }
  }, [newFolderName, fetchTree]);

  // ── Delete folder ─────────────────────────────────────────────────────
  const handleDeleteFolder = useCallback(async (folderId) => {
    try {
      await apiClient.delete(`/playlists/${folderId}`);
      if (selectedFolderId === folderId) {
        setSelectedFolderId('pool');
        setDisplayTracks(poolTracks);
      }
      await fetchTree();
    } catch (err) {
      console.error('Failed to delete folder:', err);
    }
  }, [selectedFolderId, poolTracks, fetchTree]);

  // ── Audio playback ────────────────────────────────────────────────────
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

  // ── Export folders to disk ─────────────────────────────────────────────
  const handleExport = useCallback(async () => {
    const playlistIds = allPlaylists.map(p => p.id);
    if (playlistIds.length === 0) return;
    try {
      const data = await apiClient.post('/playlists/export-folders', { playlist_ids: playlistIds });
      setChatFeedback(`Exported ${data.folders_created} folders, ${data.files_copied} files to ${data.output_directory}${data.errors?.length ? ` (${data.errors.length} errors)` : ''}`);
    } catch (err) {
      setChatFeedback(`Export failed: ${err.message}`);
    }
  }, [allPlaylists]);

  // ── Chat submit ───────────────────────────────────────────────────────
  const handleChatSubmit = useCallback(async (query) => {
    if (!query.trim()) return;
    setChatStatus('thinking');
    setChatFeedback('');
    try {
      const data = await apiClient.post('/playlists/organize', { query });
      setChatFeedback(data.message || 'Done');
      // Refresh everything — LLM may have created folders and assigned tracks
      await fetchTree();
      // Clear folder track cache so re-expand reloads
      setFolderTracks({});
    } catch (err) {
      setChatFeedback(`Error: ${err.message}`);
    } finally {
      setChatStatus('done');
      setChatQuery('');
    }
  }, [fetchTree]);

  // Selected folder name for header
  const selectedFolderName = selectedFolderId === 'pool'
    ? 'All Tracks'
    : allPlaylists.find(p => p.id === selectedFolderId)?.name || 'Folder';

  // ── Render ────────────────────────────────────────────────────────────
  return (
    <LazyMotion features={domAnimation}>
      <div style={{ display: 'flex', height: '100%', gap: 0 }}>

        {/* ═══════════ LEFT SIDEBAR — Folder Tree ═══════════ */}
        <GlassPanel
          depth={2}
          animate={false}
          style={{
            width: 280, minWidth: 280,
            display: 'flex', flexDirection: 'column',
            borderRadius: 0,
            borderRight: '1px solid var(--glass-border)',
          }}
        >
          {/* Purple accent bar */}
          <div style={{ height: 2, background: 'linear-gradient(90deg, #7c3aed, #00d4ff)' }} />

          {/* Header + New Folder button */}
          <div style={{
            padding: '12px 12px 8px',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          }}>
            <span style={{
              fontSize: 10, fontFamily: 'var(--font-mono)',
              letterSpacing: 2, color: '#64748b', fontWeight: 700,
            }}>
              FOLDERS
            </span>
            <m.button
              onClick={() => setShowNewFolderInput(v => !v)}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              style={{
                background: 'rgba(124,58,237,0.12)',
                border: '1px solid rgba(124,58,237,0.25)',
                borderRadius: 'var(--radius-pill)',
                padding: '3px 10px', color: '#a855f7',
                fontSize: 10, cursor: 'pointer',
                display: 'flex', alignItems: 'center', gap: 4,
                fontFamily: 'var(--font-ui)', fontWeight: 600,
              }}
            >
              <PlusIcon /> New
            </m.button>
          </div>

          {/* New folder input */}
          <AnimatePresence>
            {showNewFolderInput && (
              <m.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                style={{ overflow: 'hidden', padding: '0 12px' }}
              >
                <div style={{ display: 'flex', gap: 4, paddingBottom: 8 }}>
                  <input
                    value={newFolderName}
                    onChange={e => setNewFolderName(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter') handleCreateFolder();
                      if (e.key === 'Escape') { setShowNewFolderInput(false); setNewFolderName(''); }
                    }}
                    placeholder="Folder name..."
                    autoFocus
                    style={{
                      flex: 1, background: 'rgba(255,255,255,0.06)',
                      border: '1px solid var(--glass-border)',
                      borderRadius: 'var(--radius-sm)',
                      padding: '5px 10px', color: '#e2e8f0',
                      fontSize: 12, fontFamily: 'var(--font-ui)',
                      outline: 'none',
                    }}
                  />
                  <m.button
                    onClick={handleCreateFolder}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    style={{
                      background: '#00d4ff', color: '#0a0a14',
                      borderRadius: 'var(--radius-sm)',
                      padding: '5px 10px', fontSize: 11, fontWeight: 700,
                      border: 'none', cursor: 'pointer',
                      fontFamily: 'var(--font-ui)',
                    }}
                  >
                    Add
                  </m.button>
                </div>
              </m.div>
            )}
          </AnimatePresence>

          {/* All Tracks row */}
          <div
            onClick={() => handleSelectFolder('pool')}
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '6px 12px', margin: '0 4px', borderRadius: 4,
              cursor: 'pointer',
              background: selectedFolderId === 'pool' ? 'rgba(0,212,255,0.08)' : 'transparent',
              transition: 'background 80ms ease',
            }}
          >
            <span style={{ color: selectedFolderId === 'pool' ? '#00d4ff' : '#475569', display: 'flex' }}>
              <TrackIcon />
            </span>
            <span style={{
              flex: 1, fontSize: 12, fontWeight: selectedFolderId === 'pool' ? 600 : 500,
              color: selectedFolderId === 'pool' ? '#e2e8f0' : '#94a3b8',
              fontFamily: 'var(--font-ui)',
            }}>
              All Tracks
            </span>
            <span style={{ fontSize: 9, color: '#475569', fontFamily: 'var(--font-mono)' }}>
              {poolTracks.length}
            </span>
          </div>

          {/* Divider */}
          <div style={{ height: 1, background: 'var(--glass-border)', margin: '4px 12px' }} />

          {/* Folder tree (scrollable) */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '4px 0' }}>
            {loadError && (
              <div style={{
                padding: 12, fontSize: 10, color: '#94a3b8',
                fontFamily: 'var(--font-mono)', textAlign: 'center',
              }}>
                {loadError}
              </div>
            )}
            {tree.map(node => (
              <FolderTreeNode
                key={node.id}
                node={node}
                depth={0}
                selectedId={selectedFolderId}
                expandedIds={expandedIds}
                onToggleExpand={handleToggleExpand}
                onSelect={handleSelectFolder}
                onDrop={handleDropOnFolder}
                folderTracks={folderTracks}
              />
            ))}
            {tree.length === 0 && !loadError && (
              <div style={{
                padding: '20px 12px', textAlign: 'center',
                fontSize: 11, color: '#334155', fontFamily: 'var(--font-ui)',
              }}>
                No folders yet. Click "+ New" or use the chat to create folders.
              </div>
            )}
          </div>

          {/* Export button */}
          <div style={{ padding: '8px 12px', borderTop: '1px solid var(--glass-border)' }}>
            <m.button
              onClick={handleExport}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.97 }}
              style={{
                width: '100%',
                padding: '10px 0',
                background: 'linear-gradient(135deg, rgba(124,58,237,0.2), rgba(0,212,255,0.15))',
                border: '1px solid rgba(124,58,237,0.3)',
                borderRadius: 'var(--radius-sm)',
                color: '#e2e8f0', fontSize: 12, fontWeight: 700,
                cursor: 'pointer', fontFamily: 'var(--font-ui)',
                letterSpacing: '0.05em',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
              }}
            >
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <path d="M7 1v9M3.5 6.5L7 10l3.5-3.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M1 11v1.5h12V11" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              Export Folders to Disk
            </m.button>
          </div>
        </GlassPanel>

        {/* ═══════════ RIGHT MAIN AREA ═══════════ */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

          {/* Header bar */}
          <div style={{
            padding: '12px 20px',
            display: 'flex', alignItems: 'center', gap: 12,
            borderBottom: '1px solid var(--glass-border)',
            background: 'rgba(8,8,20,0.4)',
          }}>
            <span style={{ color: '#a855f7', display: 'flex' }}>
              {selectedFolderId === 'pool' ? <TrackIcon /> : <FolderIcon open size={16} />}
            </span>
            <span style={{
              fontSize: 14, fontWeight: 700, color: '#e2e8f0',
              fontFamily: 'var(--font-ui)',
            }}>
              {selectedFolderName}
            </span>
            <span style={{
              fontSize: 10, color: '#475569', fontFamily: 'var(--font-mono)',
            }}>
              {filteredTracks.length} tracks
            </span>

            {selectedTrackIds.size > 0 && (
              <span style={{
                fontSize: 10, color: '#00d4ff', fontFamily: 'var(--font-mono)',
                background: 'rgba(0,212,255,0.08)', borderRadius: 'var(--radius-pill)',
                padding: '2px 8px', marginLeft: 'auto',
              }}>
                {selectedTrackIds.size} selected -- drag to a folder
              </span>
            )}
          </div>

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

          {/* TRACK LIST */}
          <div style={{ flex: 1, overflowY: 'auto' }}>
            {filteredTracks.map((track, index) => (
              <TrackRow
                key={track.trackid || track.id}
                track={track}
                isSelected={selectedTrackIds.has(track.trackid || track.id)}
                isPlaying={playingTrackId === (track.trackid || track.id)}
                onSelect={(e) => handleTrackSelect(track.trackid || track.id, index, e)}
                onPlay={() => handlePlayTrack(track)}
                onDragStart={handleDragStart}
              />
            ))}

            {displayTracks.length === 0 && !loadError && (
              <div style={{ textAlign: 'center', padding: 60, color: '#475569' }}>
                <p style={{ fontSize: 14, fontFamily: 'var(--font-ui)' }}>
                  {selectedFolderId === 'pool'
                    ? 'No tracks in library'
                    : 'Empty folder -- drag tracks here from All Tracks'
                  }
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
            onSubmit={handleChatSubmit}
          />
        </div>
      </div>
    </LazyMotion>
  );
}
