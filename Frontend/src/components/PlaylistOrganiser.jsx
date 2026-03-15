// src/components/PlaylistOrganiser.jsx — Folder-based track organiser
import React, { useState, useEffect, useCallback, useMemo, useRef, memo } from 'react';
import { m, AnimatePresence } from 'framer-motion';
import { apiClient } from '../api/apiClient';
import GlassPanel from './ui/GlassPanel';

// ── SVG Icons ────────────────────────────────────────────────────────────────
const IconFolder = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
  </svg>
);

const IconFolderOpen = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M5 19a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4l2 3h9a2 2 0 0 1 2 2v1" />
    <path d="M5 12h16l-2 7H3l2-7z" />
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

const IconExport = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="17 8 12 3 7 8" />
    <line x1="12" y1="3" x2="12" y2="15" />
  </svg>
);

const IconSend = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <line x1="22" y1="2" x2="11" y2="13" />
    <polygon points="22 2 15 22 11 13 2 9 22 2" />
  </svg>
);

const IconSearch = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>
);

const IconX = () => (
  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true">
    <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
  </svg>
);

const IconPool = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
    <rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" />
    <rect x="3" y="14" width="7" height="7" /><rect x="14" y="14" width="7" height="7" />
  </svg>
);

const IconChevron = ({ open }) => (
  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"
    style={{ transform: open ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 150ms ease' }}>
    <polyline points="9 18 15 12 9 6" />
  </svg>
);

// ── Animation variants ───────────────────────────────────────────────────────
const rowVariants = {
  hidden: { opacity: 0, y: 6 },
  visible: i => ({
    opacity: 1, y: 0,
    transition: { delay: i * 0.02, type: 'spring', damping: 25, stiffness: 300 },
  }),
  exit: { opacity: 0, y: -6, transition: { duration: 0.12 } },
};

// ── FolderNode ───────────────────────────────────────────────────────────────
function FolderNode({ folder, depth = 0, selectedId, onSelect, onDrop, onRename, onDelete }) {
  const [dragOver, setDragOver] = useState(false);
  const [editing, setEditing] = useState(false);
  const [editName, setEditName] = useState(folder.name);
  const inputRef = useRef(null);
  const isSelected = selectedId === folder.id;

  useEffect(() => {
    if (editing && inputRef.current) inputRef.current.focus();
  }, [editing]);

  const handleDragOver = e => { e.preventDefault(); setDragOver(true); };
  const handleDragLeave = () => setDragOver(false);
  const handleDrop = e => {
    e.preventDefault();
    setDragOver(false);
    try {
      const trackData = JSON.parse(e.dataTransfer.getData('application/json'));
      onDrop(folder.id, trackData);
    } catch { /* invalid drag data */ }
  };

  const handleDoubleClick = e => {
    e.stopPropagation();
    setEditName(folder.name);
    setEditing(true);
  };

  const commitRename = () => {
    setEditing(false);
    if (editName.trim() && editName.trim() !== folder.name) {
      onRename(folder.id, editName.trim());
    }
  };

  return (
    <>
      <m.div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => onSelect(folder.id)}
        onDoubleClick={handleDoubleClick}
        whileHover={{ background: isSelected ? undefined : 'rgba(124,58,237,0.04)' }}
        style={{
          display: 'flex', alignItems: 'center', gap: 6,
          paddingLeft: 12 + depth * 20, paddingRight: 10,
          paddingTop: 8, paddingBottom: 8,
          borderRadius: 'var(--radius-sm)',
          cursor: 'pointer',
          background: dragOver
            ? 'rgba(0,212,255,0.08)'
            : isSelected ? 'rgba(124,58,237,0.1)' : 'transparent',
          border: isSelected
            ? '1px solid rgba(124,58,237,0.3)'
            : dragOver
              ? '1px solid rgba(0,212,255,0.3)'
              : '1px solid transparent',
          transition: 'background 120ms ease, border-color 120ms ease',
          margin: '1px 0',
        }}
      >
        <span style={{ color: isSelected ? '#a855f7' : '#475569', flexShrink: 0, display: 'flex' }}>
          {isSelected ? <IconFolderOpen /> : <IconFolder />}
        </span>

        {editing ? (
          <input
            ref={inputRef}
            value={editName}
            onChange={e => setEditName(e.target.value)}
            onBlur={commitRename}
            onKeyDown={e => { if (e.key === 'Enter') commitRename(); if (e.key === 'Escape') setEditing(false); }}
            onClick={e => e.stopPropagation()}
            style={{
              flex: 1, minWidth: 0,
              background: 'rgba(124,58,237,0.06)',
              border: '1px solid rgba(124,58,237,0.3)',
              borderRadius: 4, padding: '2px 6px',
              color: '#e2e8f0', fontSize: 11, fontWeight: 600,
              fontFamily: "'Inter', system-ui, sans-serif",
              outline: 'none',
            }}
          />
        ) : (
          <span style={{
            flex: 1, minWidth: 0,
            fontSize: 11, fontWeight: 600,
            color: isSelected ? '#e2e8f0' : '#94a3b8',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            {folder.name}
          </span>
        )}

        <span style={{
          fontSize: 9, color: '#475569',
          fontFamily: "'JetBrains Mono', monospace",
          flexShrink: 0,
        }}>
          {folder.track_count ?? 0}
        </span>

        <m.button
          whileHover={{ scale: 1.15 }}
          whileTap={{ scale: 0.9 }}
          onClick={e => { e.stopPropagation(); onDelete(folder.id); }}
          aria-label="Delete folder"
          style={{
            background: 'none', border: 'none', color: '#475569',
            cursor: 'pointer', padding: 2, display: 'flex',
            alignItems: 'center', justifyContent: 'center',
            opacity: 0.5, transition: 'opacity 120ms ease',
          }}
          whileHover={{ opacity: 1, color: '#a855f7' }}
        >
          <IconTrash />
        </m.button>
      </m.div>

      {/* Render children recursively */}
      {folder.children?.map(child => (
        <FolderNode
          key={child.id}
          folder={child}
          depth={depth + 1}
          selectedId={selectedId}
          onSelect={onSelect}
          onDrop={onDrop}
          onRename={onRename}
          onDelete={onDelete}
        />
      ))}
    </>
  );
}

// ── PlaylistTrackRow ─────────────────────────────────────────────────────────
const PlaylistTrackRow = memo(function PlaylistTrackRow({ track, index, onRemove, onDragStart }) {
  const tid = track.trackid || track.id;
  return (
    <m.div
      custom={index}
      variants={rowVariants}
      initial="hidden"
      animate="visible"
      exit="exit"
      layout
      draggable="true"
      onDragStart={e => {
        e.dataTransfer.setData('application/json', JSON.stringify({ id: tid, title: track.title, artist: track.artist }));
        onDragStart?.(track);
      }}
      style={{
        display: 'grid',
        gridTemplateColumns: '36px 1fr 1fr 72px 56px 36px',
        gap: 8, padding: '9px 12px',
        borderRadius: 'var(--radius-sm)',
        alignItems: 'center',
        cursor: 'grab',
        transition: 'background 120ms ease',
      }}
      whileHover={{ background: 'rgba(124,58,237,0.04)' }}
    >
      <span style={{ fontSize: 10, color: '#475569', fontFamily: "'JetBrains Mono', monospace" }}>
        {String(index + 1).padStart(2, '0')}
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
      <span style={{ fontSize: 11, color: '#00d4ff', fontFamily: "'JetBrains Mono', monospace" }}>
        {track.bpm ? Math.round(track.bpm) : '\u2014'}
      </span>
      <span style={{ fontSize: 11, color: '#a855f7', fontFamily: "'JetBrains Mono', monospace" }}>
        {track.key || '\u2014'}
      </span>
      <m.button
        whileHover={{ scale: 1.15, color: '#a855f7' }}
        whileTap={{ scale: 0.9 }}
        onClick={() => onRemove(tid)}
        aria-label="Remove track"
        style={{
          background: 'none', border: 'none', color: '#475569',
          cursor: 'pointer', padding: 4, display: 'flex',
          alignItems: 'center', justifyContent: 'center',
        }}
      >
        <IconX />
      </m.button>
    </m.div>
  );
});

// ── PlaylistChat ─────────────────────────────────────────────────────────────
function PlaylistChat({ chatQuery, setChatQuery, chatStatus, chatFeedback, onSubmit }) {
  const [expanded, setExpanded] = useState(true);
  const quickPrompts = ['ORGANIZE ALL', 'SUGGEST TRACKS', 'ADD GENRE TO FOLDER'];

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
          color: '#00d4ff', fontFamily: "'JetBrains Mono', monospace",
        }}>
          ORGANISER CHAT
        </span>
        <IconChevron open={expanded} />
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
                    fontFamily: "'Inter', system-ui, sans-serif",
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
                  fontFamily: "'Inter', system-ui, sans-serif",
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
                <IconSend />
              </m.button>
            </div>

            {/* Status feedback */}
            {chatFeedback && (
              <m.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                style={{
                  marginTop: 8, fontSize: 10, color: '#94a3b8',
                  fontFamily: "'JetBrains Mono', monospace",
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
  const [folders, setFolders] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [folderTracks, setFolderTracks] = useState([]);
  const [pool, setPool] = useState([]);
  const [showPool, setShowPool] = useState(true);
  const [searchFilter, setSearchFilter] = useState('');
  const [chatQuery, setChatQuery] = useState('');
  const [chatStatus, setChatStatus] = useState('idle');
  const [chatFeedback, setChatFeedback] = useState('');
  const [isExporting, setIsExporting] = useState(false);
  const [showNewFolderInput, setShowNewFolderInput] = useState(false);
  const [newFolderName, setNewFolderName] = useState('');
  const [loadError, setLoadError] = useState(null);
  const newFolderRef = useRef(null);

  // ── Data fetching ──────────────────────────────────────────────────────────
  const fetchTree = useCallback(async () => {
    try {
      const data = await apiClient.get('/playlists/tree');
      setFolders(Array.isArray(data) ? data : data.folders || data.playlists || []);
      setLoadError(null);
    } catch (err) {
      setLoadError(err.message);
    }
  }, []);

  const fetchPool = useCallback(async () => {
    try {
      const data = await apiClient.get('/playlists/pool');
      setPool(Array.isArray(data) ? data : data.tracks || []);
    } catch (err) {
      console.error('Failed to fetch pool:', err);
    }
  }, []);

  const fetchFolderTracks = useCallback(async (id) => {
    try {
      const data = await apiClient.get(`/playlists/${id}`);
      const tracks = data?.tracks || data?.playlist?.tracks || [];
      setFolderTracks(Array.isArray(tracks) ? tracks : []);
    } catch (err) {
      console.error('Failed to fetch folder tracks:', err);
      setFolderTracks([]);
    }
  }, []);

  useEffect(() => {
    fetchTree();
    fetchPool();
  }, [fetchTree, fetchPool]);

  // ── Folder selection ───────────────────────────────────────────────────────
  const handleSelectFolder = useCallback((id) => {
    setSelectedId(id);
    setShowPool(false);
    fetchFolderTracks(id);
  }, [fetchFolderTracks]);

  const handleShowPool = useCallback(() => {
    setSelectedId(null);
    setShowPool(true);
  }, []);

  // ── Create folder ──────────────────────────────────────────────────────────
  const handleCreateFolder = useCallback(async () => {
    const name = newFolderName.trim();
    if (!name) return;
    try {
      await apiClient.post('/playlists/create', { name });
      setNewFolderName('');
      setShowNewFolderInput(false);
      await fetchTree();
    } catch (err) {
      console.error('Failed to create folder:', err);
    }
  }, [newFolderName, fetchTree]);

  // ── Rename folder ──────────────────────────────────────────────────────────
  const handleRenameFolder = useCallback(async (id, name) => {
    try {
      await apiClient.put(`/playlists/${id}/rename`, { name });
      await fetchTree();
    } catch (err) {
      console.error('Failed to rename folder:', err);
    }
  }, [fetchTree]);

  // ── Delete folder ──────────────────────────────────────────────────────────
  const handleDeleteFolder = useCallback(async (id) => {
    try {
      await apiClient.delete(`/playlists/${id}`);
      if (selectedId === id) {
        setSelectedId(null);
        setShowPool(true);
        setFolderTracks([]);
      }
      await fetchTree();
    } catch (err) {
      console.error('Failed to delete folder:', err);
    }
  }, [selectedId, fetchTree]);

  // ── Drop track onto folder ─────────────────────────────────────────────────
  const handleDropOnFolder = useCallback(async (folderId, trackData) => {
    try {
      await apiClient.post(`/playlists/${folderId}/add-tracks`, { track_ids: [trackData.id] });
      await fetchTree();
      if (selectedId === folderId) await fetchFolderTracks(folderId);
    } catch (err) {
      console.error('Failed to add track to folder:', err);
    }
  }, [selectedId, fetchTree, fetchFolderTracks]);

  // ── Remove track from folder ───────────────────────────────────────────────
  const handleRemoveTrack = useCallback(async (trackId) => {
    if (!selectedId) return;
    try {
      await apiClient.post(`/playlists/${selectedId}/remove-tracks`, { track_ids: [trackId] });
      setFolderTracks(prev => prev.filter(t => (t.trackid || t.id) !== trackId));
      await fetchTree();
    } catch (err) {
      console.error('Failed to remove track:', err);
    }
  }, [selectedId, fetchTree]);

  // ── Export ─────────────────────────────────────────────────────────────────
  const handleExport = useCallback(async () => {
    if (folders.length === 0) return;
    setIsExporting(true);
    try {
      const playlistIds = folders.map(f => f.id);
      const data = await apiClient.post('/playlists/export-folders', { playlist_ids: playlistIds });
      if (data.errors?.length > 0) {
        setChatFeedback(`Exported ${data.files_copied} files to ${data.folders_created} folders. ${data.errors.length} errors.`);
      } else {
        setChatFeedback(`Exported ${data.files_copied} files to ${data.folders_created} folders at ${data.output_directory}`);
      }
      setChatStatus('done');
    } catch (err) {
      setChatFeedback('Export failed: ' + (err.message || 'unknown error'));
      setChatStatus('done');
    } finally {
      setIsExporting(false);
    }
  }, [folders]);

  // ── Chat submit ────────────────────────────────────────────────────────────
  const handleChatSubmit = useCallback(async (query) => {
    setChatStatus('thinking');
    setChatFeedback('');
    setChatQuery('');
    try {
      const data = await apiClient.post('/playlists/organize', { query });
      setChatFeedback(data.message || data.feedback || 'Done.');
      setChatStatus('done');
      await fetchTree();
      if (selectedId) await fetchFolderTracks(selectedId);
    } catch {
      setChatFeedback('Organiser endpoint not available yet.');
      setChatStatus('done');
    }
  }, [selectedId, fetchTree, fetchFolderTracks]);

  // ── Filtered tracks ────────────────────────────────────────────────────────
  const displayedTracks = useMemo(() => {
    let tracks = showPool ? pool : folderTracks;
    if (searchFilter) {
      const q = searchFilter.toLowerCase();
      tracks = tracks.filter(t =>
        t.title?.toLowerCase().includes(q) ||
        t.artist?.toLowerCase().includes(q)
      );
    }
    return tracks;
  }, [showPool, pool, folderTracks, searchFilter]);

  const selectedFolder = useMemo(() => {
    const find = (list) => {
      for (const f of list) {
        if (f.id === selectedId) return f;
        if (f.children) { const r = find(f.children); if (r) return r; }
      }
      return null;
    };
    return selectedId ? find(folders) : null;
  }, [folders, selectedId]);

  // ── Focus new-folder input ─────────────────────────────────────────────────
  useEffect(() => {
    if (showNewFolderInput && newFolderRef.current) newFolderRef.current.focus();
  }, [showNewFolderInput]);

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div style={{
      display: 'flex', height: '100%', width: '100%',
      padding: '80px 24px 24px', gap: 20,
      fontFamily: "'Inter', system-ui, sans-serif",
    }}>

      {/* ═══════════ FOLDER TREE SIDEBAR ═══════════ */}
      <GlassPanel
        depth={2}
        style={{
          width: 260, minWidth: 260,
          borderRadius: 'var(--radius-xl)',
          display: 'flex', flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        {/* Accent bar */}
        <div style={{
          height: 2,
          background: 'linear-gradient(90deg, #7c3aed, #00d4ff, rgba(0,212,255,0.1))',
        }} />

        {/* Header */}
        <div style={{
          padding: '16px 16px 12px',
          borderBottom: '1px solid var(--glass-border)',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <span style={{
            fontSize: 10, fontWeight: 700, letterSpacing: '0.2em', color: '#e2e8f0',
          }}>
            FOLDERS
          </span>
          <span style={{
            fontSize: 9, color: '#475569',
            fontFamily: "'JetBrains Mono', monospace",
          }}>
            {folders.length}
          </span>
        </div>

        {/* Pool button */}
        <div style={{ padding: '8px 8px 4px' }}>
          <m.div
            onClick={handleShowPool}
            whileHover={{ background: showPool ? undefined : 'rgba(0,212,255,0.04)' }}
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '8px 12px', borderRadius: 'var(--radius-sm)',
              cursor: 'pointer',
              background: showPool ? 'rgba(0,212,255,0.08)' : 'transparent',
              border: showPool ? '1px solid rgba(0,212,255,0.25)' : '1px solid transparent',
              transition: 'background 120ms ease, border-color 120ms ease',
            }}
          >
            <span style={{ color: showPool ? '#00d4ff' : '#475569', display: 'flex' }}>
              <IconPool />
            </span>
            <span style={{
              fontSize: 11, fontWeight: 600,
              color: showPool ? '#e2e8f0' : '#94a3b8',
            }}>
              All Tracks
            </span>
            <span style={{
              marginLeft: 'auto', fontSize: 9, color: '#475569',
              fontFamily: "'JetBrains Mono', monospace",
            }}>
              {pool.length}
            </span>
          </m.div>
        </div>

        {/* Folder tree */}
        <div style={{ flex: 1, overflow: 'auto', padding: '4px 8px 8px' }}>
          {loadError && (
            <div style={{
              padding: '12px', fontSize: 10, color: '#94a3b8',
              fontFamily: "'JetBrains Mono', monospace", textAlign: 'center',
            }}>
              {loadError}
            </div>
          )}
          {!loadError && folders.length === 0 && (
            <div style={{
              padding: '24px 12px', fontSize: 11, color: '#475569',
              textAlign: 'center', lineHeight: 1.6,
            }}>
              No folders yet. Create one to start organising tracks.
            </div>
          )}
          {folders.map(folder => (
            <FolderNode
              key={folder.id}
              folder={folder}
              depth={0}
              selectedId={selectedId}
              onSelect={handleSelectFolder}
              onDrop={handleDropOnFolder}
              onRename={handleRenameFolder}
              onDelete={handleDeleteFolder}
            />
          ))}
        </div>

        {/* New folder input */}
        <div style={{ padding: '8px 12px', borderTop: '1px solid var(--glass-border)' }}>
          <AnimatePresence>
            {showNewFolderInput && (
              <m.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                style={{ overflow: 'hidden', marginBottom: 8 }}
              >
                <div style={{ display: 'flex', gap: 6 }}>
                  <input
                    ref={newFolderRef}
                    value={newFolderName}
                    onChange={e => setNewFolderName(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter') handleCreateFolder(); if (e.key === 'Escape') setShowNewFolderInput(false); }}
                    placeholder="Folder name..."
                    style={{
                      flex: 1, padding: '7px 10px',
                      background: 'rgba(8,8,20,0.5)',
                      border: '1px solid var(--glass-border)',
                      borderRadius: 'var(--radius-sm)',
                      color: '#e2e8f0', fontSize: 11, outline: 'none',
                      fontFamily: "'Inter', system-ui, sans-serif",
                    }}
                  />
                  <m.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.97 }}
                    onClick={handleCreateFolder}
                    style={{
                      padding: '6px 10px',
                      background: 'rgba(124,58,237,0.1)',
                      border: '1px solid rgba(124,58,237,0.3)',
                      borderRadius: 'var(--radius-sm)',
                      color: '#a855f7', cursor: 'pointer',
                      fontSize: 10, fontWeight: 700,
                      fontFamily: "'Inter', system-ui, sans-serif",
                    }}
                  >
                    ADD
                  </m.button>
                </div>
              </m.div>
            )}
          </AnimatePresence>

          <m.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => setShowNewFolderInput(v => !v)}
            style={{
              width: '100%',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
              padding: '9px 0',
              background: 'rgba(124,58,237,0.06)',
              border: '1px solid rgba(124,58,237,0.25)',
              borderRadius: 'var(--radius-sm)',
              color: '#a855f7', cursor: 'pointer',
              fontSize: 10, fontWeight: 700, letterSpacing: '0.12em',
              fontFamily: "'Inter', system-ui, sans-serif",
            }}
          >
            <IconPlus />
            NEW FOLDER
          </m.button>
        </div>

        {/* Export */}
        <div style={{ padding: '8px 12px 12px' }}>
          <m.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
            onClick={handleExport}
            disabled={isExporting}
            style={{
              width: '100%',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
              padding: '9px 0',
              background: 'rgba(0,212,255,0.06)',
              border: '1px solid rgba(0,212,255,0.25)',
              borderRadius: 'var(--radius-sm)',
              color: isExporting ? '#475569' : '#00d4ff',
              cursor: isExporting ? 'wait' : 'pointer',
              fontSize: 10, fontWeight: 700, letterSpacing: '0.12em',
              fontFamily: "'Inter', system-ui, sans-serif",
            }}
          >
            <IconExport />
            {isExporting ? 'EXPORTING...' : 'EXPORT ALL'}
          </m.button>
        </div>
      </GlassPanel>

      {/* ═══════════ MAIN AREA ═══════════ */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <GlassPanel
          depth={2}
          animate={false}
          style={{
            flex: 1, display: 'flex', flexDirection: 'column',
            borderRadius: 'var(--radius-xl)',
            overflow: 'hidden', minHeight: 0,
          }}
        >
          {/* Accent bar */}
          <div style={{
            height: 2,
            background: 'linear-gradient(90deg, #7c3aed, #00d4ff, rgba(0,212,255,0.1))',
          }} />

          {/* Header + search */}
          <div style={{
            padding: '16px 20px 12px',
            borderBottom: '1px solid var(--glass-border)',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            flexShrink: 0, gap: 16,
          }}>
            <div>
              <div style={{ fontSize: 16, fontWeight: 700, color: '#e2e8f0' }}>
                {showPool ? 'All Tracks' : selectedFolder?.name || 'Select a folder'}
              </div>
              <div style={{
                fontSize: 9, color: '#475569', marginTop: 2,
                fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.1em',
              }}>
                {displayedTracks.length} TRACKS
              </div>
            </div>

            {/* Search */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: 8,
              padding: '6px 12px',
              background: 'rgba(8,8,20,0.4)',
              border: '1px solid var(--glass-border)',
              borderRadius: 'var(--radius-pill)',
              minWidth: 180, maxWidth: 280,
            }}>
              <span style={{ color: '#475569', display: 'flex' }}><IconSearch /></span>
              <input
                value={searchFilter}
                onChange={e => setSearchFilter(e.target.value)}
                placeholder="Filter tracks..."
                style={{
                  flex: 1, background: 'none', border: 'none',
                  color: '#e2e8f0', fontSize: 11, outline: 'none',
                  fontFamily: "'Inter', system-ui, sans-serif",
                }}
              />
              {searchFilter && (
                <m.button
                  whileTap={{ scale: 0.9 }}
                  onClick={() => setSearchFilter('')}
                  style={{
                    background: 'none', border: 'none', color: '#475569',
                    cursor: 'pointer', display: 'flex', padding: 0,
                  }}
                >
                  <IconX />
                </m.button>
              )}
            </div>
          </div>

          {/* Column headers */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '36px 1fr 1fr 72px 56px 36px',
            gap: 8, padding: '8px 12px 8px 24px',
            fontSize: 8, fontWeight: 600, letterSpacing: '0.2em',
            color: '#475569', fontFamily: "'JetBrains Mono', monospace",
            borderBottom: '1px solid var(--glass-border)',
            flexShrink: 0,
          }}>
            <span>#</span>
            <span>TITLE</span>
            <span>ARTIST</span>
            <span>BPM</span>
            <span>KEY</span>
            <span />
          </div>

          {/* Track list */}
          <div style={{ flex: 1, overflow: 'auto', padding: '4px 12px' }}>
            {displayedTracks.length === 0 && (
              <div style={{
                padding: '48px 0', textAlign: 'center',
                fontSize: 12, color: '#475569',
              }}>
                {showPool
                  ? 'No tracks in pool. Import tracks to get started.'
                  : 'No tracks in this folder. Drag tracks here from the pool.'}
              </div>
            )}
            <AnimatePresence>
              {displayedTracks.map((track, i) => (
                <PlaylistTrackRow
                  key={track.trackid || track.id}
                  track={track}
                  index={i}
                  onRemove={showPool ? () => {} : handleRemoveTrack}
                  onDragStart={() => {}}
                />
              ))}
            </AnimatePresence>
          </div>

          {/* Chat section */}
          <PlaylistChat
            chatQuery={chatQuery}
            setChatQuery={setChatQuery}
            chatStatus={chatStatus}
            chatFeedback={chatFeedback}
            onSubmit={handleChatSubmit}
          />
        </GlassPanel>
      </div>
    </div>
  );
}
