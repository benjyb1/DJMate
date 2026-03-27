// playlist/PlaylistSidebar.jsx — Left sidebar: LIBRARY tree + MY PLAYLISTS + Import/Export
import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { LazyMotion, domAnimation, m } from 'framer-motion';
import { apiClient } from '../../api/apiClient';
import GlassPanel from '../ui/GlassPanel';
import { buildFileTree } from './helpers';
import { useAuthStore } from '../../stores/authStore';
import { isFileSystemAccessSupported, pickMusicFolder } from '../../utils/localAudio';

// ── SVG icons ────────────────────────────────────────────────────────────────
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

const UploadIcon = () => (
  <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
    <path d="M7 10V1M3.5 4.5L7 1l3.5 3.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M1 11v1.5h12V11" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

// ── FolderTreeNode (playlist folders, recursive, drop target) ────────────────
function FolderTreeNode({
  node, depth, selectedId, expandedIds, onToggleExpand, onSelect,
  onDrop, folderTracks, onDelete,
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
          borderRadius: 4, margin: '0 4px',
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
        {/* Delete button on hover */}
        {onDelete && (
          <m.button
            onClick={(e) => { e.stopPropagation(); onDelete(node.id); }}
            whileHover={{ scale: 1.1 }}
            style={{
              background: 'none', border: 'none', color: '#475569', cursor: 'pointer',
              padding: 0, display: 'flex', opacity: 0, transition: 'opacity 100ms ease',
            }}
            className="folder-delete-btn"
          >
            <svg width="10" height="10" viewBox="0 0 12 12"><path d="M2 2l8 8M10 2l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
          </m.button>
        )}
      </div>

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
              onDelete={onDelete}
            />
          ))}
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

// ── PlaylistEntry (flat list item for MY PLAYLISTS, drop target) ─────────────
function PlaylistEntry({ playlist, isSelected, onSelect, onDrop, onDelete }) {
  const [dragOver, setDragOver] = useState(false);

  const handleDragOver = (e) => { e.preventDefault(); e.stopPropagation(); setDragOver(true); };
  const handleDragLeave = (e) => { e.stopPropagation(); setDragOver(false); };
  const handleDrop = (e) => { e.stopPropagation(); setDragOver(false); onDrop(playlist.id, e); };

  return (
    <div
      onClick={() => onSelect(playlist.id)}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      style={{
        display: 'flex', alignItems: 'center', gap: 6,
        padding: '6px 12px', margin: '0 4px', borderRadius: 4,
        cursor: 'pointer',
        background: dragOver
          ? 'rgba(0,212,255,0.12)'
          : isSelected
            ? 'rgba(124,58,237,0.1)'
            : 'transparent',
        border: dragOver ? '1px solid rgba(0,212,255,0.4)' : '1px solid transparent',
        transition: 'background 80ms ease, border-color 80ms ease',
        minHeight: 26,
      }}
    >
      <span style={{ color: isSelected ? '#a855f7' : '#64748b', display: 'flex' }}>
        <FolderIcon open={false} />
      </span>
      <span style={{
        flex: 1, minWidth: 0, fontSize: 12, fontWeight: isSelected ? 600 : 500,
        color: isSelected ? '#e2e8f0' : '#94a3b8',
        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        fontFamily: 'var(--font-ui)',
      }}>
        {playlist.name}
      </span>
      {(playlist.track_count > 0 || playlist.track_count === 0) && (
        <span style={{
          fontSize: 9, color: '#475569', fontFamily: 'var(--font-mono)',
          flexShrink: 0, marginLeft: 4,
        }}>
          {playlist.track_count ?? 0}
        </span>
      )}
      {onDelete && (
        <m.button
          onClick={(e) => { e.stopPropagation(); onDelete(playlist.id); }}
          whileHover={{ scale: 1.1 }}
          style={{
            background: 'none', border: 'none', color: '#475569', cursor: 'pointer',
            padding: 0, display: 'flex', opacity: 0, transition: 'opacity 100ms ease',
          }}
          className="folder-delete-btn"
        >
          <svg width="10" height="10" viewBox="0 0 12 12"><path d="M2 2l8 8M10 2l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
        </m.button>
      )}
    </div>
  );
}

// ── LibraryTreeNode (read-only source tree from filepaths, drag FROM) ────────
function LibraryTreeNode({ node, depth, selectedId, expandedIds, onToggleExpand, onSelect, tracksByPath }) {
  const isExpanded = expandedIds.has(node.id);
  const isSelected = selectedId === node.id;
  const hasChildren = node.children && node.children.length > 0;
  const directTracks = tracksByPath[node.path] || [];

  return (
    <div>
      <div
        onClick={() => { onToggleExpand(node.id); onSelect(node.id, node.path); }}
        style={{
          display: 'flex', alignItems: 'center', gap: 4,
          padding: '4px 8px 4px ' + (8 + depth * 16) + 'px',
          cursor: 'pointer',
          background: isSelected ? 'rgba(0,212,255,0.08)' : 'transparent',
          borderRadius: 4, margin: '0 4px',
          transition: 'background 80ms ease',
          minHeight: 26,
        }}
      >
        <span style={{ color: '#64748b', display: 'flex', width: 10 }}>
          {(hasChildren || directTracks.length > 0) && <ChevronIcon open={isExpanded} />}
        </span>
        <span style={{ color: isSelected ? '#00d4ff' : '#64748b', display: 'flex' }}>
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
        {node.track_count > 0 && (
          <span style={{ fontSize: 9, color: '#475569', fontFamily: 'var(--font-mono)', flexShrink: 0, marginLeft: 4 }}>
            {node.track_count}
          </span>
        )}
      </div>

      {isExpanded && (
        <div>
          {node.children.map(child => (
            <LibraryTreeNode
              key={child.id}
              node={child}
              depth={depth + 1}
              selectedId={selectedId}
              expandedIds={expandedIds}
              onToggleExpand={onToggleExpand}
              onSelect={onSelect}
              tracksByPath={tracksByPath}
            />
          ))}
          {directTracks.map(t => (
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
        </div>
      )}
    </div>
  );
}


// ── Main Sidebar Component ───────────────────────────────────────────────────
export default function PlaylistSidebar({
  allPlaylists,
  poolTracks,
  poolLoading,
  activePlaylistId,
  onSelectPlaylist,
  onCreatePlaylist,
  onDeletePlaylist,
  onDropOnPlaylist,
  onExport,
  onStartIngest,
  onIngestComplete,
  fetchTree,
  fetchPool,
}) {
  const profile = useAuthStore(s => s.profile);

  // Library file tree
  const { roots: fileTreeRoots, tracksByPath } = useMemo(
    () => buildFileTree(poolTracks), [poolTracks]
  );

  // Expand/collapse state
  const [libExpandedIds, setLibExpandedIds] = useState(new Set());

  // Ingest
  const [showIngest, setShowIngest] = useState(false);
  const [ingestFolder, setIngestFolder] = useState('');
  const [ingestStatus, setIngestStatus] = useState('idle');
  const [ingestLog, setIngestLog] = useState([]);
  const ingestPollRef = useRef(null);

  const [loadError, setLoadError] = useState(null);

  // ── Library tree toggle/select ────────────────────────────────────────
  const handleLibToggleExpand = useCallback((nodeId) => {
    setLibExpandedIds(prev => {
      const next = new Set(prev);
      if (next.has(nodeId)) next.delete(nodeId);
      else next.add(nodeId);
      return next;
    });
  }, []);

  // When clicking a library folder, we don't change activePlaylistId
  // Library is source-only for dragging
  const handleLibSelectFolder = useCallback(() => {
    // No-op for now — library folders are source-only, not navigation targets
  }, []);

  // ── Drop on playlist entry ───────────────────────────────────────────
  const handleDropOnFolder = useCallback(async (folderId, e) => {
    e.preventDefault();
    try {
      const raw = e.dataTransfer.getData('application/json');
      if (!raw) return;
      const trackData = JSON.parse(raw);
      const trackIds = Array.isArray(trackData) ? trackData.map(t => t.id) : [trackData.id];

      await apiClient.post(`/playlists/${folderId}/add-tracks`, { track_ids: trackIds });
      if (fetchTree) await fetchTree();
    } catch (err) {
      console.error('Drop failed:', err);
    }
  }, [fetchTree]);

  // ── Ingest ────────────────────────────────────────────────────────────
  const startIngest = useCallback(async (folder) => {
    const path = (folder || ingestFolder).trim();
    if (!path) return;
    try {
      setIngestStatus('running');
      setIngestLog([]);
      await apiClient.post('/ingest/start', { folder: path });
      let linesSeen = 0;
      ingestPollRef.current = setInterval(async () => {
        try {
          const data = await apiClient.get(`/ingest/status`, { since: linesSeen });
          if (data.log_lines?.length) {
            setIngestLog(prev => [...prev, ...data.log_lines]);
            linesSeen = data.total_lines;
          }
          if (data.status === 'done' || data.status === 'error') {
            clearInterval(ingestPollRef.current);
            ingestPollRef.current = null;
            setIngestStatus(data.status);
            if (data.status === 'done') {
              apiClient.clearCache();
              if (fetchPool) fetchPool();
              if (fetchTree) fetchTree();
              useAuthStore.getState().setImportedLibrary();
              if (onIngestComplete) onIngestComplete();
            }
          }
        } catch { /* ignore poll errors */ }
      }, 1500);
    } catch (err) {
      setIngestStatus('error');
      setIngestLog(prev => [...prev, `Failed to start: ${err.message}`]);
    }
  }, [ingestFolder, fetchPool, fetchTree, onIngestComplete]);

  useEffect(() => {
    return () => { if (ingestPollRef.current) clearInterval(ingestPollRef.current); };
  }, []);

  return (
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

      {/* ── LIBRARY section ──────────────────────────────────────── */}
      <div style={{
        padding: '12px 12px 4px',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <span style={{
          fontSize: 10, fontFamily: 'var(--font-mono)',
          letterSpacing: 2, color: '#64748b', fontWeight: 700,
        }}>
          LIBRARY
        </span>
        <span style={{ fontSize: 9, color: '#475569', fontFamily: 'var(--font-mono)', display: 'flex', alignItems: 'center', gap: 6 }}>
          {poolLoading && (
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none" style={{ animation: 'spin 1s linear infinite' }}>
              <circle cx="8" cy="8" r="6" stroke="rgba(124,58,237,0.25)" strokeWidth="2" />
              <path d="M14 8a6 6 0 0 0-6-6" stroke="#a855f7" strokeWidth="2" strokeLinecap="round" />
            </svg>
          )}
          {poolTracks.length}
        </span>
      </div>

      {/* All Tracks row */}
      <div
        style={{
          display: 'flex', alignItems: 'center', gap: 6,
          padding: '6px 12px', margin: '0 4px', borderRadius: 4,
          cursor: 'default',
          transition: 'background 80ms ease',
        }}
      >
        <span style={{ color: '#475569', display: 'flex' }}>
          <TrackIcon />
        </span>
        <span style={{
          flex: 1, fontSize: 12, fontWeight: 500,
          color: '#94a3b8', fontFamily: 'var(--font-ui)',
        }}>
          All Tracks
        </span>
        <span style={{ fontSize: 9, color: '#475569', fontFamily: 'var(--font-mono)' }}>
          {poolTracks.length}
        </span>
      </div>

      {/* Library folder tree */}
      <div style={{ overflowY: 'auto', padding: '2px 0', maxHeight: '40vh' }}>
        {poolLoading && poolTracks.length === 0 && (
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            gap: 8, padding: '16px 12px',
          }}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style={{ animation: 'spin 1s linear infinite' }}>
              <circle cx="8" cy="8" r="6" stroke="rgba(124,58,237,0.25)" strokeWidth="2" />
              <path d="M14 8a6 6 0 0 0-6-6" stroke="#a855f7" strokeWidth="2" strokeLinecap="round" />
            </svg>
            <span style={{ fontSize: 10, color: '#64748b', fontFamily: 'var(--font-mono)' }}>
              Loading library...
            </span>
          </div>
        )}
        {fileTreeRoots.map(node => (
          <LibraryTreeNode
            key={node.id}
            node={node}
            depth={0}
            selectedId={null}
            expandedIds={libExpandedIds}
            onToggleExpand={handleLibToggleExpand}
            onSelect={handleLibSelectFolder}
            tracksByPath={tracksByPath}
          />
        ))}
        {!poolLoading && poolTracks.length > 0 && fileTreeRoots.length === 0 && (
          <div style={{
            padding: '8px 12px', fontSize: 10, color: '#475569',
            fontFamily: 'var(--font-ui)', fontStyle: 'italic',
          }}>
            No folder structure found in track filepaths
          </div>
        )}
      </div>

      {/* ── Divider ──────────────────────────────────────────────── */}
      <div style={{ height: 1, background: 'var(--glass-border)', margin: '6px 12px' }} />

      {/* ── MY PLAYLISTS section ─────────────────────────────────── */}
      <div style={{
        padding: '8px 12px 4px',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <span style={{
          fontSize: 10, fontFamily: 'var(--font-mono)',
          letterSpacing: 2, color: '#64748b', fontWeight: 700,
        }}>
          MY PLAYLISTS
        </span>
        <m.button
          onClick={onCreatePlaylist}
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

      {/* Playlist flat list (scrollable) */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '4px 0' }}>
        {allPlaylists.map(pl => (
          <PlaylistEntry
            key={pl.id}
            playlist={pl}
            isSelected={activePlaylistId === pl.id}
            onSelect={onSelectPlaylist}
            onDrop={handleDropOnFolder}
            onDelete={onDeletePlaylist}
          />
        ))}
        {allPlaylists.length === 0 && (
          <div style={{
            padding: '12px 12px', textAlign: 'center',
            fontSize: 10, color: '#334155', fontFamily: 'var(--font-ui)', fontStyle: 'italic',
          }}>
            No playlists yet -- use chat or + New
          </div>
        )}
      </div>

      {/* ── Import Music Folder ──────────────────────────────────── */}
      <div style={{ padding: '8px 12px', borderTop: '1px solid var(--glass-border)' }}>
        <m.button
          onClick={async () => {
            if (!isFileSystemAccessSupported()) return;
            try {
              await pickMusicFolder();
            } catch { /* user cancelled */ }
          }}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.97 }}
          style={{
            width: '100%', padding: '10px 0',
            background: 'linear-gradient(135deg, rgba(0,212,255,0.12), rgba(124,58,237,0.08))',
            border: '1px solid rgba(0,212,255,0.25)',
            borderRadius: 'var(--radius-sm)',
            color: '#e2e8f0', fontSize: 12, fontWeight: 700,
            cursor: 'pointer', fontFamily: 'var(--font-ui)',
            letterSpacing: '0.05em',
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
          }}
        >
          <UploadIcon />
          {profile?.has_imported_library ? 'Re-scan Library' : 'Import Music Folder'}
        </m.button>
      </div>

      {/* Export button */}
      <div style={{ padding: '8px 12px', borderTop: '1px solid var(--glass-border)' }}>
        <m.button
          onClick={onExport}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.97 }}
          style={{
            width: '100%', padding: '10px 0',
            background: 'linear-gradient(135deg, rgba(124,58,237,0.2), rgba(0,212,255,0.15))',
            border: '1px solid rgba(124,58,237,0.3)', borderRadius: 'var(--radius-sm)',
            color: '#e2e8f0', fontSize: 12, fontWeight: 700,
            cursor: 'pointer', fontFamily: 'var(--font-ui)', letterSpacing: '0.05em',
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
  );
}
