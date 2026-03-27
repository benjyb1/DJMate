// playlist/PlaylistOrganiser.jsx — Layout shell: sidebar + split main area (active playlist | suggestions | chat)
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { LazyMotion, domAnimation } from 'framer-motion';
import { apiClient } from '../../api/apiClient';
import { supabase } from '../../utils/supabaseClient';
import { getLocalAudioUrl, hasMusicFolder } from '../../utils/localAudio';
import { useAudioPreloader } from '../../hooks/useAudioPreloader';
import PlaylistSidebar from './PlaylistSidebar';
import ActivePlaylistPanel from './ActivePlaylistPanel';
import SuggestionsPanel from './SuggestionsPanel';
import SuggestedPlaylistsPanel from './SuggestedPlaylistsPanel';
import PlaylistChatBar from './PlaylistChatBar';

export default function PlaylistOrganiser({ onIngestComplete }) {
  // ── Data ──────────────────────────────────────────────────────────────
  const [allPlaylists, setAllPlaylists] = useState([]);
  const [poolTracks, setPoolTracks] = useState([]);
  const [poolLoading, setPoolLoading] = useState(true);

  // Active playlist (shown in top half)
  const [activePlaylistId, setActivePlaylistId] = useState(null);
  const [activePlaylistName, setActivePlaylistName] = useState(null);
  const [activePlaylistTracks, setActivePlaylistTracks] = useState([]);
  const [isUnsavedPlaylist, setIsUnsavedPlaylist] = useState(false);

  // LLM suggestions (shown in bottom half)
  const [suggestions, setSuggestions] = useState(null);
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);

  // Audio — preload URLs for the active playlist so playback is instant
  const audioRef = useRef(new Audio());
  const [playingTrackId, setPlayingTrackId] = useState(null);
  const { getUrl: getPreloadedUrl } = useAudioPreloader(activePlaylistTracks);

  // Chat
  const [chatQuery, setChatQuery] = useState('');
  const [chatStatus, setChatStatus] = useState('idle');
  const [chatFeedback, setChatFeedback] = useState('');

  // UI
  const [searchFilter, setSearchFilter] = useState('');

  // ── Data fetching ─────────────────────────────────────────────────────
  const fetchTree = useCallback(async () => {
    try {
      const data = await apiClient.get('/playlists/tree');
      const list = Array.isArray(data) ? data : data.playlists || [];
      setAllPlaylists(list);
    } catch (err) {
      console.error('Failed to fetch tree:', err);
    }
  }, []);

  const fetchPool = useCallback(async () => {
    try {
      setPoolLoading(true);
      // Fetch directly from Supabase (faster than going through backend)
      if (supabase) {
        const { data, error } = await supabase
          .from('tracks')
          .select('trackid, title, artist, bpm, key, filepath, track_labels(energy, semantic_tags, vibe)');
        if (!error && data) {
          const tracks = data.map(row => {
            const labels = row.track_labels || {};
            return {
              trackid: row.trackid,
              title: row.title || '',
              artist: row.artist || '',
              bpm: row.bpm,
              key: row.key,
              filepath: row.filepath || '',
              energy: labels.energy,
              semantic_tags: labels.semantic_tags,
              vibe: labels.vibe,
            };
          });
          setPoolTracks(tracks);
          return;
        }
      }
      // Fallback to backend API
      const data = await apiClient.get('/playlists/pool');
      const tracks = Array.isArray(data) ? data : data.tracks || [];
      setPoolTracks(tracks);
    } catch (err) {
      console.error('Failed to fetch pool:', err);
    } finally {
      setPoolLoading(false);
    }
  }, []);

  useEffect(() => { fetchTree(); fetchPool(); }, [fetchTree, fetchPool]);

  // Audio cleanup
  useEffect(() => {
    const audio = audioRef.current;
    const handleEnded = () => setPlayingTrackId(null);
    audio.addEventListener('ended', handleEnded);
    return () => { audio.pause(); audio.removeEventListener('ended', handleEnded); };
  }, []);

  // ── Load active playlist tracks ───────────────────────────────────────
  const loadPlaylistTracks = useCallback(async (playlistId) => {
    try {
      const data = await apiClient.get(`/playlists/${playlistId}`);
      setActivePlaylistTracks(data.tracks || []);
      setActivePlaylistName(data.name || 'Playlist');
    } catch (err) {
      console.error('Failed to load playlist:', err);
      setActivePlaylistTracks([]);
    }
  }, []);

  // ── Select playlist (from sidebar) ────────────────────────────────────
  const handleSelectPlaylist = useCallback((playlistId) => {
    setIsUnsavedPlaylist(false);
    setActivePlaylistId(playlistId);
    setSearchFilter('');
    loadPlaylistTracks(playlistId);
  }, [loadPlaylistTracks]);

  // ── Create unsaved playlist (local only, no API call) ────────────────
  const handleCreateUnsavedPlaylist = useCallback(() => {
    setIsUnsavedPlaylist(true);
    setActivePlaylistId(null);
    setActivePlaylistName('');
    setActivePlaylistTracks([]);
  }, []);

  // ── Save unsaved playlist to backend ────────────────────────────────
  const handleSavePlaylist = useCallback(async (name) => {
    if (!name || !name.trim()) return;
    try {
      const result = await apiClient.post('/playlists/create', { name: name.trim() });
      if (result?.id) {
        if (activePlaylistTracks.length > 0) {
          const trackIds = activePlaylistTracks.map(t => t.trackid || t.id);
          await apiClient.post(`/playlists/${result.id}/add-tracks`, { track_ids: trackIds });
        }
        setIsUnsavedPlaylist(false);
        setActivePlaylistId(result.id);
        setActivePlaylistName(name.trim());
        await fetchTree();
      }
    } catch (err) {
      console.error('Failed to save playlist:', err);
    }
  }, [activePlaylistTracks, fetchTree]);

  // ── Delete playlist ───────────────────────────────────────────────────
  const handleDeletePlaylist = useCallback(async (playlistId) => {
    try {
      await apiClient.delete(`/playlists/${playlistId}`);
      if (activePlaylistId === playlistId) {
        setActivePlaylistId(null);
        setActivePlaylistName(null);
        setActivePlaylistTracks([]);
      }
      await fetchTree();
    } catch (err) {
      console.error('Failed to delete playlist:', err);
    }
  }, [activePlaylistId, fetchTree]);

  // ── Rename playlist ───────────────────────────────────────────────────
  const handleRenamePlaylist = useCallback(async (playlistId, newName) => {
    try {
      await apiClient.put(`/playlists/${playlistId}/rename`, { name: newName });
      setActivePlaylistName(newName);
      await fetchTree();
    } catch (err) {
      console.error('Failed to rename playlist:', err);
    }
  }, [fetchTree]);

  // ── Drop tracks onto active playlist ──────────────────────────────────
  const handleDropTracksOnActive = useCallback(async (trackIds) => {
    if (isUnsavedPlaylist) {
      // Append locally, deduplicate by trackid
      const existingIds = new Set(activePlaylistTracks.map(t => t.trackid || t.id));
      const newTracks = trackIds
        .filter(id => !existingIds.has(id))
        .map(id => {
          const found = poolTracks.find(t => (t.trackid || t.id) === id);
          return found || { trackid: id, title: '', artist: '' };
        });
      if (newTracks.length > 0) {
        setActivePlaylistTracks(prev => [...prev, ...newTracks]);
      }
      return;
    }
    if (!activePlaylistId) return;
    try {
      await apiClient.post(`/playlists/${activePlaylistId}/add-tracks`, { track_ids: trackIds });
      await loadPlaylistTracks(activePlaylistId);
      await fetchTree();
    } catch (err) {
      console.error('Failed to add tracks:', err);
    }
  }, [isUnsavedPlaylist, activePlaylistId, activePlaylistTracks, poolTracks, loadPlaylistTracks, fetchTree]);

  // ── Remove track from active playlist ─────────────────────────────────
  const handleRemoveTrack = useCallback(async (trackId) => {
    if (isUnsavedPlaylist) {
      setActivePlaylistTracks(prev => prev.filter(t => (t.trackid || t.id) !== trackId));
      return;
    }
    if (!activePlaylistId) return;
    try {
      await apiClient.post(`/playlists/${activePlaylistId}/remove-tracks`, { track_ids: [trackId] });
      await loadPlaylistTracks(activePlaylistId);
      await fetchTree();
    } catch (err) {
      console.error('Failed to remove track:', err);
    }
  }, [isUnsavedPlaylist, activePlaylistId, loadPlaylistTracks, fetchTree]);

  // ── Audio playback (uses preloaded URLs for instant start) ───────────
  const API_BASE = import.meta.env.VITE_API_URL
    || (import.meta.env.DEV ? 'http://localhost:8000' : 'https://djmate.onrender.com');

  const handlePlayTrack = useCallback(async (track) => {
    const trackId = track.trackid || track.id;

    // Toggle off if already playing this track
    if (playingTrackId === trackId) {
      audioRef.current.pause();
      setPlayingTrackId(null);
      return;
    }

    // 1. Try preloaded cache first (instant)
    let url = getPreloadedUrl(trackId);

    // 2. Fallback: resolve on-demand via local filesystem
    if (!url && track.filepath) {
      url = await getLocalAudioUrl(track.filepath);
    }

    // 3. Last resort: backend streaming endpoint
    if (!url) {
      url = `${API_BASE}/tracks/${trackId}/audio`;
    }

    audioRef.current.src = url;
    audioRef.current.play().catch(() => setPlayingTrackId(null));
    setPlayingTrackId(trackId);
  }, [playingTrackId, getPreloadedUrl]);

  // ── Export folders to disk ────────────────────────────────────────────
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

  // ── Create playlist from suggestion group ─────────────────────────────
  const handleCreatePlaylistFromSuggestion = useCallback(async (name, trackIds) => {
    try {
      const result = await apiClient.post('/playlists/create', { name });
      if (result?.id) {
        await apiClient.post(`/playlists/${result.id}/add-tracks`, { track_ids: trackIds });
        await fetchTree();
        setActivePlaylistId(result.id);
        setIsUnsavedPlaylist(false);
        await loadPlaylistTracks(result.id);
        // Do NOT clear suggestions — keep them visible
      }
    } catch (err) {
      console.error('Failed to create playlist from suggestions:', err);
    }
  }, [fetchTree, loadPlaylistTracks]);

  // ── Add single suggestion track to active playlist ────────────────────
  const handleAddSuggestionTrack = useCallback(async (trackId) => {
    if (isUnsavedPlaylist) {
      const existingIds = new Set(activePlaylistTracks.map(t => t.trackid || t.id));
      if (!existingIds.has(trackId)) {
        const found = poolTracks.find(t => (t.trackid || t.id) === trackId);
        const track = found || { trackid: trackId, title: '', artist: '' };
        setActivePlaylistTracks(prev => [...prev, track]);
      }
      return;
    }
    if (!activePlaylistId) return;
    try {
      await apiClient.post(`/playlists/${activePlaylistId}/add-tracks`, { track_ids: [trackId] });
      await loadPlaylistTracks(activePlaylistId);
      await fetchTree();
    } catch (err) {
      console.error('Failed to add track:', err);
    }
  }, [isUnsavedPlaylist, activePlaylistId, activePlaylistTracks, poolTracks, loadPlaylistTracks, fetchTree]);

  // ── Add all suggestion tracks to active playlist ──────────────────────
  const handleAddAllSuggestions = useCallback(async (trackIds) => {
    if (isUnsavedPlaylist) {
      const existingIds = new Set(activePlaylistTracks.map(t => t.trackid || t.id));
      const newTracks = trackIds
        .filter(id => !existingIds.has(id))
        .map(id => {
          const found = poolTracks.find(t => (t.trackid || t.id) === id);
          return found || { trackid: id, title: '', artist: '' };
        });
      if (newTracks.length > 0) {
        setActivePlaylistTracks(prev => [...prev, ...newTracks]);
      }
      return;
    }
    if (!activePlaylistId) return;
    try {
      await apiClient.post(`/playlists/${activePlaylistId}/add-tracks`, { track_ids: trackIds });
      await loadPlaylistTracks(activePlaylistId);
      await fetchTree();
    } catch (err) {
      console.error('Failed to add tracks:', err);
    }
  }, [isUnsavedPlaylist, activePlaylistId, activePlaylistTracks, poolTracks, loadPlaylistTracks, fetchTree]);

  // ── Chat submit ───────────────────────────────────────────────────────
  const handleChatSubmit = useCallback(async (query) => {
    if (!query.trim()) return;
    setChatStatus('thinking');
    setChatFeedback('');
    setChatQuery('');
    setSuggestionsLoading(true);

    try {
      // Try suggest-tracks endpoint first (preview mode)
      let data;
      let usedSuggest = false;
      try {
        data = await apiClient.post('/playlists/suggest-tracks', {
          query,
          playlist_id: activePlaylistId || undefined,
        });
        usedSuggest = true;
      } catch (suggestErr) {
        // Endpoint not deployed yet — fall back to organize (executes immediately)
        data = await apiClient.post('/playlists/organize', { query });
      }

      if (usedSuggest) {
        // Set suggestions for bottom panel (preview mode)
        setSuggestions({
          mode: 'tracks',
          name: data.name || null,
          tracks: data.tracks || [],
        });
        setChatFeedback(data.message || `Found ${(data.tracks || []).length} tracks`);
      } else {
        // Organize executed directly — refresh sidebar
        setChatFeedback(data.message || 'Done');
        await fetchTree();
        const newTree = await apiClient.get('/playlists/tree');
        const newList = Array.isArray(newTree) ? newTree : newTree.playlists || [];
        setAllPlaylists(newList);
      }
    } catch (err) {
      setChatFeedback(`Error: ${err.message}`);
    } finally {
      setChatStatus('done');
      setSuggestionsLoading(false);
    }
  }, [activePlaylistId, fetchTree]);

  // ── Render ────────────────────────────────────────────────────────────
  return (
    <LazyMotion features={domAnimation}>
      <div style={{ display: 'flex', height: '100%', gap: 0, paddingTop: 60 }}>

        {/* ═══════════ LEFT SIDEBAR ═══════════ */}
        <PlaylistSidebar
          allPlaylists={allPlaylists}
          poolTracks={poolTracks}
          poolLoading={poolLoading}
          activePlaylistId={activePlaylistId}
          onSelectPlaylist={handleSelectPlaylist}
          onCreateUnsavedPlaylist={handleCreateUnsavedPlaylist}
          onDeletePlaylist={handleDeletePlaylist}
          onDropOnPlaylist={handleDropTracksOnActive}
          onExport={handleExport}
          onIngestComplete={onIngestComplete}
          fetchTree={fetchTree}
          fetchPool={fetchPool}
        />

        {/* ═══════════ MAIN AREA (always: workspace top + chat bottom) ═══════════ */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

          {/* TOP — Active Playlist Workspace */}
          <ActivePlaylistPanel
            playlist={activePlaylistId ? { id: activePlaylistId, name: activePlaylistName } : isUnsavedPlaylist ? { id: null, name: activePlaylistName } : null}
            tracks={activePlaylistTracks}
            playingTrackId={playingTrackId}
            onPlay={handlePlayTrack}
            onDropTracks={handleDropTracksOnActive}
            onRemoveTrack={handleRemoveTrack}
            onCreatePlaylist={handleCreateUnsavedPlaylist}
            onRenamePlaylist={handleRenamePlaylist}
            isUnsavedPlaylist={isUnsavedPlaylist}
            onSavePlaylist={handleSavePlaylist}
            searchFilter={searchFilter}
            onSearchChange={setSearchFilter}
          />

          {/* BOTTOM — LLM Suggestions from chat */}
          <SuggestionsPanel
            suggestions={suggestions}
            loading={suggestionsLoading}
            activePlaylistName={activePlaylistName}
            onAddTrack={handleAddSuggestionTrack}
            onAddAll={handleAddAllSuggestions}
            playingTrackId={playingTrackId}
            onPlay={handlePlayTrack}
          />

          {/* CHAT BAR (fixed at bottom) */}
          <PlaylistChatBar
            query={chatQuery}
            onQueryChange={setChatQuery}
            status={chatStatus}
            feedback={chatFeedback}
            onSubmit={handleChatSubmit}
          />
        </div>

        {/* ═══════════ RIGHT SIDEBAR — AI Suggested Playlists ═══════════ */}
        <div style={{
          width: 280, flexShrink: 0, overflow: 'hidden',
          borderLeft: '1px solid var(--glass-border)',
          display: 'flex', flexDirection: 'column',
        }}>
          <SuggestedPlaylistsPanel
            onCreatePlaylist={handleCreatePlaylistFromSuggestion}
            playingTrackId={playingTrackId}
            onPlay={handlePlayTrack}
            fetchTree={fetchTree}
          />
        </div>
      </div>
    </LazyMotion>
  );
}
