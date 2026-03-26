// playlist/PlaylistOrganiser.jsx — Layout shell: sidebar + split main area (active playlist | suggestions | chat)
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { LazyMotion, domAnimation } from 'framer-motion';
import { apiClient } from '../../api/apiClient';
import PlaylistSidebar from './PlaylistSidebar';
import ActivePlaylistPanel from './ActivePlaylistPanel';
import SuggestionsPanel from './SuggestionsPanel';
import PlaylistChatBar from './PlaylistChatBar';

export default function PlaylistOrganiser() {
  // ── Data ──────────────────────────────────────────────────────────────
  const [allPlaylists, setAllPlaylists] = useState([]);
  const [poolTracks, setPoolTracks] = useState([]);

  // Active playlist (shown in top half)
  const [activePlaylistId, setActivePlaylistId] = useState(null);
  const [activePlaylistName, setActivePlaylistName] = useState(null);
  const [activePlaylistTracks, setActivePlaylistTracks] = useState([]);

  // LLM suggestions (shown in bottom half)
  const [suggestions, setSuggestions] = useState(null);
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);

  // Audio
  const audioRef = useRef(new Audio());
  const [playingTrackId, setPlayingTrackId] = useState(null);

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
      const data = await apiClient.get('/playlists/pool');
      const tracks = Array.isArray(data) ? data : data.tracks || [];
      setPoolTracks(tracks);
    } catch (err) {
      console.error('Failed to fetch pool:', err);
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
    setActivePlaylistId(playlistId);
    setSearchFilter('');
    loadPlaylistTracks(playlistId);
  }, [loadPlaylistTracks]);

  // ── Create empty playlist ─────────────────────────────────────────────
  const handleCreatePlaylist = useCallback(async () => {
    // This will be triggered by sidebar's + New or top half empty state
    // The sidebar handles its own input UI; this is a fallback for top half
    try {
      const result = await apiClient.post('/playlists/create', { name: 'New Playlist' });
      await fetchTree();
      if (result?.id) {
        setActivePlaylistId(result.id);
        loadPlaylistTracks(result.id);
      }
    } catch (err) {
      console.error('Failed to create playlist:', err);
    }
  }, [fetchTree, loadPlaylistTracks]);

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
    if (!activePlaylistId) return;
    try {
      await apiClient.post(`/playlists/${activePlaylistId}/add-tracks`, { track_ids: trackIds });
      await loadPlaylistTracks(activePlaylistId);
      await fetchTree();
    } catch (err) {
      console.error('Failed to add tracks:', err);
    }
  }, [activePlaylistId, loadPlaylistTracks, fetchTree]);

  // ── Remove track from active playlist ─────────────────────────────────
  const handleRemoveTrack = useCallback(async (trackId) => {
    if (!activePlaylistId) return;
    try {
      await apiClient.post(`/playlists/${activePlaylistId}/remove-tracks`, { track_ids: [trackId] });
      await loadPlaylistTracks(activePlaylistId);
      await fetchTree();
    } catch (err) {
      console.error('Failed to remove track:', err);
    }
  }, [activePlaylistId, loadPlaylistTracks, fetchTree]);

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
        await loadPlaylistTracks(result.id);
        setSuggestions(null); // Clear suggestions after creating
      }
    } catch (err) {
      console.error('Failed to create playlist from suggestions:', err);
    }
  }, [fetchTree, loadPlaylistTracks]);

  // ── Add single suggestion track to active playlist ────────────────────
  const handleAddSuggestionTrack = useCallback(async (trackId) => {
    if (!activePlaylistId) return;
    try {
      await apiClient.post(`/playlists/${activePlaylistId}/add-tracks`, { track_ids: [trackId] });
      await loadPlaylistTracks(activePlaylistId);
      await fetchTree();
    } catch (err) {
      console.error('Failed to add track:', err);
    }
  }, [activePlaylistId, loadPlaylistTracks, fetchTree]);

  // ── Add all suggestion tracks to active playlist ──────────────────────
  const handleAddAllSuggestions = useCallback(async (trackIds) => {
    if (!activePlaylistId) return;
    try {
      await apiClient.post(`/playlists/${activePlaylistId}/add-tracks`, { track_ids: trackIds });
      await loadPlaylistTracks(activePlaylistId);
      await fetchTree();
    } catch (err) {
      console.error('Failed to add tracks:', err);
    }
  }, [activePlaylistId, loadPlaylistTracks, fetchTree]);

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
          mode: data.mode || 'tracks',
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
          activePlaylistId={activePlaylistId}
          onSelectPlaylist={handleSelectPlaylist}
          onCreatePlaylist={handleCreatePlaylist}
          onDeletePlaylist={handleDeletePlaylist}
          onDropOnPlaylist={handleDropTracksOnActive}
          onExport={handleExport}
          fetchTree={fetchTree}
          fetchPool={fetchPool}
        />

        {/* ═══════════ MAIN AREA (split vertical) ═══════════ */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

          {/* TOP HALF — Active Playlist Workspace */}
          <ActivePlaylistPanel
            playlist={activePlaylistId ? { id: activePlaylistId, name: activePlaylistName } : null}
            tracks={activePlaylistTracks}
            playingTrackId={playingTrackId}
            onPlay={handlePlayTrack}
            onDropTracks={handleDropTracksOnActive}
            onRemoveTrack={handleRemoveTrack}
            onCreatePlaylist={handleCreatePlaylist}
            onRenamePlaylist={handleRenamePlaylist}
            searchFilter={searchFilter}
            onSearchChange={setSearchFilter}
          />

          {/* BOTTOM HALF — LLM Suggestions */}
          <SuggestionsPanel
            suggestions={suggestions}
            loading={suggestionsLoading}
            activePlaylistName={activePlaylistId ? activePlaylistName : null}
            onCreatePlaylist={handleCreatePlaylistFromSuggestion}
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
      </div>
    </LazyMotion>
  );
}
