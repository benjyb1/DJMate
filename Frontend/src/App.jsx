import React, { useEffect, useState, useRef, useCallback } from 'react';
import { m, AnimatePresence, useReducedMotion } from 'framer-motion';
import { supabase, getSupabase } from './utils/supabaseClient';
import { resetSupabaseClient } from './utils/supabaseClient';
import ForceGraph3D from 'react-force-graph-3d';
import * as THREE from 'three';
import DJChatbox from './components/DJChatbox';
import LiveMode from './components/LiveMode';
import PlaylistOrganiser from './components/playlist/PlaylistOrganiser';
import AuthScreen from './components/AuthScreen';
import ProfileTab from './components/ProfileTab';
import SupabaseLinkModal from './components/ui/SupabaseLinkModal';
import Toast from './components/ui/Toast';
import GlassPanel from './components/ui/GlassPanel';
import { makeSupabaseCoverUrl } from './utils/coverUrl';
import { IconPlay, IconPause, IconSearch, IconWaveform } from './components/icons';
import { apiClient } from './api/apiClient';
import { useAuthStore } from './stores/authStore';
import { getCredentials, clearCredentials, syncFromProfile } from './stores/credentialStore';
import { getLocalAudioUrl, isFileSystemAccessSupported, pickMusicFolder } from './utils/localAudio';

// ── Parse semantic_tags (handles array, JSON string, or CSV) ──────────────
function parseTags(raw) {
  if (!raw) return [];
  if (Array.isArray(raw)) return raw;
  if (typeof raw === 'string') {
    try { const p = JSON.parse(raw); return Array.isArray(p) ? p : [raw]; }
    catch { return raw.split(',').map(s => s.trim()).filter(Boolean); }
  }
  return [];
}

// ── Generative canvas texture for tracks without album art ────────────────
function makeGenerativeTexture(node) {
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');

  let hash = 0;
  const str = (node.name || '') + (node.artist || '');
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash) + str.charCodeAt(i);
    hash |= 0;
  }
  const hue = 200 + (Math.abs(hash) % 100);

  // Modernized: rounded gradient with softer look
  const bg = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size * 0.7);
  bg.addColorStop(0, `hsl(${hue}, 50%, 12%)`);
  bg.addColorStop(1, `hsl(${hue}, 70%, 5%)`);
  ctx.fillStyle = bg;
  ctx.beginPath();
  ctx.roundRect(0, 0, size, size, 16);
  ctx.fill();

  // Softer border (no HUD corners)
  ctx.strokeStyle = `hsla(${hue}, 80%, 55%, 0.3)`;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(2, 2, size - 4, size - 4, 14);
  ctx.stroke();

  // Glow
  const glow = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size * 0.35);
  glow.addColorStop(0, `hsla(${hue}, 80%, 60%, 0.15)`);
  glow.addColorStop(1, 'transparent');
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, size, size);

  const initials = (node.name || '?').replace(/[^a-zA-Z0-9]/g, '').slice(0, 2).toUpperCase() || '??';
  ctx.fillStyle = `hsla(${hue}, 90%, 72%, 0.95)`;
  ctx.font = `600 ${Math.round(size * 0.3)}px "JetBrains Mono", monospace`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(initials, size / 2, size / 2);

  return new THREE.CanvasTexture(canvas);
}

// Icons imported from ./components/icons

// ── Waveform visualization bars ──────────────────────────────────────────
function WaveformViz({ bpm, isPlaying, audioRef, currentTime, duration }) {
  const barCount = 32;
  const seed = bpm ? Math.round(bpm) : 120;
  const containerRef = useRef(null);
  const progress = duration > 0 ? currentTime / duration : 0;

  const handleClick = (e) => {
    if (!audioRef?.current || !duration) return;
    const rect = containerRef.current.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    audioRef.current.currentTime = ratio * duration;
  };

  return (
    <div
      ref={containerRef}
      onClick={handleClick}
      style={{
        display: 'flex', alignItems: 'flex-end', gap: 2,
        height: 80, width: '100%', padding: '0 4px',
        cursor: duration > 0 ? 'pointer' : 'default',
      }}
    >
      {Array.from({ length: barCount }, (_, i) => {
        const h = 15 + ((seed * (i + 1) * 7) % 65);
        const barProgress = (i + 1) / barCount;
        const isPast = barProgress <= progress;
        return (
          <div key={i} style={{
            flex: 1, height: `${h}%`,
            borderRadius: 'var(--radius-xs)',
            background: isPast
              ? 'linear-gradient(180deg, #00d4ff, rgba(124,58,237,0.8))'
              : 'linear-gradient(180deg, rgba(0,212,255,0.35), rgba(124,58,237,0.2))',
            opacity: isPlaying ? (isPast ? 1 : 0.5) : (isPast ? 0.7 : 0.3),
            animation: isPlaying && isPast ? `waveBar ${0.4 + (i % 5) * 0.15}s ease-in-out ${i * 0.03}s infinite alternate` : 'none',
            transition: 'opacity 300ms ease, background 200ms ease',
          }} />
        );
      })}
    </div>
  );
}

// ── Inline editable field with autocomplete ──────────────────────────────
function InlineEdit({ value, onSave, suggestions, type = 'text', color = '#00d4ff', small }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value || '');
  const inputRef = useRef(null);

  const match = type === 'text' && draft.length > 0
    ? (suggestions || []).find(s => s.toLowerCase().startsWith(draft.toLowerCase()) && s.toLowerCase() !== draft.toLowerCase())
    : null;

  useEffect(() => { if (editing) { setDraft(value || ''); setTimeout(() => inputRef.current?.focus(), 0); } }, [editing]);

  const commit = () => {
    const final = type === 'number' ? Math.max(1, Math.min(10, parseInt(draft) || 1)) : (match && draft.length > 0 ? match : draft);
    if (String(final) !== String(value)) onSave(final);
    setEditing(false);
  };

  const handleKey = (e) => {
    if (e.key === 'Enter') { e.preventDefault(); commit(); }
    else if (e.key === 'Escape') setEditing(false);
    else if (e.key === 'Tab' && match) { e.preventDefault(); setDraft(match); }
  };

  if (!editing) {
    return (
      <div
        onClick={() => setEditing(true)}
        title="Click to edit"
        style={{
          fontSize: small ? 13 : 20, fontWeight: 700, color,
          fontFamily: "'JetBrains Mono', monospace",
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          cursor: 'pointer', position: 'relative',
        }}
      >
        {value || '—'}
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" style={{ position: 'absolute', right: 0, top: 2, opacity: 0.3 }}>
          <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
          <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
        </svg>
      </div>
    );
  }

  return (
    <div style={{ position: 'relative', fontSize: small ? 13 : 20, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace" }}>
      {type === 'text' && match && (
        <div style={{ position: 'absolute', top: 0, left: 0, color: 'rgba(255,255,255,0.2)', pointerEvents: 'none', whiteSpace: 'nowrap' }}>
          {draft}{match.slice(draft.length)}
        </div>
      )}
      <input
        ref={inputRef}
        type={type}
        min={type === 'number' ? 1 : undefined}
        max={type === 'number' ? 10 : undefined}
        value={draft}
        onChange={e => setDraft(e.target.value)}
        onKeyDown={handleKey}
        onBlur={commit}
        style={{
          width: '100%', background: 'transparent', border: 'none', outline: 'none',
          color, fontSize: 'inherit', fontWeight: 'inherit', fontFamily: 'inherit',
          padding: 0, position: 'relative', zIndex: 1,
        }}
      />
    </div>
  );
}

const NAV_TABS = ['DISCOVERY', 'LIVE', 'PLAYLISTS', 'PROFILE'];
const NAV_TABS_SHORT = { DISCOVERY: 'DISCOVER', LIVE: 'LIVE', PLAYLISTS: 'LISTS', PROFILE: 'PROFILE' };

export default function App() {
  const reducedMotion = useReducedMotion();
  const { session, profile, loading, init, signOut } = useAuthStore();

  useEffect(() => { init(); }, []);

  // Sync BYOS credentials from profile to localStorage whenever profile changes
  useEffect(() => {
    if (profile) {
      syncFromProfile(profile);
    }
  }, [profile]);

  if (loading) {
    return (
      <div style={{ width: '100vw', height: '100dvh', background: '#0a0a14', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ color: '#64748b', fontFamily: "'Inter', system-ui, sans-serif", fontSize: 14 }}>Loading...</div>
      </div>
    );
  }

  if (!session) {
    return <AuthScreen />;
  }

  return <AppMain reducedMotion={reducedMotion} onReconfigure={async () => {
    clearCredentials();
    resetSupabaseClient();
    apiClient.clearCache();
    await signOut();
  }} />;
}

function useIsMobile(breakpoint = 768) {
  const [isMobile, setIsMobile] = useState(() => window.innerWidth <= breakpoint);
  useEffect(() => {
    const mq = window.matchMedia(`(max-width: ${breakpoint}px)`);
    const handler = (e) => setIsMobile(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, [breakpoint]);
  return isMobile;
}

function AppMain({ reducedMotion, onReconfigure }) {
  const isMobile = useIsMobile();
  const { profile } = useAuthStore();
  const fgRef      = useRef();
  const chatboxRef = useRef(null);
  const audioRef   = useRef(null);
  const texCache     = useRef({});
  const spriteMapRef = useRef({});
  const prevBlobUrl  = useRef(null);
  const engineStoppedOnce = useRef(false);

  const [trackData,     setTrackData]     = useState({ nodes: [], links: [] });
  const [allNodes,      setAllNodes]      = useState([]);
  const [allLinks,      setAllLinks]      = useState([]);
  const [isLoading,     setIsLoading]     = useState(true);
  const [error,         setError]         = useState(null);
  const [selectedTrack, setSelectedTrack] = useState(null);
  const [isPlaying,     setIsPlaying]     = useState(false);
  const [activeTab,     setActiveTab]     = useState('DISCOVERY');
  const [graphDims,     setGraphDims]     = useState({ w: window.innerWidth, h: window.innerHeight - 80 });
  const [setList,       setSetList]       = useState([]);
  const [currentTime,   setCurrentTime]   = useState(0);
  const [duration,      setDuration]      = useState(0);
  const [availableTags, setAvailableTags] = useState([]);
  const [showLinkModal, setShowLinkModal] = useState(false);
  const [toast, setToast] = useState({ visible: false, message: '' });
  const [graphVersion, setGraphVersion] = useState(0);

  const handleUploadLibrary = async () => {
    if (!isFileSystemAccessSupported()) {
      setToast({ visible: true, message: 'Local folder access requires Chrome, Edge, or Arc' });
      return;
    }
    try {
      await pickMusicFolder();
      setActiveTab('PLAYLISTS');
    } catch (err) {
      // User cancelled folder picker
    }
  };

  // ── Fetch available tags for autocomplete ──────────────────────────────
  useEffect(() => {
    if (!profile?.supabase_url) return;
    apiClient.get('/tags/available').then(data => {
      setAvailableTags(data.semantic_tags || []);
    }).catch(() => {});
  }, [profile?.supabase_url]);

  // ── Save label edit + trigger online learning ──────────────────────────
  const handleLabelSave = useCallback(async (field, value) => {
    if (!selectedTrack) return;
    const trackid = selectedTrack.id || selectedTrack.trackid;
    const payload = {};
    if (field === 'genre') payload.semantic_tags = [value];
    else if (field === 'energy') payload.energy = value;
    try {
      await apiClient.put(`/tags/${trackid}`, payload);
      // Update local state
      setSelectedTrack(prev => {
        if (!prev) return prev;
        if (field === 'genre') return { ...prev, semanticTags: [value, ...(prev.semanticTags || []).slice(1)] };
        if (field === 'energy') return { ...prev, energy: value };
        return prev;
      });
      // Also update allNodes so the graph reflects the change
      setAllNodes(prev => prev.map(n => {
        if (String(n.id) !== String(trackid)) return n;
        if (field === 'genre') return { ...n, semanticTags: [value, ...(n.semanticTags || []).slice(1)] };
        if (field === 'energy') return { ...n, energy: value };
        return n;
      }));
      // Trigger online learning propagation (fire-and-forget)
      apiClient.post('/tags/learning/propagate', { dry_run: false }).catch(() => {});
    } catch (err) {
      console.error('Failed to save label:', err);
    }
  }, [selectedTrack]);

  // ── Track container size for ForceGraph ────────────────────────────────
  useEffect(() => {
    const onResize = () => setGraphDims({ w: window.innerWidth, h: window.innerHeight });
    window.addEventListener('resize', onResize);
    onResize(); // set initial size
    return () => window.removeEventListener('resize', onResize);
  }, []);

  // ── Load data (only if Supabase is linked) ──────────────────────────────
  useEffect(() => {
    if (!profile?.supabase_url) {
      setIsLoading(false);
      setAllNodes([]);
      setAllLinks([]);
      setTrackData({ nodes: [], links: [] });
      return;
    }
    (async () => {
      try {
        setIsLoading(true);
        if (!getSupabase()) throw new Error('Supabase not connected — link your project first');
        const { data, error: err } = await supabase
          .from('tracks')
          .select('trackid,title,artist,bpm,key,album_art_url,audio_url,x_coord,y_coord,z_coord,track_labels(energy,semantic_tags,vibe),track_features(mfcc)');
        if (err) throw err;
        if (!data?.length) throw new Error('No tracks found in database');

        const nodes = data.map(t => {
          const labels   = Array.isArray(t.track_labels)   ? t.track_labels[0]   : t.track_labels;
          const features = Array.isArray(t.track_features) ? t.track_features[0] : t.track_features;
          return {
            id:       t.trackid,
            name:     t.title  || 'Unknown',
            artist:   t.artist || 'Unknown',
            bpm:      t.bpm,
            key:      t.key,
            energy:   labels?.energy ?? null,
            semanticTags: parseTags(labels?.semantic_tags),
            vibe:     labels?.vibe ?? null,
            albumArt: t.album_art_url || makeSupabaseCoverUrl(t.artist, t.title),
            audioUrl: t.audio_url || null,
            mfccFp:   features?.mfcc ?? null,
            x: t.x_coord || (Math.random() - 0.5) * 1000,
            y: t.y_coord || (Math.random() - 0.5) * 1000,
            z: t.z_coord || (Math.random() - 0.5) * 1000,
          };
        });

        const artistMap = {};
        data.forEach(t => {
          const a = t.artist || 'Unknown';
          if (!artistMap[a]) artistMap[a] = [];
          artistMap[a].push(t);
        });
        const links = [];
        Object.values(artistMap).forEach(tracks => {
          for (let i = 0; i < tracks.length - 1; i++)
            links.push({ source: tracks[i].trackid, target: tracks[i + 1].trackid });
        });

        setAllNodes(nodes);
        setAllLinks(links);
        setTrackData({ nodes, links });
      } catch (e) {
        setError(e.message);
      } finally {
        setIsLoading(false);
      }
    })();
  }, [profile?.supabase_url, graphVersion]);

  useEffect(() => {
    if (audioRef.current) { audioRef.current.pause(); setIsPlaying(false); }
    setCurrentTime(0); setDuration(0);
  }, [selectedTrack]);

  // ── Handlers ─────────────────────────────────────────────────────────────
  // Helper: resolve audio source — try local File System Access first, fall back to backend
  const resolveAudioSrc = useCallback(async (node) => {
    // Revoke previous blob URL to prevent memory leaks
    if (prevBlobUrl.current) {
      URL.revokeObjectURL(prevBlobUrl.current);
      prevBlobUrl.current = null;
    }
    // If the node already has a direct URL, use it
    if (node.audioUrl) return node.audioUrl;
    // Try local filesystem via File System Access API
    if (node.filepath) {
      const blobUrl = await getLocalAudioUrl(node.filepath);
      if (blobUrl) {
        prevBlobUrl.current = blobUrl;
        return blobUrl;
      }
    }
    // Fallback: backend audio endpoint (works in local dev)
    const base = import.meta.env.VITE_API_URL
      || (import.meta.env.DEV ? 'http://localhost:8000' : 'https://djmate.onrender.com');
    return `${base}/tracks/${node.id}/audio`;
  }, []);

  const handleNodeClick = useCallback((node) => {
    if (!fgRef.current) return;
    if (selectedTrack?.id === node.id) {
      const a = audioRef.current;
      if (!a) return;
      if (isPlaying) { a.pause(); setIsPlaying(false); }
      else {
        resolveAudioSrc(node).then(src => {
          a.src = src;
          a.play().catch(() => setIsPlaying(false));
          setIsPlaying(true);
        });
      }
      return;
    }
    setSelectedTrack(node);
    const pos = new THREE.Vector3(node.x, node.y, node.z);
    fgRef.current.cameraPosition(pos.clone().add(new THREE.Vector3(0, 20, 80)), pos, 1500);
  }, [selectedTrack, isPlaying, resolveAudioSrc]);

  const handleChatTrackSelect = useCallback((trackid) => {
    const node = allNodes.find(n => String(n.id) === String(trackid));
    if (node) handleNodeClick(node);
  }, [allNodes, handleNodeClick]);

  const togglePlay = useCallback(() => {
    if (!audioRef.current || !selectedTrack) return;
    if (isPlaying) { audioRef.current.pause(); setIsPlaying(false); }
    else {
      resolveAudioSrc(selectedTrack).then(src => {
        audioRef.current.src = src;
        audioRef.current.play().catch(() => setIsPlaying(false));
        setIsPlaying(true);
      });
    }
  }, [isPlaying, selectedTrack, resolveAudioSrc]);

  const handleFindSimilar = useCallback((trackid, trackName) => {
    chatboxRef.current?.openAndFindSimilar(trackid, trackName);
  }, []);

  // ── Glow ring texture (for selected/hovered nodes) ─────────────────────
  const glowTexRef = useRef(null);
  const getGlowTexture = useCallback(() => {
    if (glowTexRef.current) return glowTexRef.current;
    const size = 128;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    const gradient = ctx.createRadialGradient(size / 2, size / 2, size * 0.2, size / 2, size / 2, size * 0.5);
    gradient.addColorStop(0, 'rgba(124,58,237,0)');
    gradient.addColorStop(0.5, 'rgba(124,58,237,0.25)');
    gradient.addColorStop(0.8, 'rgba(124,58,237,0.08)');
    gradient.addColorStop(1, 'rgba(124,58,237,0)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, size, size);
    glowTexRef.current = new THREE.CanvasTexture(canvas);
    return glowTexRef.current;
  }, []);

  // ── 3D node renderer ────────────────────────────────────────────────────
  const nodeThreeObject = useCallback((node) => {
    const isSelected = selectedTrack?.id === node.id;
    const s = isSelected ? 48 : 28;

    let tex = texCache.current[node.id];
    if (!tex) {
      if (node.albumArt) {
        tex = makeGenerativeTexture(node);
        texCache.current[node.id] = tex;
        new THREE.TextureLoader().load(
          node.albumArt,
          (loaded) => {
            loaded.anisotropy = 8;
            texCache.current[node.id] = loaded;
            const sp = spriteMapRef.current[node.id];
            if (sp?.material) { sp.material.map = loaded; sp.material.needsUpdate = true; }
          },
          undefined,
          () => {}
        );
      } else {
        tex = makeGenerativeTexture(node);
        texCache.current[node.id] = tex;
      }
    }

    const group = new THREE.Group();

    // Glow ring behind selected node
    if (isSelected) {
      const glowMat = new THREE.SpriteMaterial({
        map: getGlowTexture(), transparent: true, depthWrite: false, depthTest: false, opacity: 0.7,
      });
      const glow = new THREE.Sprite(glowMat);
      glow.scale.set(s * 1.8, s * 1.8, 1);
      group.add(glow);
    }

    const mat = new THREE.SpriteMaterial({
      map: tex, transparent: true, depthWrite: false, depthTest: false,
      opacity: isSelected ? 1.0 : 0.72,
    });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(s, s, 1);
    spriteMapRef.current[node.id] = sprite;
    group.add(sprite);
    return group;
  }, [selectedTrack, getGlowTexture]);

  // ── Node tooltip ───────────────────────────────────────────────────
  const nodeLabel = useCallback((n) => `
    <div style="
      background:rgba(8,8,20,0.92);
      backdrop-filter:blur(16px);
      border:1px solid rgba(124,58,237,0.3);
      border-radius:12px;
      padding:0;
      font-family:'Inter',system-ui,sans-serif;
      min-width:160px;
      max-width:220px;
      overflow:hidden;
      box-shadow:0 8px 32px rgba(0,0,0,0.5), 0 0 20px rgba(124,58,237,0.1);
    ">
      <div style="padding:12px 14px">
        <div style="font-size:12px;font-weight:700;color:#e2e8f0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:3px">${n.name}</div>
        <div style="font-size:10px;color:#94a3b8;margin-bottom:6px">${n.artist}</div>
        ${n.bpm ? `<div style="font-size:10px;color:#7c3aed;font-family:'JetBrains Mono',monospace;letter-spacing:0.05em">${Math.round(n.bpm)} BPM${n.key ? ` · ${n.key}` : ''}</div>` : ''}
      </div>
    </div>
  `, []);

  // ── Animation variants ──────────────────────────────────────────────────
  const panelVariants = {
    hidden: { opacity: 0, x: 40, scale: 0.95 },
    visible: { opacity: 1, x: 0, scale: 1 },
    exit: { opacity: 0, x: 40, scale: 0.95 },
  };

  const tabVariants = {
    initial: { opacity: 0, y: 16 },
    enter: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -16 },
  };

  // ── Loading ─────────────────────────────────────────────────────────────
  if (isLoading) return (
    <div style={{
      width: '100vw', height: '100vh', background: '#050507',
      display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column',
      fontFamily: "'Inter', system-ui, sans-serif", position: 'relative', overflow: 'hidden',
    }}>
      <m.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ type: 'spring', damping: 20, stiffness: 200 }}
        style={{ position: 'relative', padding: '32px 40px', textAlign: 'center', zIndex: 1 }}
      >
        <div style={{ fontSize: 54, fontWeight: 800, letterSpacing: '0.18em', lineHeight: 1 }}>
          <span style={{ color: '#e2e8f0' }}>DJ</span>
          <span style={{
            color: 'transparent', WebkitBackgroundClip: 'text', backgroundClip: 'text',
            backgroundImage: 'linear-gradient(135deg, #7c3aed, #00d4ff)',
          }}>MATE</span>
        </div>

        <m.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          style={{
            marginTop: 10, fontSize: 9, letterSpacing: '0.35em', color: 'rgba(148,163,184,0.5)',
            fontFamily: "'JetBrains Mono', monospace",
          }}
        >
          INITIALIZING MUSIC CLOUD
        </m.div>
      </m.div>

      {/* Circular loading ring */}
      <m.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
        style={{ marginTop: 36, zIndex: 1 }}
      >
        <svg width="40" height="40" viewBox="0 0 40 40" style={{ animation: 'spin 1.5s linear infinite' }}>
          <circle cx="20" cy="20" r="16" fill="none" stroke="rgba(124,58,237,0.15)" strokeWidth="2" />
          <circle cx="20" cy="20" r="16" fill="none" stroke="url(#loadGrad)" strokeWidth="2"
            strokeDasharray="80" strokeDashoffset="60" strokeLinecap="round" />
          <defs>
            <linearGradient id="loadGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#7c3aed" />
              <stop offset="100%" stopColor="#00d4ff" />
            </linearGradient>
          </defs>
        </svg>
      </m.div>
    </div>
  );

  // ── Error ───────────────────────────────────────────────────────────────
  if (error) return (
    <div style={{
      width: '100vw', height: '100vh', background: '#050507',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: "'Inter', system-ui, sans-serif",
    }}>
      <GlassPanel style={{ padding: '32px 36px', maxWidth: 420, position: 'relative', zIndex: 1, borderColor: 'rgba(239,68,68,0.25)' }}>
        <div style={{ fontSize: 9, letterSpacing: '0.3em', color: 'rgba(239,68,68,0.7)', fontFamily: 'monospace', marginBottom: 14 }}>
          SYSTEM ERROR
        </div>
        <div style={{ fontSize: 18, fontWeight: 700, color: '#e2e8f0', marginBottom: 8 }}>Failed to initialize</div>
        <div style={{ fontSize: 12, color: '#94a3b8', fontFamily: 'monospace', marginBottom: 24, wordBreak: 'break-all', lineHeight: 1.6 }}>
          {error}
        </div>
        <m.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.97 }}
          onClick={() => window.location.reload()}
          style={{
            background: 'transparent', border: '1px solid rgba(239,68,68,0.45)',
            color: '#ef4444', padding: '10px 28px', cursor: 'pointer', borderRadius: 'var(--radius-sm)',
            fontSize: 11, letterSpacing: '0.15em', fontFamily: 'inherit', fontWeight: 600,
          }}
        >
          RETRY
        </m.button>
      </GlassPanel>
    </div>
  );

  // ── Main UI ─────────────────────────────────────────────────────────────
  return (
    <div style={{ width: '100vw', height: '100dvh', background: '#050507', position: 'relative', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

      {/* ═══════════ FLOATING PILL NAV ═══════════ */}
      <m.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ type: 'spring', damping: 25, stiffness: 300, delay: 0.1 }}
        style={{
          position: 'absolute', top: isMobile ? 8 : 16, left: '50%', transform: 'translateX(-50%)',
          zIndex: 200,
          background: 'var(--glass-bg)',
          backdropFilter: 'blur(28px)',
          WebkitBackdropFilter: 'blur(28px)',
          border: '1px solid var(--glass-border)',
          borderRadius: 'var(--radius-pill)',
          boxShadow: 'var(--shadow-float)',
          display: 'flex', alignItems: 'center', gap: isMobile ? 2 : 4,
          padding: isMobile ? '4px 4px' : '6px 8px',
          maxWidth: isMobile ? 'calc(100vw - 16px)' : undefined,
          overflow: 'hidden',
        }}
      >
        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: isMobile ? 3 : 6, padding: isMobile ? '0 4px' : '0 12px 0 8px' }}>
          <div style={{
            width: isMobile ? 20 : 26, height: isMobile ? 20 : 26, borderRadius: 'var(--radius-sm)',
            background: 'linear-gradient(135deg, rgba(124,58,237,0.25), rgba(0,212,255,0.15))',
            border: '1px solid rgba(124,58,237,0.3)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <IconWaveform />
          </div>
          {!isMobile && (
            <span style={{ fontSize: 14, fontWeight: 800, letterSpacing: '0.12em' }}>
              <span style={{ color: '#e2e8f0' }}>DJ</span>
              <span style={{ color: '#a855f7' }}>MATE</span>
            </span>
          )}
        </div>

        {/* Divider */}
        {!isMobile && <div style={{ width: 1, height: 24, background: 'var(--border-panel)', margin: '0 4px' }} />}

        {/* Nav tabs with sliding indicator */}
        <div style={{ display: 'flex', gap: 2, position: 'relative', padding: '2px' }}>
          {NAV_TABS.map(tab => (
            <m.button
              key={tab}
              onClick={() => setActiveTab(tab)}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.97 }}
              style={{
                background: 'none', border: 'none', cursor: 'pointer',
                padding: isMobile ? '6px 8px' : '8px 20px', position: 'relative', zIndex: 1,
                fontSize: isMobile ? 9 : 11, fontWeight: 600, letterSpacing: isMobile ? '0.06em' : '0.12em',
                fontFamily: "'Inter', system-ui, sans-serif",
                color: activeTab === tab ? '#e2e8f0' : '#475569',
                borderRadius: 'var(--radius-pill)',
                transition: 'color 200ms ease',
                display: 'flex', alignItems: 'center', gap: isMobile ? 4 : 6,
              }}
            >
              {isMobile ? NAV_TABS_SHORT[tab] : tab}
              {tab === 'LIVE' && setList.length > 0 && (
                <div style={{
                  width: 6, height: 6, borderRadius: '50%', background: '#ef4444',
                  boxShadow: '0 0 6px #ef4444',
                  animation: 'blink 1.5s ease-in-out infinite',
                }} />
              )}
              {/* Sliding indicator behind active tab */}
              {activeTab === tab && (
                <m.div
                  layoutId="navIndicator"
                  style={{
                    position: 'absolute', inset: 0,
                    background: 'var(--gradient-accent-soft)',
                    borderRadius: 'var(--radius-pill)',
                    border: '1px solid rgba(124,58,237,0.2)',
                    zIndex: -1,
                  }}
                  transition={{ type: 'spring', damping: 25, stiffness: 300 }}
                />
              )}
            </m.button>
          ))}
        </div>

        {/* Divider */}
        {!isMobile && <div style={{ width: 1, height: 24, background: 'var(--border-panel)', margin: '0 4px' }} />}

        {/* Status — hidden on mobile, only show when tracks loaded */}
        {!isMobile && allNodes.length > 0 && <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '0 8px 0 4px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <div style={{
              width: 6, height: 6, borderRadius: '50%', background: '#00d4ff',
              boxShadow: '0 0 8px #00d4ff',
              animation: 'statusPulse 2s ease-in-out infinite',
            }} />
            <span style={{ fontSize: 9, color: '#00d4ff', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.1em', fontWeight: 600 }}>
              SYNCED
            </span>
          </div>
          <span style={{ fontSize: 9, color: '#2d3748', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.08em' }}>
            {allNodes.length} TRACKS
          </span>
        </div>}
      </m.div>

      {/* Username header */}
      {profile?.username && (
        <div style={{
          position: 'absolute', top: isMobile ? 52 : 64, left: '50%', transform: 'translateX(-50%)',
          zIndex: 190,
          fontSize: 11, color: 'rgba(226,232,240,0.4)',
          fontFamily: "'JetBrains Mono', monospace",
          letterSpacing: '0.1em',
          pointerEvents: 'none',
        }}>
          {profile.username}&apos;s Space
        </div>
      )}

      {/* ═══════════ MAIN CONTENT ═══════════ */}
      <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        <AnimatePresence mode="wait">

          {/* ── LIVE TAB ── */}
          {activeTab === 'LIVE' && (
            <m.div
              key="live"
              variants={tabVariants}
              initial="initial"
              animate="enter"
              exit="exit"
              transition={{ duration: 0.25 }}
              style={{ position: 'absolute', inset: 0 }}
            >
              <LiveMode setList={setList} setSetList={setSetList} allNodes={allNodes} />
            </m.div>
          )}

          {/* ── PLAYLISTS TAB ── */}
          {activeTab === 'PLAYLISTS' && (
            <m.div
              key="playlists"
              variants={tabVariants}
              initial="initial"
              animate="enter"
              exit="exit"
              transition={{ duration: 0.25 }}
              style={{ position: 'absolute', inset: 0 }}
            >
              <PlaylistOrganiser onIngestComplete={() => setGraphVersion(v => v + 1)} />
            </m.div>
          )}

          {/* ── PROFILE TAB ── */}
          {activeTab === 'PROFILE' && (
            <m.div
              key="profile"
              variants={tabVariants}
              initial="initial"
              animate="enter"
              exit="exit"
              transition={{ duration: 0.25 }}
              style={{ position: 'absolute', inset: 0, overflowY: 'auto' }}
            >
              <ProfileTab />
            </m.div>
          )}

          {/* ── DISCOVERY TAB ── */}
          {activeTab === 'DISCOVERY' && (
            <m.div
              key="discovery"
              variants={tabVariants}
              initial="initial"
              animate="enter"
              exit="exit"
              transition={{ duration: 0.25 }}
              style={{ position: 'absolute', inset: 0 }}
            >
              {/* 3D Graph */}
              <ForceGraph3D
                ref={fgRef}
                graphData={trackData}
                width={graphDims.w}
                height={graphDims.h}
                backgroundColor="rgba(0,0,0,0)"
                nodeThreeObject={nodeThreeObject}
                linkColor={() => 'rgba(124,58,237,0.18)'}
                linkWidth={() => 0.4}
                linkDirectionalParticles={reducedMotion ? 0 : 2}
                linkDirectionalParticleWidth={1.5}
                linkDirectionalParticleColor={() => '#7c3aed'}
                linkDirectionalParticleSpeed={0.005}
                controlType="trackball"
                onNodeClick={handleNodeClick}
                onEngineStop={() => {
                  if (!engineStoppedOnce.current) {
                    engineStoppedOnce.current = true;
                    fgRef.current?.zoomToFit(400, 100);
                  }
                  const c = fgRef.current?.controls();
                  if (c) { c.maxDistance = 8000; c.minDistance = 10; }

                  // Add ambient particles to scene
                  const scene = fgRef.current?.scene();
                  if (scene && !scene.userData._particlesAdded) {
                    scene.userData._particlesAdded = true;
                    const count = 200;
                    const positions = new Float32Array(count * 3);
                    for (let i = 0; i < count; i++) {
                      positions[i * 3]     = (Math.random() - 0.5) * 2000;
                      positions[i * 3 + 1] = (Math.random() - 0.5) * 2000;
                      positions[i * 3 + 2] = (Math.random() - 0.5) * 2000;
                    }
                    const geo = new THREE.BufferGeometry();
                    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                    const pMat = new THREE.PointsMaterial({
                      color: 0x7c3aed, size: 2, transparent: true, opacity: 0.25,
                      sizeAttenuation: true, depthWrite: false,
                    });
                    scene.add(new THREE.Points(geo, pMat));
                  }
                }}
                nodeLabel={nodeLabel}
              />

              {/* Onboarding overlay — hide once tracks are loaded */}
              {!profile?.supabase_url && allNodes.length === 0 && (
                <div style={{
                  position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
                  zIndex: 100, textAlign: 'center',
                  display: 'flex', flexDirection: 'column', gap: 16, alignItems: 'center',
                }}>
                  <m.button
                    onClick={() => setShowLinkModal(true)}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.97 }}
                    style={{
                      padding: '14px 32px',
                      background: 'linear-gradient(135deg, rgba(124,58,237,0.3), rgba(0,212,255,0.15))',
                      border: '1px solid rgba(124,58,237,0.3)',
                      borderRadius: 'var(--radius-sm)',
                      color: '#e2e8f0', fontSize: 14, fontWeight: 600,
                      cursor: 'pointer', fontFamily: "'Inter', system-ui, sans-serif",
                      backdropFilter: 'blur(16px)',
                    }}
                  >
                    Link Supabase
                  </m.button>
                </div>
              )}

              {/* Hidden audio */}
              {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
              <audio
                ref={audioRef}
                playsInline
                preload="none"
                onEnded={() => { setIsPlaying(false); setCurrentTime(0); }}
                onError={() => { setIsPlaying(false); setCurrentTime(0); }}
                onTimeUpdate={() => setCurrentTime(audioRef.current?.currentTime || 0)}
                onLoadedMetadata={() => setDuration(audioRef.current?.duration || 0)}
                style={{ display: 'none' }}
              />

              {/* ── Selected track panel (right side) ───────────────────────── */}
              <AnimatePresence>
                {selectedTrack && (
                  <m.div
                    key="trackPanel"
                    variants={panelVariants}
                    initial="hidden"
                    animate="visible"
                    exit="exit"
                    transition={{ type: 'spring', damping: 25, stiffness: 300 }}
                    style={{
                      position: 'absolute',
                      ...(isMobile
                        ? { bottom: 0, left: 0, right: 0, zIndex: 100, width: '100%', maxHeight: '60vh', overflowY: 'auto', borderRadius: 'var(--radius-xl) var(--radius-xl) 0 0' }
                        : { top: 80, right: 20, zIndex: 100, width: 320, borderRadius: 'var(--radius-xl)' }
                      ),
                      background: 'var(--glass-bg)',
                      backdropFilter: 'blur(28px)',
                      WebkitBackdropFilter: 'blur(28px)',
                      border: '1px solid var(--glass-border)',
                      boxShadow: 'var(--shadow-float)',
                      overflow: 'hidden',
                    }}
                  >
                    {/* Subtle gradient glow at top */}
                    <div style={{
                      height: 2, borderRadius: 'var(--radius-xl) var(--radius-xl) 0 0',
                      background: 'linear-gradient(90deg, #7c3aed, #00d4ff, rgba(0,212,255,0.1))',
                    }} />

                    {/* Header */}
                    <div style={{ padding: '14px 18px 0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{
                        fontSize: 8, letterSpacing: '0.3em',
                        color: 'rgba(0,212,255,0.6)', fontFamily: "'JetBrains Mono', monospace",
                      }}>NOW SELECTED</span>
                      <m.button
                        whileHover={{ scale: 1.1 }}
                        whileTap={{ scale: 0.9 }}
                        onClick={() => {
                          setSelectedTrack(null);
                          setTrackData({ nodes: allNodes, links: allLinks });
                          if (audioRef.current) { audioRef.current.pause(); setIsPlaying(false); }
                        }}
                        style={{
                          background: 'none', border: 'none', color: '#475569', cursor: 'pointer',
                          display: 'flex', padding: 6, borderRadius: 'var(--radius-sm)',
                          transition: '150ms ease',
                        }}
                        aria-label="Close panel"
                      >
                        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                          <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                        </svg>
                      </m.button>
                    </div>

                    <div style={{ padding: '10px 18px 18px' }}>
                      {/* Title + Artist */}
                      <div style={{ fontSize: 17, fontWeight: 700, color: '#e2e8f0', lineHeight: 1.3, marginBottom: 2, wordBreak: 'break-word' }}>
                        {selectedTrack.name}
                      </div>
                      <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 16, letterSpacing: '0.02em' }}>
                        {selectedTrack.artist}
                      </div>

                      {/* Metadata grid (2x2) */}
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 16 }}>
                        {[
                          { label: 'TEMPO/BPM', value: selectedTrack.bpm ? parseFloat(selectedTrack.bpm).toFixed(2) : '—', color: '#00d4ff' },
                          { label: 'HARMONIC KEY', value: selectedTrack.key || '—', color: '#a855f7' },
                        ].map(({ label, value, color }) => (
                          <div key={label} style={{
                            background: 'var(--bg-card)', border: '1px solid var(--border-subtle)',
                            borderRadius: 'var(--radius-sm)', padding: '10px 12px',
                          }}>
                            <div style={{ fontSize: 8, letterSpacing: '0.2em', color: '#475569', fontFamily: "'JetBrains Mono', monospace", marginBottom: 4 }}>{label}</div>
                            <div style={{
                              fontSize: 20, fontWeight: 700, color,
                              fontFamily: "'JetBrains Mono', monospace",
                              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                            }}>{value}</div>
                          </div>
                        ))}
                        {/* Editable Energy */}
                        <div style={{
                          background: 'var(--bg-card)', border: '1px solid var(--border-subtle)',
                          borderRadius: 'var(--radius-sm)', padding: '10px 12px',
                        }}>
                          <div style={{ fontSize: 8, letterSpacing: '0.2em', color: '#475569', fontFamily: "'JetBrains Mono', monospace", marginBottom: 4 }}>ENERGY</div>
                          <InlineEdit
                            value={selectedTrack.energy != null ? Math.round(selectedTrack.energy) : '—'}
                            type="number"
                            color="#7c3aed"
                            onSave={v => handleLabelSave('energy', parseInt(v))}
                          />
                        </div>
                        {/* Editable Genre */}
                        <div style={{
                          background: 'var(--bg-card)', border: '1px solid var(--border-subtle)',
                          borderRadius: 'var(--radius-sm)', padding: '10px 12px',
                        }}>
                          <div style={{ fontSize: 8, letterSpacing: '0.2em', color: '#475569', fontFamily: "'JetBrains Mono', monospace", marginBottom: 4 }}>GENRE</div>
                          <InlineEdit
                            value={selectedTrack.semanticTags?.[0] || '—'}
                            suggestions={availableTags}
                            color="#00d4ff"
                            small
                            onSave={v => handleLabelSave('genre', v)}
                          />
                        </div>
                      </div>

                      {/* Waveform */}
                      <div style={{
                        background: 'var(--bg-card)', border: '1px solid var(--border-subtle)',
                        borderRadius: 'var(--radius-sm)', padding: '8px 4px', marginBottom: 16,
                      }}>
                        <WaveformViz bpm={selectedTrack.bpm} isPlaying={isPlaying} audioRef={audioRef} currentTime={currentTime} duration={duration} />
                      </div>

                      {/* Action buttons */}
                      <div style={{ display: 'flex', gap: 8 }}>
                        <m.button
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.97 }}
                          onClick={togglePlay}
                          aria-label={isPlaying ? 'Pause' : 'Play'}
                          style={{
                            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
                            flex: 1, padding: '10px 0',
                            background: isPlaying ? 'rgba(0,212,255,0.08)' : 'rgba(124,58,237,0.08)',
                            border: `1px solid ${isPlaying ? 'rgba(0,212,255,0.4)' : 'rgba(124,58,237,0.4)'}`,
                            borderRadius: 'var(--radius-sm)',
                            color: isPlaying ? '#00d4ff' : '#a855f7',
                            cursor: 'pointer', fontSize: 10, fontWeight: 700,
                            letterSpacing: '0.14em', fontFamily: "'Inter', system-ui, sans-serif",
                          }}
                        >
                          {isPlaying ? <IconPause /> : <IconPlay />}
                          {isPlaying ? 'PAUSE' : 'PLAY'}
                        </m.button>
                      </div>

                      <div style={{ display: 'flex', gap: 8, marginTop: 6 }}>
                        <m.button
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.97 }}
                          onClick={() => handleFindSimilar(selectedTrack.id, selectedTrack.name)}
                          style={{
                            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
                            flex: 1, padding: '10px 0',
                            background: 'transparent',
                            border: '1px solid rgba(0,212,255,0.25)',
                            borderRadius: 'var(--radius-sm)',
                            color: '#00d4ff', cursor: 'pointer', fontSize: 10, fontWeight: 700,
                            letterSpacing: '0.14em', fontFamily: "'Inter', system-ui, sans-serif",
                          }}
                        >
                          <IconSearch /> FIND SIMILAR
                        </m.button>
                        <m.button
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.97 }}
                          onClick={() => {
                            setSetList(prev => {
                              const id = String(selectedTrack.id);
                              if (prev.some(t => String(t.id) === id)) return prev;
                              return [...prev, selectedTrack];
                            });
                            setActiveTab('LIVE');
                          }}
                          style={{
                            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
                            flex: 1, padding: '10px 0',
                            background: 'transparent',
                            border: '1px solid rgba(239,68,68,0.25)',
                            borderRadius: 'var(--radius-sm)',
                            color: '#ef4444', cursor: 'pointer', fontSize: 10, fontWeight: 700,
                            letterSpacing: '0.14em', fontFamily: "'Inter', system-ui, sans-serif",
                          }}
                        >
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true">
                            <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
                          </svg> ADD TO SET
                        </m.button>
                      </div>
                    </div>
                  </m.div>
                )}
              </AnimatePresence>

              <DJChatbox
                ref={chatboxRef}
                selectedTrack={selectedTrack}
                trackCount={allNodes.length}
                onTrackSelect={handleChatTrackSelect}
              />
            </m.div>
          )}
        </AnimatePresence>
      </div>

      {/* ═══════════ FLOATING BOTTOM BAR ═══════════ */}
      <AnimatePresence>
        {selectedTrack && (
          <m.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            style={{
              position: 'absolute', bottom: isMobile ? 8 : 16,
              left: '50%', transform: 'translateX(-50%)',
              zIndex: 200,
              background: 'var(--glass-bg)',
              backdropFilter: 'blur(24px)',
              WebkitBackdropFilter: 'blur(24px)',
              border: '1px solid var(--glass-border)',
              borderRadius: 'var(--radius-pill)',
              boxShadow: 'var(--shadow-panel)',
              display: isMobile ? 'none' : 'flex', alignItems: 'center', gap: 16,
              padding: '8px 20px',
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 10, letterSpacing: '0.08em',
            }}
          >
            <span style={{ color: '#e2e8f0', fontWeight: 600, fontFamily: "'Inter', sans-serif", maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {selectedTrack.name}
            </span>
            <span style={{ color: '#94a3b8', fontSize: 9 }}>{selectedTrack.artist}</span>
            <div style={{ width: 1, height: 16, background: 'var(--border-panel)' }} />
            <span style={{ color: '#00d4ff' }}>{selectedTrack.bpm ? Math.round(selectedTrack.bpm) + ' BPM' : '—'}</span>
            <span style={{ color: '#a855f7' }}>{selectedTrack.key || '—'}</span>
          </m.div>
        )}
      </AnimatePresence>

      <SupabaseLinkModal
        open={showLinkModal}
        onClose={() => setShowLinkModal(false)}
        onLink={async (url, key) => {
          await useAuthStore.getState().linkSupabase(url, key);
          syncFromProfile(useAuthStore.getState().profile);
          setShowLinkModal(false);
        }}
        defaultUrl={profile?.supabase_url || ''}
        defaultKey={profile?.supabase_key || ''}
      />

      <Toast
        message={toast.message}
        visible={toast.visible}
        onDismiss={() => setToast({ visible: false, message: '' })}
      />
    </div>
  );
}
