import React, { useEffect, useState, useRef, useCallback } from 'react';
import { m, AnimatePresence, useReducedMotion } from 'framer-motion';
import { createClient } from '@supabase/supabase-js';
import ForceGraph3D from 'react-force-graph-3d';
import * as THREE from 'three';
import DJChatbox from './components/DJChatbox';
import LiveMode from './components/LiveMode';
import ShaderBackground from './components/ui/ShaderBackground';
import GlassPanel from './components/ui/GlassPanel';
import { makeSupabaseCoverUrl } from './utils/coverUrl';

const supabase = createClient(
  "https://cvermotfxamubejfnoje.supabase.co",
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImN2ZXJtb3RmeGFtdWJlamZub2plIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTk2NTU4MTcsImV4cCI6MjA3NTIzMTgxN30.clXSFQ4QVhL8nUK_6shyhDVxhKaHUtnrdyqCnDeCCag"
);

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

// ── SVG Icons ─────────────────────────────────────────────────────────────
const IconPlay = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
    <polygon points="5,3 19,12 5,21" />
  </svg>
);
const IconPause = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
    <rect x="6" y="4" width="4" height="16" rx="1" /><rect x="14" y="4" width="4" height="16" rx="1" />
  </svg>
);
const IconSearch = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true">
    <circle cx="11" cy="11" r="7" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>
);
const IconWaveform = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
    <line x1="12" y1="2" x2="12" y2="22" /><line x1="8" y1="6" x2="8" y2="18" />
    <line x1="16" y1="6" x2="16" y2="18" /><line x1="4" y1="10" x2="4" y2="14" /><line x1="20" y1="10" x2="20" y2="14" />
  </svg>
);

// ── Waveform visualization bars ──────────────────────────────────────────
function WaveformViz({ bpm, isPlaying }) {
  const barCount = 32;
  const seed = bpm ? Math.round(bpm) : 120;
  return (
    <div style={{
      display: 'flex', alignItems: 'flex-end', gap: 2,
      height: 80, width: '100%', padding: '0 4px',
    }}>
      {Array.from({ length: barCount }, (_, i) => {
        const h = 15 + ((seed * (i + 1) * 7) % 65);
        return (
          <div key={i} style={{
            flex: 1, height: `${h}%`,
            borderRadius: 'var(--radius-xs)',
            background: 'linear-gradient(180deg, #00d4ff, rgba(124,58,237,0.6))',
            opacity: isPlaying ? 0.9 : 0.4,
            animation: isPlaying ? `waveBar ${0.4 + (i % 5) * 0.15}s ease-in-out ${i * 0.03}s infinite alternate` : 'none',
            transition: 'opacity 300ms ease',
          }} />
        );
      })}
    </div>
  );
}

const NAV_TABS = ['DISCOVERY', 'LIVE'];

export default function App() {
  const reducedMotion = useReducedMotion();
  const fgRef      = useRef();
  const chatboxRef = useRef(null);
  const audioRef   = useRef(null);
  const texCache     = useRef({});
  const spriteMapRef = useRef({});

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

  // ── Track container size for ForceGraph ────────────────────────────────
  useEffect(() => {
    const onResize = () => setGraphDims({ w: window.innerWidth, h: window.innerHeight - 80 });
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  // ── Load data ────────────────────────────────────────────────────────────
  useEffect(() => {
    (async () => {
      try {
        setIsLoading(true);
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
  }, []);

  useEffect(() => {
    if (audioRef.current) { audioRef.current.pause(); setIsPlaying(false); }
  }, [selectedTrack]);

  // ── Handlers ─────────────────────────────────────────────────────────────
  const handleNodeClick = useCallback((node) => {
    if (!fgRef.current) return;
    if (selectedTrack?.id === node.id) {
      const a = audioRef.current;
      if (!a) return;
      if (isPlaying) { a.pause(); setIsPlaying(false); }
      else {
        const src = node.audioUrl || `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/tracks/${node.id}/audio`;
        a.src = src;
        a.play().catch(() => setIsPlaying(false));
        setIsPlaying(true);
      }
      return;
    }
    setSelectedTrack(node);
    const pos = new THREE.Vector3(node.x, node.y, node.z);
    fgRef.current.cameraPosition(pos.clone().add(new THREE.Vector3(0, 20, 80)), pos, 1500);
  }, [selectedTrack, isPlaying]);

  const handleChatTrackSelect = useCallback((trackid) => {
    const node = allNodes.find(n => String(n.id) === String(trackid));
    if (node) handleNodeClick(node);
  }, [allNodes, handleNodeClick]);

  const togglePlay = useCallback(() => {
    if (!audioRef.current || !selectedTrack) return;
    if (isPlaying) { audioRef.current.pause(); setIsPlaying(false); }
    else {
      const src = selectedTrack.audioUrl || `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/tracks/${selectedTrack.id}/audio`;
      audioRef.current.src = src;
      audioRef.current.play().catch(() => setIsPlaying(false));
      setIsPlaying(true);
    }
  }, [isPlaying, selectedTrack]);

  const handleFindSimilar = useCallback((trackid) => {
    chatboxRef.current?.openAndFindSimilar(trackid);
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
      {!reducedMotion && <ShaderBackground />}

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
      {!reducedMotion && <ShaderBackground />}
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
    <div style={{ width: '100vw', height: '100vh', background: '#050507', position: 'relative', display: 'flex', flexDirection: 'column' }}>

      {/* Animated shader background */}
      {!reducedMotion && <ShaderBackground />}

      {/* ═══════════ FLOATING PILL NAV ═══════════ */}
      <m.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ type: 'spring', damping: 25, stiffness: 300, delay: 0.1 }}
        style={{
          position: 'absolute', top: 16, left: '50%', transform: 'translateX(-50%)',
          zIndex: 200,
          background: 'var(--glass-bg)',
          backdropFilter: 'blur(28px)',
          WebkitBackdropFilter: 'blur(28px)',
          border: '1px solid var(--glass-border)',
          borderRadius: 'var(--radius-pill)',
          boxShadow: 'var(--shadow-float)',
          display: 'flex', alignItems: 'center', gap: 4,
          padding: '6px 8px',
        }}
      >
        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '0 12px 0 8px' }}>
          <div style={{
            width: 26, height: 26, borderRadius: 'var(--radius-sm)',
            background: 'linear-gradient(135deg, rgba(124,58,237,0.25), rgba(0,212,255,0.15))',
            border: '1px solid rgba(124,58,237,0.3)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <IconWaveform />
          </div>
          <span style={{ fontSize: 14, fontWeight: 800, letterSpacing: '0.12em' }}>
            <span style={{ color: '#e2e8f0' }}>DJ</span>
            <span style={{ color: '#a855f7' }}>MATE</span>
          </span>
        </div>

        {/* Divider */}
        <div style={{ width: 1, height: 24, background: 'var(--border-panel)', margin: '0 4px' }} />

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
                padding: '8px 20px', position: 'relative', zIndex: 1,
                fontSize: 11, fontWeight: 600, letterSpacing: '0.12em',
                fontFamily: "'Inter', system-ui, sans-serif",
                color: activeTab === tab ? '#e2e8f0' : '#475569',
                borderRadius: 'var(--radius-pill)',
                transition: 'color 200ms ease',
                display: 'flex', alignItems: 'center', gap: 6,
              }}
            >
              {tab}
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
        <div style={{ width: 1, height: 24, background: 'var(--border-panel)', margin: '0 4px' }} />

        {/* Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '0 8px 0 4px' }}>
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
        </div>
      </m.div>

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
                  fgRef.current?.zoomToFit(400, 100);
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

              {/* Hidden audio */}
              <audio ref={audioRef} onEnded={() => setIsPlaying(false)} onError={() => setIsPlaying(false)} style={{ display: 'none' }} />

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
                      position: 'absolute', top: 80, right: 20, zIndex: 100, width: 320,
                      background: 'var(--glass-bg)',
                      backdropFilter: 'blur(28px)',
                      WebkitBackdropFilter: 'blur(28px)',
                      border: '1px solid var(--glass-border)',
                      borderRadius: 'var(--radius-xl)',
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
                          { label: 'ENERGY', value: selectedTrack.energy != null ? parseFloat(selectedTrack.energy).toFixed(2) : '—', color: '#7c3aed' },
                          { label: 'GENRE', value: selectedTrack.semanticTags?.[0] || '—', color: '#00d4ff', small: true },
                        ].map(({ label, value, color, small }) => (
                          <div key={label} style={{
                            background: 'var(--bg-card)', border: '1px solid var(--border-subtle)',
                            borderRadius: 'var(--radius-sm)', padding: '10px 12px',
                          }}>
                            <div style={{ fontSize: 8, letterSpacing: '0.2em', color: '#475569', fontFamily: "'JetBrains Mono', monospace", marginBottom: 4 }}>{label}</div>
                            <div style={{
                              fontSize: small ? 13 : 20, fontWeight: 700, color,
                              fontFamily: "'JetBrains Mono', monospace",
                              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                            }}>{value}</div>
                          </div>
                        ))}
                      </div>

                      {/* Waveform */}
                      <div style={{
                        background: 'var(--bg-card)', border: '1px solid var(--border-subtle)',
                        borderRadius: 'var(--radius-sm)', padding: '8px 4px', marginBottom: 16,
                      }}>
                        <WaveformViz bpm={selectedTrack.bpm} isPlaying={isPlaying} />
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
                          onClick={() => handleFindSimilar(selectedTrack.id)}
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
              position: 'absolute', bottom: 16, left: '50%', transform: 'translateX(-50%)',
              zIndex: 200,
              background: 'var(--glass-bg)',
              backdropFilter: 'blur(24px)',
              WebkitBackdropFilter: 'blur(24px)',
              border: '1px solid var(--glass-border)',
              borderRadius: 'var(--radius-pill)',
              boxShadow: 'var(--shadow-panel)',
              display: 'flex', alignItems: 'center', gap: 16,
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
    </div>
  );
}
