import React, { useEffect, useState, useRef, useCallback } from 'react';
import { createClient } from '@supabase/supabase-js';
import ForceGraph3D from 'react-force-graph-3d';
import * as THREE from 'three';
import DJChatbox from './components/DJChatbox';
import LiveMode from './components/LiveMode';

const supabase = createClient(
  "https://cvermotfxamubejfnoje.supabase.co",
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImN2ZXJtb3RmeGFtdWJlamZub2plIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTk2NTU4MTcsImV4cCI6MjA3NTIzMTgxN30.clXSFQ4QVhL8nUK_6shyhDVxhKaHUtnrdyqCnDeCCag"
);

// ── Supabase storage cover URL (mirrors create_safe_filename in upload scripts) ──
const SUPABASE_COVERS_BASE = 'https://cvermotfxamubejfnoje.supabase.co/storage/v1/object/public/album-covers/';
function makeSupabaseCoverUrl(artist, title) {
  if (!artist || !title) return null;
  let safe = `${artist}_${title}`.toLowerCase();
  safe = safe.split('').map(c => /[a-z0-9\-_]/.test(c) ? c : '_').join('');
  safe = safe.split('_').filter(Boolean).join('_').slice(0, 150);
  return `${SUPABASE_COVERS_BASE}${safe}.jpg`;
}

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

  const bg = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size * 0.7);
  bg.addColorStop(0, `hsl(${hue}, 50%, 9%)`);
  bg.addColorStop(1, `hsl(${hue}, 70%, 4%)`);
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, size, size);

  ctx.strokeStyle = `hsla(${hue}, 80%, 55%, 0.5)`;
  ctx.lineWidth = 1.5;
  ctx.strokeRect(1.5, 1.5, size - 3, size - 3);

  const m = 10;
  ctx.strokeStyle = `hsla(${hue}, 90%, 65%, 0.8)`;
  ctx.lineWidth = 1.5;
  [[2, m, 2, 2, m, 2], [size - m, 2, size - 2, 2, size - 2, m],
   [2, size - m, 2, size - 2, m, size - 2],
   [size - m, size - 2, size - 2, size - 2, size - 2, size - m]
  ].forEach(([x1, y1, x2, y2, x3, y3]) => {
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.lineTo(x3, y3); ctx.stroke();
  });

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
const IconReset = () => (
  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true">
    <polyline points="1,4 1,10 7,10" /><path d="M3.51 15a9 9 0 1 0 .49-5.9" />
  </svg>
);
const IconWaveform = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
    <line x1="12" y1="2"  x2="12" y2="22" />
    <line x1="8"  y1="6"  x2="8"  y2="18" />
    <line x1="16" y1="6"  x2="16" y2="18" />
    <line x1="4"  y1="10" x2="4"  y2="14" />
    <line x1="20" y1="10" x2="20" y2="14" />
  </svg>
);
const IconBell = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
    <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" /><path d="M13.73 21a2 2 0 0 1-3.46 0" />
  </svg>
);
const IconUser = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" />
  </svg>
);
const IconAnalyze = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
    <path d="M21 12V7H5a2 2 0 0 1 0-4h14v4" /><path d="M3 5v14a2 2 0 0 0 2 2h16v-5" /><path d="M18 12a2 2 0 0 0 0 4h4v-4h-4z" />
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
            background: `linear-gradient(180deg, #00d4ff, rgba(124,58,237,0.6))`,
            opacity: isPlaying ? 0.9 : 0.4,
            animation: isPlaying ? `waveBar ${0.4 + (i % 5) * 0.15}s ease-in-out ${i * 0.03}s infinite alternate` : 'none',
            transition: 'opacity 300ms ease',
          }} />
        );
      })}
    </div>
  );
}

// ── Shared styles ──────────────────────────────────────────────────────────
const BTN_BASE = {
  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
  width: '100%', padding: '9px 0',
  background: 'transparent',
  cursor: 'pointer', fontSize: 10, fontWeight: 700,
  letterSpacing: '0.14em', fontFamily: "'Inter', system-ui, sans-serif",
  transition: '150ms ease', borderRadius: 0,
};

const NAV_TABS = ['DISCOVERY', 'LIVE'];

export default function App() {
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
  const [setList,       setSetList]       = useState([]); // persists between tab switches

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
          .select('trackid,title,artist,bpm,key,album_art_url,audio_url,x_coord,y_coord,z_coord,track_labels(energy,semantic_tags,vibe)');
        if (err) throw err;
        if (!data?.length) throw new Error('No tracks found in database');

        const nodes = data.map(t => {
          const labels = Array.isArray(t.track_labels) ? t.track_labels[0] : t.track_labels;
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
    setSelectedTrack(node);
    const pos = new THREE.Vector3(node.x, node.y, node.z);
    fgRef.current.cameraPosition(pos.clone().add(new THREE.Vector3(0, 30, 120)), pos, 1000);
  }, []);

  const handleChatTrackSelect = useCallback((trackid) => {
    const node = allNodes.find(n => String(n.id) === String(trackid));
    if (node) handleNodeClick(node);
  }, [allNodes, handleNodeClick]);

  const togglePlay = useCallback(() => {
    if (!audioRef.current || !selectedTrack) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      // Prefer Supabase-hosted audio; fall back to backend endpoint
      const src = selectedTrack.audioUrl
        || `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/tracks/${selectedTrack.id}/audio`;
      audioRef.current.src = src;
      audioRef.current.play().catch(() => setIsPlaying(false));
      setIsPlaying(true);
    }
  }, [isPlaying, selectedTrack]);

  const handleFindSimilar = useCallback((trackid) => {
    chatboxRef.current?.openAndFindSimilar(trackid);
  }, []);

  // ── 3D node renderer ────────────────────────────────────────────────────
  const nodeThreeObject = useCallback((node) => {
    const isSelected = selectedTrack?.id === node.id;
    const s = isSelected ? 44 : 30;

    let tex = texCache.current[node.id];
    if (!tex) {
      if (node.albumArt) {
        // Show generative immediately; swap to real art when it loads
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
          () => { /* 404 — generative stays */ }
        );
      } else {
        tex = makeGenerativeTexture(node);
        texCache.current[node.id] = tex;
      }
    }

    const mat = new THREE.SpriteMaterial({
      map: tex, transparent: true, depthWrite: false, depthTest: false,
      opacity: isSelected ? 1.0 : 0.72,
    });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(s, s, 1);
    spriteMapRef.current[node.id] = sprite;
    return sprite;
  }, [selectedTrack]);

  // ── Node tooltip HTML ───────────────────────────────────────────────────
  const nodeLabel = useCallback((n) => `
    <div style="
      background:rgba(5,5,7,0.97);
      border:1px solid rgba(124,58,237,0.5);
      padding:0;
      font-family:'Inter',system-ui,sans-serif;
      min-width:160px;
      max-width:220px;
      overflow:hidden;
    ">
      <div style="height:1px;background:linear-gradient(90deg,#7c3aed,#00d4ff,transparent)"></div>
      <div style="padding:10px 12px">
        <div style="font-size:12px;font-weight:700;color:#e2e8f0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:2px">${n.name}</div>
        <div style="font-size:10px;color:#94a3b8;margin-bottom:6px">${n.artist}</div>
        ${n.bpm ? `<div style="font-size:10px;color:#7c3aed;font-family:'JetBrains Mono',monospace;letter-spacing:0.05em">${Math.round(n.bpm)} BPM${n.key ? ` · ${n.key}` : ''}</div>` : ''}
      </div>
    </div>
  `, []);

  // ── Loading ─────────────────────────────────────────────────────────────
  if (isLoading) return (
    <div style={{
      width: '100vw', height: '100vh', background: '#050507',
      display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column',
      fontFamily: "'Inter', system-ui, sans-serif", position: 'relative', overflow: 'hidden',
    }}>
      <div style={{
        position: 'absolute', inset: 0, pointerEvents: 'none',
        backgroundImage: 'linear-gradient(rgba(124,58,237,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(124,58,237,0.04) 1px, transparent 1px)',
        backgroundSize: '48px 48px',
      }} />

      <div style={{ position: 'relative', padding: '32px 40px', textAlign: 'center' }}>
        {[['top:0;left:0','borderTop','borderLeft'],['top:0;right:0','borderTop','borderRight'],
          ['bottom:0;left:0','borderBottom','borderLeft'],['bottom:0;right:0','borderBottom','borderRight']
        ].map(([pos, b1, b2], i) => (
          <div key={i} style={{
            position: 'absolute', ...Object.fromEntries(pos.split(';').map(p => p.split(':').map(s => s.trim()))),
            width: 18, height: 18,
            [b1]: '1px solid rgba(0,212,255,0.5)',
            [b2]: '1px solid rgba(0,212,255,0.5)',
          }} />
        ))}

        <div style={{ fontSize: 54, fontWeight: 800, letterSpacing: '0.18em', lineHeight: 1 }}>
          <span style={{ color: '#e2e8f0' }}>DJ</span>
          <span style={{
            color: 'transparent', WebkitBackgroundClip: 'text', backgroundClip: 'text',
            backgroundImage: 'linear-gradient(135deg, #7c3aed, #00d4ff)',
          }}>MATE</span>
        </div>

        <div style={{
          marginTop: 10, fontSize: 9, letterSpacing: '0.35em', color: 'rgba(148,163,184,0.5)',
          fontFamily: "'JetBrains Mono', monospace",
        }}>
          INITIALIZING MUSIC CLOUD
        </div>
      </div>

      <div style={{ marginTop: 36, width: 220, height: 1, background: 'rgba(124,58,237,0.15)', position: 'relative', overflow: 'hidden' }}>
        <div style={{
          position: 'absolute', top: 0, height: '100%', width: '55%',
          background: 'linear-gradient(90deg, transparent, #7c3aed, #00d4ff, transparent)',
          animation: 'scanBar 1.8s ease-in-out infinite',
        }} />
      </div>
    </div>
  );

  // ── Error ───────────────────────────────────────────────────────────────
  if (error) return (
    <div style={{
      width: '100vw', height: '100vh', background: '#050507',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: "'Inter', system-ui, sans-serif",
    }}>
      <div style={{
        background: 'rgba(8,8,16,0.95)', border: '1px solid rgba(239,68,68,0.35)',
        padding: '32px 36px', maxWidth: 420, position: 'relative',
      }}>
        <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 1, background: 'linear-gradient(90deg, #ef4444, rgba(239,68,68,0.2), transparent)' }} />
        <div style={{ fontSize: 9, letterSpacing: '0.3em', color: 'rgba(239,68,68,0.7)', fontFamily: 'monospace', marginBottom: 14 }}>
          SYSTEM ERROR
        </div>
        <div style={{ fontSize: 18, fontWeight: 700, color: '#e2e8f0', marginBottom: 8 }}>
          Failed to initialize
        </div>
        <div style={{ fontSize: 12, color: '#94a3b8', fontFamily: 'monospace', marginBottom: 24, wordBreak: 'break-all', lineHeight: 1.6 }}>
          {error}
        </div>
        <button
          onClick={() => window.location.reload()}
          style={{
            background: 'transparent', border: '1px solid rgba(239,68,68,0.45)',
            color: '#ef4444', padding: '8px 24px', cursor: 'pointer',
            fontSize: 11, letterSpacing: '0.15em', fontFamily: 'inherit', fontWeight: 600,
            transition: '150ms ease',
          }}
          onMouseEnter={e => { e.currentTarget.style.background = 'rgba(239,68,68,0.08)'; }}
          onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; }}
        >
          RETRY
        </button>
      </div>
    </div>
  );

  // ── Main UI ─────────────────────────────────────────────────────────────
  return (
    <div style={{ width: '100vw', height: '100vh', background: '#050507', position: 'relative', display: 'flex', flexDirection: 'column' }}>

      {/* ═══════════ TOP NAV BAR ═══════════ */}
      <div style={{
        height: 48, flexShrink: 0, zIndex: 200,
        background: 'rgba(5,5,7,0.95)',
        borderBottom: '1px solid rgba(124,58,237,0.22)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 20px',
        backdropFilter: 'blur(20px)',
      }}>
        {/* Left: Logo + Tabs */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
          {/* Logo badge */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{
              width: 28, height: 28,
              background: 'linear-gradient(135deg, rgba(124,58,237,0.25), rgba(0,212,255,0.15))',
              border: '1px solid rgba(124,58,237,0.4)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" strokeWidth="2.5" strokeLinecap="round">
                <line x1="12" y1="3" x2="12" y2="21" />
                <line x1="8" y1="7" x2="8" y2="17" />
                <line x1="16" y1="7" x2="16" y2="17" />
                <line x1="4" y1="11" x2="4" y2="13" />
                <line x1="20" y1="11" x2="20" y2="13" />
              </svg>
            </div>
            <span style={{ fontSize: 15, fontWeight: 800, letterSpacing: '0.14em' }}>
              <span style={{ color: '#e2e8f0' }}>DJ</span>
              <span style={{ color: '#a855f7' }}>MATE</span>
            </span>
          </div>

          {/* Nav tabs */}
          <div style={{ display: 'flex', gap: 0 }}>
            {NAV_TABS.map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                style={{
                  background: 'none', border: 'none', cursor: 'pointer',
                  padding: '6px 16px', position: 'relative',
                  fontSize: 11, fontWeight: 600, letterSpacing: '0.12em',
                  fontFamily: "'Inter', system-ui, sans-serif",
                  color: activeTab === tab ? '#00d4ff' : '#475569',
                  borderBottom: activeTab === tab ? '2px solid #00d4ff' : '2px solid transparent',
                  transition: '150ms ease',
                  display: 'flex', alignItems: 'center', gap: 6,
                }}
                onMouseEnter={e => { if (activeTab !== tab) e.currentTarget.style.color = '#94a3b8'; }}
                onMouseLeave={e => { if (activeTab !== tab) e.currentTarget.style.color = '#475569'; }}
              >
                {tab}
                {tab === 'LIVE' && setList.length > 0 && (
                  <div style={{
                    width: 6, height: 6, background: '#ef4444',
                    boxShadow: '0 0 6px #ef4444',
                    animation: 'blink 1.5s ease-in-out infinite',
                  }} />
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Right: Status + Icons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 18 }}>
          {/* Sync status */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{
              width: 6, height: 6, background: '#00d4ff',
              boxShadow: '0 0 8px #00d4ff',
              animation: 'statusPulse 2s ease-in-out infinite',
            }} />
            <span style={{ fontSize: 10, color: '#00d4ff', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.1em', fontWeight: 600 }}>
              SYNCED
            </span>
          </div>

          {/* BPM readout */}
          {selectedTrack?.bpm && (
            <span style={{ fontSize: 10, color: '#a855f7', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.08em' }}>
              // {Math.round(selectedTrack.bpm)} BPM
            </span>
          )}

          {/* Track count */}
          <span style={{ fontSize: 9, color: '#2d3748', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.1em' }}>
            {allNodes.length} TRACKS
          </span>

          {/* Icon buttons */}
          <div style={{ display: 'flex', gap: 8, marginLeft: 4 }}>
            <div style={{
              width: 32, height: 32,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              border: '1px solid rgba(124,58,237,0.2)', color: '#475569',
              cursor: 'pointer', transition: '150ms ease',
            }}
            >
              <IconBell />
            </div>
            <div style={{
              width: 32, height: 32,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              border: '1px solid rgba(124,58,237,0.2)', color: '#475569',
              cursor: 'pointer', transition: '150ms ease',
            }}
            >
              <IconUser />
            </div>
          </div>
        </div>
      </div>

      {/* ═══════════ MAIN CONTENT ═══════════ */}
      <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>

        {/* ── LIVE TAB ── */}
        {activeTab === 'LIVE' && (
          <LiveMode
            setList={setList}
            setSetList={setSetList}
            allNodes={allNodes}
          />
        )}

        {/* ── DISCOVERY TAB ── */}
        {activeTab === 'DISCOVERY' && <>

        {/* Grid overlay */}
        <div style={{
          position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 0,
          backgroundImage: 'linear-gradient(rgba(124,58,237,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(124,58,237,0.025) 1px, transparent 1px)',
          backgroundSize: '60px 60px',
        }} />

        {/* 3D Graph */}
        <ForceGraph3D
          ref={fgRef}
          graphData={trackData}
          width={graphDims.w}
          height={graphDims.h}
          backgroundColor="#050507"
          nodeThreeObject={nodeThreeObject}
          linkColor={() => 'rgba(124,58,237,0.18)'}
          linkWidth={() => 0.4}
          controlType="trackball"
          onNodeClick={handleNodeClick}
          onEngineStop={() => {
            fgRef.current?.zoomToFit(400, 100);
            const c = fgRef.current?.controls();
            if (c) { c.maxDistance = 8000; c.minDistance = 10; }
          }}
          nodeLabel={nodeLabel}
        />

        {/* Hidden audio */}
        <audio ref={audioRef} onEnded={() => setIsPlaying(false)} onError={() => setIsPlaying(false)} style={{ display: 'none' }} />

        {/* ── Selected track panel (right side) ───────────────────────── */}
        {selectedTrack && (
          <div style={{
            position: 'absolute', top: 16, right: 16, zIndex: 100, width: 300,
            background: 'rgba(5,5,7,0.94)',
            backdropFilter: 'blur(24px)',
            border: '1px solid rgba(124,58,237,0.3)',
            boxShadow: '0 0 50px rgba(124,58,237,0.12), 0 20px 60px rgba(0,0,0,0.8)',
            animation: 'slideInRight 250ms ease',
          }}>
            {/* Top gradient line */}
            <div style={{ height: 1, background: 'linear-gradient(90deg, #7c3aed, #00d4ff, rgba(0,212,255,0.1))' }} />

            {/* Header label */}
            <div style={{
              padding: '12px 16px 0',
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            }}>
              <span style={{
                fontSize: 8, letterSpacing: '0.3em',
                color: 'rgba(0,212,255,0.6)', fontFamily: "'JetBrains Mono', monospace",
              }}>NOW SELECTED</span>
              <button
                onClick={() => {
                  setSelectedTrack(null);
                  setTrackData({ nodes: allNodes, links: allLinks });
                  if (audioRef.current) { audioRef.current.pause(); setIsPlaying(false); }
                }}
                style={{
                  background: 'none', border: 'none', color: '#475569',
                  cursor: 'pointer', display: 'flex', padding: 4, transition: '150ms ease',
                }}
                onMouseEnter={e => e.currentTarget.style.color = '#ef4444'}
                onMouseLeave={e => e.currentTarget.style.color = '#475569'}
                aria-label="Close panel"
              >
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                  <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>

            <div style={{ padding: '10px 16px 16px' }}>
              {/* Title + Artist */}
              <div style={{ fontSize: 16, fontWeight: 700, color: '#e2e8f0', lineHeight: 1.3, marginBottom: 2, wordBreak: 'break-word' }}>
                {selectedTrack.name}
              </div>
              <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 14, letterSpacing: '0.02em' }}>
                {selectedTrack.artist}
              </div>

              {/* Metadata grid (2x2) */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 14 }}>
                <div style={{
                  background: 'rgba(8,8,20,0.7)', border: '1px solid rgba(124,58,237,0.12)',
                  padding: '10px 12px',
                }}>
                  <div style={{ fontSize: 8, letterSpacing: '0.2em', color: '#475569', fontFamily: "'JetBrains Mono', monospace", marginBottom: 4 }}>TEMPO/BPM</div>
                  <div style={{ fontSize: 20, fontWeight: 700, color: '#00d4ff', fontFamily: "'JetBrains Mono', monospace" }}>
                    {selectedTrack.bpm ? parseFloat(selectedTrack.bpm).toFixed(2) : '—'}
                  </div>
                </div>
                <div style={{
                  background: 'rgba(8,8,20,0.7)', border: '1px solid rgba(124,58,237,0.12)',
                  padding: '10px 12px',
                }}>
                  <div style={{ fontSize: 8, letterSpacing: '0.2em', color: '#475569', fontFamily: "'JetBrains Mono', monospace", marginBottom: 4 }}>HARMONIC KEY</div>
                  <div style={{ fontSize: 20, fontWeight: 700, color: '#a855f7', fontFamily: "'JetBrains Mono', monospace" }}>
                    {selectedTrack.key || '—'}
                  </div>
                </div>
                <div style={{
                  background: 'rgba(8,8,20,0.7)', border: '1px solid rgba(124,58,237,0.12)',
                  padding: '10px 12px',
                }}>
                  <div style={{ fontSize: 8, letterSpacing: '0.2em', color: '#475569', fontFamily: "'JetBrains Mono', monospace", marginBottom: 4 }}>ENERGY</div>
                  <div style={{ fontSize: 20, fontWeight: 700, color: '#7c3aed', fontFamily: "'JetBrains Mono', monospace" }}>
                    {selectedTrack.energy != null ? parseFloat(selectedTrack.energy).toFixed(2) : '—'}
                  </div>
                </div>
                <div style={{
                  background: 'rgba(8,8,20,0.7)', border: '1px solid rgba(124,58,237,0.12)',
                  padding: '10px 12px',
                }}>
                  <div style={{ fontSize: 8, letterSpacing: '0.2em', color: '#475569', fontFamily: "'JetBrains Mono', monospace", marginBottom: 4 }}>GENRE</div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: '#00d4ff', fontFamily: "'JetBrains Mono', monospace", overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {selectedTrack.semanticTags?.[0] || '—'}
                  </div>
                </div>
              </div>

              {/* Waveform visualization */}
              <div style={{
                background: 'rgba(8,8,20,0.5)', border: '1px solid rgba(124,58,237,0.12)',
                padding: '8px 4px', marginBottom: 14,
              }}>
                <WaveformViz bpm={selectedTrack.bpm} isPlaying={isPlaying} />
              </div>

              {/* Action buttons */}
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  onClick={togglePlay}
                  aria-label={isPlaying ? 'Pause' : 'Play'}
                  style={{
                    ...BTN_BASE,
                    flex: 1,
                    border: `1px solid ${isPlaying ? 'rgba(0,212,255,0.5)' : 'rgba(124,58,237,0.5)'}`,
                    color: isPlaying ? '#00d4ff' : '#a855f7',
                    background: isPlaying ? 'rgba(0,212,255,0.06)' : 'rgba(124,58,237,0.06)',
                  }}
                  onMouseEnter={e => { e.currentTarget.style.background = isPlaying ? 'rgba(0,212,255,0.12)' : 'rgba(124,58,237,0.12)'; }}
                  onMouseLeave={e => { e.currentTarget.style.background = isPlaying ? 'rgba(0,212,255,0.06)' : 'rgba(124,58,237,0.06)'; }}
                >
                  {isPlaying ? <IconPause /> : <IconPlay />}
                  {isPlaying ? 'PAUSE' : 'PLAY'}
                </button>
              </div>

              <div style={{ display: 'flex', gap: 8, marginTop: 6 }}>
                <button
                  onClick={() => handleFindSimilar(selectedTrack.id)}
                  aria-label="Find similar tracks"
                  style={{ ...BTN_BASE, flex: 1, border: '1px solid rgba(0,212,255,0.3)', color: '#00d4ff' }}
                  onMouseEnter={e => { e.currentTarget.style.background = 'rgba(0,212,255,0.06)'; }}
                  onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; }}
                >
                  <IconSearch /> FIND SIMILAR
                </button>
                <button
                  onClick={() => {
                    setSetList(prev => {
                      const id = String(selectedTrack.id);
                      if (prev.some(t => String(t.id) === id)) return prev;
                      return [...prev, selectedTrack];
                    });
                    setActiveTab('LIVE');
                  }}
                  aria-label="Add to live set"
                  style={{ ...BTN_BASE, flex: 1, border: '1px solid rgba(239,68,68,0.3)', color: '#ef4444' }}
                  onMouseEnter={e => { e.currentTarget.style.background = 'rgba(239,68,68,0.06)'; }}
                  onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; }}
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true">
                    <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
                  </svg> ADD TO SET
                </button>
              </div>
            </div>
          </div>
        )}

        <DJChatbox
          ref={chatboxRef}
          selectedTrack={selectedTrack}
          trackCount={allNodes.length}
          onTrackSelect={handleChatTrackSelect}
        />
        </>}
      </div>

      {/* ═══════════ BOTTOM STATUS BAR ═══════════ */}
      <div style={{
        height: 32, flexShrink: 0, zIndex: 200,
        background: 'rgba(5,5,7,0.95)',
        borderTop: '1px solid rgba(124,58,237,0.15)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 20px',
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 9, letterSpacing: '0.1em',
      }}>
        <div style={{ display: 'flex', gap: 24, alignItems: 'center' }}>
          <span style={{ color: '#2d3748' }}>
            LATENCY <span style={{ color: '#475569' }}>4.2ms</span>
          </span>
          <span style={{ color: '#2d3748' }}>
            BPM <span style={{ color: selectedTrack ? '#00d4ff' : '#475569' }}>{selectedTrack?.bpm ? Math.round(selectedTrack.bpm) : '—'}</span>
          </span>
          <span style={{ color: '#2d3748' }}>
            BUFFER <span style={{ color: '#475569' }}>2.048</span>
          </span>
          <span style={{ color: '#2d3748' }}>
            KEY <span style={{ color: selectedTrack?.key ? '#a855f7' : '#475569' }}>{selectedTrack?.key || '—'}</span>
          </span>
        </div>
        <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
          <span style={{ color: '#2d3748' }}>
            DB <span style={{ color: '#475569' }}>LINK CONNECTED</span>
          </span>
          <span style={{ color: '#2d3748' }}>
            v0.6.4-BETA
          </span>
        </div>
      </div>
    </div>
  );
}
