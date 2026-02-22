import React, { useEffect, useState, useRef, useCallback } from 'react';
import { createClient } from '@supabase/supabase-js';
import ForceGraph3D from 'react-force-graph-3d';
import * as THREE from 'three';
import DJChatbox from './components/DJChatbox';

const supabase = createClient(
  "https://cvermotfxamubejfnoje.supabase.co",
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImN2ZXJtb3RmeGFtdWJlamZub2plIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTk2NTU4MTcsImV4cCI6MjA3NTIzMTgxN30.clXSFQ4QVhL8nUK_6shyhDVxhKaHUtnrdyqCnDeCCag"
);

const DEFAULT_COVER = "https://placehold.co/400x400/000000/00ffff?text=No+Cover";

export default function App() {
  const fgRef = useRef();
  const chatboxRef = useRef(null);
  const audioRef = useRef(null);
  const [trackData, setTrackData] = useState({ nodes: [], links: [] });
  const [allNodes, setAllNodes] = useState([]);
  const [allLinks, setAllLinks] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedTrack, setSelectedTrack] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    const getData = async () => {
      try {
        setIsLoading(true);
        const { data, error } = await supabase.from('tracks').select('*');
        if (error) throw error;
        if (!data || data.length === 0) throw new Error('No tracks found');

        const nodes = data.map(track => ({
          id:       track.trackid,
          name:     track.title   || 'Unknown Title',
          artist:   track.artist  || 'Unknown Artist',
          bpm:      track.bpm,
          key:      track.key,
          energy:   track.energy,
          albumArt: track.album_art_url || DEFAULT_COVER,
          x:        track.x_coord || (Math.random() - 0.5) * 1000,
          y:        track.y_coord || (Math.random() - 0.5) * 1000,
          z:        track.z_coord || (Math.random() - 0.5) * 1000,
        }));

        const artistMap = {};
        data.forEach(t => {
          const a = t.artist || 'Unknown';
          if (!artistMap[a]) artistMap[a] = [];
          artistMap[a].push(t);
        });
        const links = [];
        Object.values(artistMap).forEach(tracks => {
          for (let i = 0; i < tracks.length - 1; i++)
            links.push({ source: tracks[i].trackid, target: tracks[i+1].trackid });
        });

        setAllNodes(nodes);
        setAllLinks(links);
        setTrackData({ nodes, links });
        setIsLoading(false);
      } catch (err) {
        setError(err.message);
        setIsLoading(false);
      }
    };
    getData();
  }, []);

  // Stop audio when selected track changes
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      setIsPlaying(false);
    }
  }, [selectedTrack]);

  const handleNodeClick = useCallback((node) => {
    if (!fgRef.current) return;
    setSelectedTrack(node);
    // Camera looks directly at the node; position is offset behind and slightly above
    const nodePos = new THREE.Vector3(node.x, node.y, node.z);
    const cameraPos = nodePos.clone().add(new THREE.Vector3(0, 30, 120));
    fgRef.current.cameraPosition(cameraPos, nodePos, 1000);
  }, []);

  // Feature 1: When a chat result is clicked, fly to that node in the 3D cloud
  const handleChatTrackSelect = useCallback((trackid) => {
    // trackid from chat is a string, node.id from Supabase may be int
    const node = allNodes.find(n => String(n.id) === String(trackid));
    if (node) {
      handleNodeClick(node);
    }
  }, [allNodes, handleNodeClick]);

  // Feature 2: Toggle audio playback for the selected track
  const togglePlay = useCallback(() => {
    if (!audioRef.current || !selectedTrack) return;

    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.src = `http://localhost:8000/tracks/${selectedTrack.id}/audio`;
      audioRef.current.play().catch(err => {
        console.error('Audio playback failed:', err);
        setIsPlaying(false);
      });
      setIsPlaying(true);
    }
  }, [isPlaying, selectedTrack]);

  // Feature 3: Open chatbox and find similar tracks (embedding-based, same as streamSimilar.py)
  const handleFindSimilar = useCallback((trackid) => {
    if (chatboxRef.current) {
      chatboxRef.current.openAndFindSimilar(trackid);
    }
  }, []);

  const nodeThreeObject = useCallback((node) => {
    const texture = new THREE.TextureLoader().load(node.albumArt);
    texture.anisotropy = 16;
    const isSelected = selectedTrack?.id === node.id;
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
      map: texture, transparent: true, depthWrite: false, depthTest: false,
      opacity: isSelected ? 1.0 : 0.75,
    }));
    const s = isSelected ? 38 : 28;
    sprite.scale.set(s, s, 1);
    return sprite;
  }, [selectedTrack]);

  if (isLoading) return (
    <div style={{ width:'100vw', height:'100vh', background:'#000', display:'flex', alignItems:'center', justifyContent:'center', flexDirection:'column', color:'#00ffff', fontFamily:'Inter,sans-serif' }}>
      <div style={{ width:'50px', height:'50px', border:'4px solid #333', borderTop:'4px solid #00ffff', borderRadius:'50%', animation:'spin 1s linear infinite', marginBottom:'20px' }} />
      <p>Loading your music cloud...</p>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
    </div>
  );

  if (error) return (
    <div style={{ width:'100vw', height:'100vh', background:'#000', display:'flex', alignItems:'center', justifyContent:'center', flexDirection:'column', color:'#ff4444', fontFamily:'Inter,sans-serif' }}>
      <h1>Error</h1><p>{error}</p>
      <button onClick={() => window.location.reload()} style={{ background:'#00ffff', color:'#000', border:'none', padding:'12px 24px', borderRadius:'6px', cursor:'pointer' }}>Retry</button>
    </div>
  );

  return (
    <div style={{ width:'100vw', height:'100vh', background:'#000' }}>
      <ForceGraph3D
        ref={fgRef}
        graphData={trackData}
        backgroundColor="#000000"
        nodeThreeObject={nodeThreeObject}
        linkColor={() => 'rgba(0,255,255,0.15)'}
        linkWidth={() => 0.5}
        controlType="trackball"
        onNodeClick={handleNodeClick}
        onEngineStop={() => {
          fgRef.current?.zoomToFit(400, 100);
          const c = fgRef.current?.controls();
          if (c) { c.maxDistance = 8000; c.minDistance = 10; }
        }}
        nodeLabel={n => `<div style="color:#00ffff;background:rgba(0,0,0,0.85);padding:10px;border:1px solid #00ffff;border-radius:6px;text-align:center;font-family:Inter,sans-serif;"><strong>${n.name}</strong><br/><span style="font-size:11px;opacity:.8">${n.artist}</span>${n.bpm?`<br/><span style="font-size:10px">${Math.round(n.bpm)} BPM • ${n.key||'?'}</span>`:''}</div>`}
      />

      {/* Hidden audio element for playback */}
      <audio
        ref={audioRef}
        onEnded={() => setIsPlaying(false)}
        onError={() => setIsPlaying(false)}
        style={{ display: 'none' }}
      />

      {/* Selected track info panel */}
      {selectedTrack && (
        <div style={{ position:'fixed', bottom:'20px', right:'20px', background:'rgba(0,0,0,0.9)', border:'1px solid #00ffff', borderRadius:'8px', padding:'14px 18px', color:'#00ffff', fontFamily:'Inter,sans-serif', zIndex:1000, minWidth: 220 }}>
          <div style={{ fontSize:'11px', opacity:0.6, marginBottom:'4px' }}>SELECTED</div>
          <div style={{ fontSize:'15px', fontWeight:'bold' }}>{selectedTrack.name}</div>
          <div style={{ fontSize:'12px', opacity:0.7 }}>{selectedTrack.artist}</div>
          {selectedTrack.bpm && <div style={{ fontSize:'11px', marginTop:'6px', opacity:0.6 }}>{Math.round(selectedTrack.bpm)} BPM • {selectedTrack.key || '?'}</div>}

          {/* Play / Pause */}
          <button
            onClick={togglePlay}
            style={{
              marginTop: '10px', width: '100%', padding: '8px',
              background: isPlaying ? '#002a00' : '#001a2a',
              color: isPlaying ? '#00ff88' : '#00ffff',
              border: `1px solid ${isPlaying ? '#00ff88' : '#00ffff'}`,
              borderRadius: '4px', cursor: 'pointer', fontSize: '12px',
              fontWeight: 600, transition: 'all 0.15s',
            }}
          >
            {isPlaying ? '⏸ Pause' : '▶ Play'}
          </button>

          {/* Find Similar */}
          <button
            onClick={() => handleFindSimilar(selectedTrack.id)}
            style={{
              marginTop: '6px', width: '100%', padding: '8px',
              background: '#1a0a2a',
              color: '#cc88ff',
              border: '1px solid #cc88ff55',
              borderRadius: '4px', cursor: 'pointer', fontSize: '12px',
              fontWeight: 600, transition: 'all 0.15s',
            }}
          >
            Find Similar
          </button>

          {/* Reset View */}
          <button onClick={() => {
            setSelectedTrack(null);
            setTrackData({ nodes: allNodes, links: allLinks });
            if (audioRef.current) { audioRef.current.pause(); setIsPlaying(false); }
          }}
            style={{ marginTop:'6px', width:'100%', padding:'6px', background:'#333', color:'#00ffff', border:'1px solid #00ffff', borderRadius:'4px', cursor:'pointer', fontSize:'11px' }}>
            Reset View
          </button>
        </div>
      )}

      {!selectedTrack && (
        <div style={{ position:'fixed', bottom:'20px', right:'20px', fontSize:'11px', color:'rgba(0,255,255,0.35)', fontFamily:'Inter,sans-serif' }}>
          {allNodes.length} tracks
        </div>
      )}

      <DJChatbox
        ref={chatboxRef}
        selectedTrack={selectedTrack}
        trackCount={allNodes.length}
        onTrackSelect={handleChatTrackSelect}
      />
    </div>
  );
}
