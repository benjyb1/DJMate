import React, { useEffect, useState, useRef, useCallback } from 'react';
import { createClient } from '@supabase/supabase-js';
import ForceGraph3D from 'react-force-graph-3d';
import * as THREE from 'three';
import NaturalLanguageBar from './components/NaturalLanguageBar';
import { useRecommendationAPI } from './hooks/useRecommendationAPI';

// --- CONFIGURATION ---
const supabase = createClient(
  "https://cvermotfxamubejfnoje.supabase.co",
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImN2ZXJtb3RmeGFtdWJlamZub2plIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTk2NTU4MTcsImV4cCI6MjA3NTIzMTgxN30.clXSFQ4QVhL8nUK_6shyhDVxhKaHUtnrdyqCnDeCCag"
);

const DEFAULT_COVER = "https://placehold.co/400x400/000000/00ffff?text=No+Cover";

// Normalise any response shape from getSimilarTracks into flat objects.
// The API can return SimilarityResult { track: TrackMetadata, similarity_score }
// OR plain { id, similarity_score } depending on the code path.
function normaliseSimilar(raw) {
  return (raw || []).map(s => ({
    id:               s.id        || s.track?.trackid || s.track?.id || s.trackid,
    similarity_score: s.similarity_score ?? s.similarity ?? 0.5,
    title:            s.title     || s.track?.title   || 'Unknown',
    artist:           s.artist    || s.track?.artist  || 'Unknown',
  })).filter(s => s.id);
}

export default function App() {
  const fgRef = useRef();
  const [trackData, setTrackData] = useState({ nodes: [], links: [] });
  const [allNodes, setAllNodes] = useState([]);
  const [allLinks, setAllLinks] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const recommendationAPI = useRecommendationAPI();

  const [selectedTrack, setSelectedTrack] = useState(null);
  const [similarTracks, setSimilarTracks] = useState([]);
  const [queryInterpretation, setQueryInterpretation] = useState(null);
  const [isQuerying, setIsQuerying] = useState(false);

  // --- FETCH DATA ---
  useEffect(() => {
    const getData = async () => {
      try {
        setIsLoading(true);
        setError(null);

        const { data, error } = await supabase.from('tracks').select('*');
        if (error) throw error;
        if (!data || data.length === 0) throw new Error('No tracks found');

        console.log(`Loaded ${data.length} tracks`);

        const nodes = data.map(track => ({
          id:       track.trackid,
          name:     track.title    || 'Unknown Title',
          artist:   track.artist   || 'Unknown Artist',
          bpm:      track.bpm,
          key:      track.key,
          energy:   track.energy,
          albumArt: track.album_art_url || DEFAULT_COVER,
          x:        track.x_coord  || (Math.random() - 0.5) * 1000,
          y:        track.y_coord  || (Math.random() - 0.5) * 1000,
          z:        track.z_coord  || (Math.random() - 0.5) * 1000,
        }));

        // Artist connections
        const artistMap = {};
        data.forEach(track => {
          const artist = track.artist || 'Unknown';
          if (!artistMap[artist]) artistMap[artist] = [];
          artistMap[artist].push(track);
        });

        const links = [];
        Object.values(artistMap).forEach(tracks => {
          for (let i = 0; i < tracks.length - 1; i++) {
            links.push({ source: tracks[i].trackid, target: tracks[i + 1].trackid, type: 'artist' });
          }
        });

        setAllNodes(nodes);
        setAllLinks(links);
        setTrackData({ nodes, links });
        setIsLoading(false);
      } catch (err) {
        console.error('Error loading tracks:', err);
        setError(err.message);
        setIsLoading(false);
      }
    };
    getData();
  }, []);

  // --- HANDLE TRACK CLICK ---
  const handleNodeClick = useCallback(async (node) => {
    if (!fgRef.current) return;
    setSelectedTrack(node);
    setQueryInterpretation(null);

    // Fly camera to track
    const distance = 60;
    const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);
    const targetPos = new THREE.Vector3(node.x * distRatio, node.y * distRatio, node.z * distRatio);
    fgRef.current.cameraPosition(
      targetPos.clone().add(new THREE.Vector3(0, 50, 150)),
      targetPos,
      1000
    );

    try {
      const raw = await recommendationAPI.getSimilarTracks(node.id, 8);
      const similar = normaliseSimilar(raw);

      if (similar.length > 0) {
        setSimilarTracks(similar);
        const similarityLinks = similar.map(s => ({
          source: node.id, target: s.id, type: 'similarity', similarity: s.similarity_score,
        }));
        setTrackData(prev => ({ nodes: prev.nodes, links: [...allLinks, ...similarityLinks] }));
      } else {
        setSimilarTracks([]);
      }
    } catch (err) {
      console.error('Failed to fetch similar tracks:', err);
      setSimilarTracks([]);
    }
  }, [recommendationAPI, allLinks]);

  // --- HANDLE NATURAL LANGUAGE QUERY ---
  const handleNaturalLanguageQuery = useCallback(async (queryText) => {
    setIsQuerying(true);
    setSimilarTracks([]);
    setSelectedTrack(null);

    try {
      const interpretation = await recommendationAPI.parseIntent({
        query:   queryText,
        context: selectedTrack ? { current_track: selectedTrack } : null,
      });
      setQueryInterpretation(interpretation);

      const recommendations = await recommendationAPI.getIntelligentRecommendations({
        structured_query: interpretation.structured_query,
        context_track_id: selectedTrack?.id,
      });

      // Normalise: handle array, { recommendations }, or { tracks }
      let recommendedTracks = [];
      if (Array.isArray(recommendations))                         recommendedTracks = recommendations;
      else if (Array.isArray(recommendations?.recommendations))   recommendedTracks = recommendations.recommendations;
      else if (Array.isArray(recommendations?.tracks))            recommendedTracks = recommendations.tracks;

      if (recommendedTracks.length > 0) {
        const ids = recommendedTracks.map(r =>
          r?.track?.id || r?.track?.trackid || r?.id || r?.trackid
        ).filter(Boolean);

        const filteredNodes = allNodes.filter(n => ids.includes(n.id));
        if (filteredNodes.length > 0) {
          setTrackData({ nodes: filteredNodes, links: [] });
          setTimeout(() => fgRef.current?.zoomToFit(400, 100), 100);
        }
      }
    } catch (err) {
      console.error('Query failed:', err);
      alert('Query failed: ' + err.message);
    } finally {
      setIsQuerying(false);
    }
  }, [recommendationAPI, selectedTrack, allNodes]);

  // --- RESET VIEW ---
  const resetView = () => {
    setSelectedTrack(null);
    setSimilarTracks([]);
    setQueryInterpretation(null);
    setTrackData({ nodes: allNodes, links: allLinks });
  };

  // --- NODE RENDERING ---
  const nodeThreeObject = useCallback((node) => {
    const texture = new THREE.TextureLoader().load(node.albumArt);
    texture.anisotropy = 16;
    const isSelected = selectedTrack?.id === node.id;
    const isSimilar  = similarTracks.some(s => s.id === node.id);
    const material = new THREE.SpriteMaterial({
      map: texture, transparent: true, depthWrite: false, depthTest: false,
      opacity: isSelected ? 1 : isSimilar ? 0.9 : 0.7,
    });
    const sprite = new THREE.Sprite(material);
    const scale = isSelected ? 35 : isSimilar ? 32 : 28;
    sprite.scale.set(scale, scale, 1);
    return sprite;
  }, [selectedTrack, similarTracks]);

  const linkColor = useCallback((link) =>
    link.type === 'similarity' ? `rgba(0,255,136,${link.similarity || 0.5})` : 'rgba(0,255,255,0.2)',
  []);

  const linkWidth = useCallback((link) => link.type === 'similarity' ? 2 : 0.5, []);

  // --- LOADING / ERROR ---
  if (isLoading) return (
    <div style={{ width:'100vw', height:'100vh', background:'#000', display:'flex', alignItems:'center', justifyContent:'center', flexDirection:'column', color:'#00ffff', fontFamily:'Inter,sans-serif' }}>
      <div style={{ width:'50px', height:'50px', border:'4px solid #333', borderTop:'4px solid #00ffff', borderRadius:'50%', animation:'spin 1s linear infinite', marginBottom:'20px' }} />
      <p>Loading your music cloud...</p>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );

  if (error) return (
    <div style={{ width:'100vw', height:'100vh', background:'#000', display:'flex', alignItems:'center', justifyContent:'center', flexDirection:'column', color:'#ff4444', fontFamily:'Inter,sans-serif' }}>
      <h1>Error Loading Tracks</h1>
      <p>{error}</p>
      <button onClick={() => window.location.reload()} style={{ background:'#00ffff', color:'#000', border:'none', padding:'12px 24px', borderRadius:'6px', cursor:'pointer' }}>Retry</button>
    </div>
  );

  // --- MAIN UI ---
  return (
    <div style={{ width:'100vw', height:'100vh', background:'#000', margin:0, padding:0 }}>
      <div style={{ position:'fixed', top:0, left:0, right:0, zIndex:1000 }}>
        <NaturalLanguageBar onQuery={handleNaturalLanguageQuery} isLoading={isQuerying} lastInterpretation={queryInterpretation} />
      </div>

      <ForceGraph3D
        ref={fgRef}
        graphData={trackData}
        backgroundColor="#000000"
        nodeThreeObject={nodeThreeObject}
        linkColor={linkColor}
        linkWidth={linkWidth}
        controlType="trackball"
        onNodeClick={handleNodeClick}
        onEngineStop={() => {
          if (fgRef.current) {
            fgRef.current.zoomToFit(400, 100);
            const c = fgRef.current.controls();
            if (c) { c.maxDistance = 8000; c.minDistance = 10; }
          }
        }}
        nodeLabel={node => {
          const sim = similarTracks.find(s => s.id === node.id);
          const pct = sim ? ` • ${Math.round(sim.similarity_score * 100)}% match` : '';
          return `<div style="color:#00ffff;background:rgba(0,0,0,0.85);padding:10px;border:1px solid #00ffff;border-radius:6px;text-align:center;font-family:Inter,sans-serif;">
            <strong style="font-size:14px;">${node.name}</strong><br/>
            <span style="font-size:11px;opacity:0.8;">${node.artist}</span>
            ${node.bpm ? `<br/><span style="font-size:10px;">${Math.round(node.bpm)} BPM • ${node.key || '?'}</span>` : ''}
            ${pct ? `<br/><span style="font-size:10px;color:#00ff88;">${pct}</span>` : ''}
          </div>`;
        }}
      />

      {/* Info Panel */}
      <div style={{ position:'fixed', bottom:'20px', left:'20px', background:'rgba(0,0,0,0.9)', border:'1px solid #00ffff', borderRadius:'8px', padding:'15px', maxWidth:'350px', color:'#00ffff', fontFamily:'Inter,sans-serif', zIndex:1000 }}>
        {selectedTrack && (
          <div style={{ padding:'10px', background:'#111', borderRadius:'4px', borderLeft:'3px solid #00ff88', marginBottom:'10px' }}>
            <div style={{ fontSize:'12px', fontWeight:'bold', opacity:0.7 }}>Selected:</div>
            <div style={{ fontSize:'14px', marginTop:'4px' }}>{selectedTrack.name}</div>
            <div style={{ fontSize:'11px', opacity:0.7 }}>{selectedTrack.artist}</div>
            {selectedTrack.bpm && <div style={{ fontSize:'11px', marginTop:'4px' }}>{Math.round(selectedTrack.bpm)} BPM • {selectedTrack.key || '?'}</div>}
            {similarTracks.length > 0 && <div style={{ fontSize:'11px', marginTop:'6px', color:'#00ff88' }}>{similarTracks.length} similar tracks highlighted</div>}
          </div>
        )}
        <div style={{ display:'flex', gap:'8px', marginTop:'10px' }}>
          <button onClick={resetView} style={{ flex:1, padding:'8px', background:'#333', color:'#00ffff', border:'1px solid #00ffff', borderRadius:'4px', cursor:'pointer', fontSize:'12px' }}>Reset View</button>
        </div>
        <div style={{ fontSize:'11px', opacity:0.5, marginTop:'10px' }}>Showing {trackData.nodes.length} of {allNodes.length} tracks</div>
      </div>
    </div>
  );
}