import React, { useEffect, useState, useRef, useCallback } from 'react';
import { createClient } from '@supabase/supabase-js';
import ForceGraph3D from 'react-force-graph-3d';
import * as THREE from 'three';

// --- CONFIGURATION ---
const supabase = createClient(
  "https://cvermotfxamubejfnoje.supabase.co", 
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImN2ZXJtb3RmeGFtdWJlamZub2plIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTk2NTU4MTcsImV4cCI6MjA3NTIzMTgxN30.clXSFQ4QVhL8nUK_6shyhDVxhKaHUtnrdyqCnDeCCag" 
);

const DEFAULT_COVER = "https://placehold.co/400x400/000000/00ffff?text=No+Cover";

export default function App() {
  const fgRef = useRef();
  const [trackData, setTrackData] = useState({ nodes: [], links: [] });

  // --- 1. FETCH DATA ---
  useEffect(() => {
    const getData = async () => {
      const { data, error } = await supabase.from('tracks').select('*');
      if (error) return console.error("Supabase error:", error);

      if (data && data.length > 0) {
        const artistMap = {};
        data.forEach(track => {
          const artist = track.artist || "Unknown";
          if (!artistMap[artist]) artistMap[artist] = [];
          artistMap[artist].push(track);
        });

        const nodes = data.map(track => ({
          id: track.trackid,
          name: track.title || "Unknown Title",
          artist: track.artist || "Unknown Artist",
          albumArt: track.album_art_url || DEFAULT_COVER,
          x: track.x_coord || (Math.random() - 0.5) * 1000,
          y: track.y_coord || (Math.random() - 0.5) * 1000,
          z: track.z_coord || (Math.random() - 0.5) * 1000,
        }));

        const links = [];
        Object.values(artistMap).forEach(tracks => {
          for (let i = 0; i < tracks.length - 1; i++) {
            links.push({
              source: tracks[i].trackid,
              target: tracks[i + 1].trackid,
            });
          }
        });

        setTrackData({ nodes, links });
      }
    };
    getData();
  }, []);

  // --- 2. NODE SPRITE ---
  const nodeThreeObject = useCallback((node) => {
    const textureLoader = new THREE.TextureLoader();
    const texture = textureLoader.load(node.albumArt);
    texture.anisotropy = 16;

    const material = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthWrite: false,
      depthTest: false
    });

    const sprite = new THREE.Sprite(material);
    sprite.scale.set(28, 28, 1);
    return sprite;
  }, []);

  // --- 3. CAMERA FLY-TO NODE ---
  const handleNodeClick = (node) => {
    if (!fgRef.current) return;
    const distance = 60;
    const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);
    const targetPos = new THREE.Vector3(node.x * distRatio, node.y * distRatio, node.z * distRatio);
    fgRef.current.cameraPosition(targetPos.clone().add(new THREE.Vector3(0, 50, 150)), targetPos, 1000);
  };

  return (
    <div style={{ width: '100vw', height: '100vh', background: '#000', margin: 0, padding: 0 }}>
      <ForceGraph3D
        ref={fgRef}
        graphData={trackData}
        backgroundColor="#000000"
        nodeThreeObject={nodeThreeObject}
        linkColor={() => 'rgba(0, 255, 255, 0.2)'}
        linkWidth={0.5}
        controlType="trackball"
        onNodeClick={handleNodeClick}
        onEngineStop={() => {
          if (fgRef.current) {
            fgRef.current.zoomToFit(400, 100);
            const controls = fgRef.current.controls();
            if (controls) {
              controls.maxDistance = 8000;
              controls.minDistance = 10;
            }
          }
        }}
        nodeLabel={node => `
          <div style="
            color: #00ffff;
            background: rgba(0,0,0,0.85);
            padding: 10px;
            border: 1px solid #00ffff;
            border-radius: 6px;
            text-align: center;
            font-family: 'Inter', sans-serif;
          ">
            <strong style="font-size: 14px;">${node.name}</strong><br/>
            <span style="font-size: 11px; opacity: 0.8;">${node.artist}</span>
          </div>
        `}
      />
    </div>
  );
}
