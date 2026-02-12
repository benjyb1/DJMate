// Frontend/src/components/MusicCloud3D.jsx
import React, { useRef, useMemo, useState, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { Text, Billboard, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

const TrackNode = ({ 
  track, 
  position, 
  isActive, 
  isHighlighted, 
  onClick, 
  onDragStart 
}) => {
  const meshRef = useRef();
  const [hovered, setHovered] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  
  // Dynamic sizing based on energy and activity state
  const nodeSize = useMemo(() => {
    const baseSize = 0.5;
    const energyMultiplier = (track.energy || 0.5) * 0.5 + 0.75;
    const activeMultiplier = isActive ? 1.5 : 1;
    const highlightMultiplier = isHighlighted ? 1.3 : 1;
    return baseSize * energyMultiplier * activeMultiplier * highlightMultiplier;
  }, [track.energy, isActive, isHighlighted]);

  // Color mapping based on track properties
  const nodeColor = useMemo(() => {
    if (isActive) return '#00ff88';
    if (isHighlighted) return '#ff6b35';
    if (hovered) return '#ffffff';
    
    // Color by BPM range
    const bpm = track.bmp || 120;
    if (bpm < 110) return '#4a90e2'; // Blue for slower tracks
    if (bpm < 125) return '#7b68ee'; // Purple for mid-tempo
    if (bpm < 140) return '#ff6b6b'; // Red for faster tracks
    return '#ff1744'; // Bright red for very fast tracks
  }, [track.bpm, isActive, isHighlighted, hovered]);

  // Animation frame updates
  useFrame((state) => {
    if (meshRef.current) {
      // Gentle floating animation
      meshRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime + position[0]) * 0.1;
      
      // Rotation based on energy
      const rotationSpeed = (track.energy || 0.5) * 0.5;
      meshRef.current.rotation.z += rotationSpeed * 0.01;
      
      // Pulsing effect for highlighted tracks
      if (isHighlighted) {
        const pulse = Math.sin(state.clock.elapsedTime * 4) * 0.1 + 1;
        meshRef.current.scale.setScalar(nodeSize * pulse);
      } else {
        meshRef.current.scale.setScalar(nodeSize);
      }
    }
  });

  const handleClick = (event) => {
    event.stopPropagation();
    onClick(track);
  };

  const handlePointerDown = (event) => {
    event.stopPropagation();
    setIsDragging(true);
    onDragStart(track, event);
  };

  return (
    <group position={position}>
      <mesh
        ref={meshRef}
        onClick={handleClick}
        onPointerDown={handlePointerDown}
        onPointerEnter={() => setHovered(true)}
        onPointerLeave={() => setHovered(false)}
      >
        {/* Main node geometry */}
        <icosahedronGeometry args={[1, 1]} />
        <meshStandardMaterial
          color={nodeColor}
          emissive={nodeColor}
          emissiveIntensity={isHighlighted ? 0.3 : 0.1}
          transparent
          opacity={isDragging ? 0.7 : 0.9}
        />
      </mesh>
      
      {/* Track information overlay */}
      {(hovered || isActive) && (
        <Billboard position={[0, 1.5, 0]}>
          <Text
            fontSize={0.3}
            color="white"
            anchorX="center"
            anchorY="middle"
            maxWidth={6}
          >
            {`${track.artist} - ${track.title}`}
            {'\n'}
            {`${Math.round(track.bpm || 120)} BPM | ${track.key || '?'} | E${((track.energy || 0.5) * 10).toFixed(1)}`}
          </Text>
        </Billboard>
      )}
      
      {/* Genre/tag indicators */}
      {track.semantic_tags && track.semantic_tags.slice(0, 2).map((tag, index) => (
        <Billboard key={tag} position={[0, -1.2 - (index * 0.3), 0]}>
          <Text
            fontSize={0.2}
            color="#888"
            anchorX="center"
            anchorY="middle"
          >
            {tag}
          </Text>
        </Billboard>
      ))}
    </group>
  );
};

const ConnectionLines = ({ tracks, activeTrack, highlightedTracks, pathwayData }) => {
  const linesRef = useRef();
  
  const connectionGeometry = useMemo(() => {
    if (!activeTrack || !highlightedTracks.length) return null;
    
    const points = [];
    const activePosition = activeTrack.position;
    
    highlightedTracks.forEach(track => {
      if (track.position && track.id !== activeTrack.id) {
        points.push(new THREE.Vector3(...activePosition));
        points.push(new THREE.Vector3(...track.position));
      }
    });
    
    return new THREE.BufferGeometry().setFromPoints(points);
  }, [activeTrack, highlightedTracks]);
  
  if (!connectionGeometry) return null;
  
  return (
    <lineSegments ref={linesRef} geometry={connectionGeometry}>
      <lineBasicMaterial color="#00ff88" transparent opacity={0.3} />
    </lineSegments>
  );
};

const MusicCloud3D = ({ 
  tracks = [], 
  activeTrack, 
  highlightedTracks = [], 
  pathwayData,
  onTrackSelect,
  onTrackDragToCrate
}) => {
  const { camera } = useThree();
  const [draggedTrack, setDraggedTrack] = useState(null);
  const [isDragMode, setIsDragMode] = useState(false);
  
  // UMAP-based positioning (simulated for now - you'll connect to your actual UMAP data)
  const trackPositions = useMemo(() => {
    if (!tracks.length) return {};
    
    // For now, generate positions based on track properties
    // In production, you'll use your UMAP coordinates
    const positions = {};
    
    tracks.forEach((track, index) => {
      // Simulate UMAP coordinates based on audio features
      const bpm = track.bpm || 120;
      const energy = track.energy || 0.5;
      
      // Create clusters based on BPM and energy
      const bpmNormalized = (bpm - 80) / 100; // Normalize BPM to 0-1 range
      const angle = (index / tracks.length) * Math.PI * 2;
      const radius = energy * 15 + 5;
      
      positions[track.id || track.trackid] = [
        Math.cos(angle) * radius + (bpmNormalized - 0.5) * 20,
        (energy - 0.5) * 20,
        Math.sin(angle) * radius
      ];
    });
    
    return positions;
  }, [tracks]);
  
  // Enhanced track data with positions
  const enhancedTracks = useMemo(() => {
    return tracks.map(track => ({
      ...track,
      position: trackPositions[track.id || track.trackid] || [0, 0, 0]
    }));
  }, [tracks, trackPositions]);

  const handleTrackClick = (track) => {
    if (isDragMode) return;
    onTrackSelect(track.id || track.trackid);
  };

  const handleDragStart = (track, event) => {
    setDraggedTrack(track);
    setIsDragMode(true);
    
    // Visual feedback for drag operation
    document.body.style.cursor = 'grabbing';
  };

  const handleDragEnd = (event) => {
    if (draggedTrack) {
      // Check if dropped in crate area (you'll implement drop zone detection)
      const dropZone = detectDropZone(event);
      if (dropZone === 'crate') {
        onTrackDragToCrate(draggedTrack.id || draggedTrack.trackid);
      }
    }
    
    setDraggedTrack(null);
    setIsDragMode(false);
    document.body.style.cursor = 'default';
  };

  const detectDropZone = (event) => {
    // Implement drop zone detection logic
    // For now, return 'crate' if dragged to right side of screen
    const screenWidth = window.innerWidth;
    return event.clientX > screenWidth * 0.7 ? 'crate' : null;
  };

  // Camera controls and interactions
  useEffect(() => {
    const handleKeyDown = (event) => {
      switch (event.key) {
        case 'r':
          // Reset camera position
          camera.position.set(0, 0, 50);
          camera.lookAt(0, 0, 0);
          break;
        case 'f':
          // Focus on active track
          if (activeTrack && activeTrack.position) {
            camera.position.set(
              activeTrack.position[0] + 10,
              activeTrack.position[1] + 10,
              activeTrack.position[2] + 10
            );
            camera.lookAt(...activeTrack.position);
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [camera, activeTrack]);

  return (
    <>
      {/* Lighting setup */}
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={0.8} />
      <pointLight position={[-10, -10, -10]} intensity={0.4} color="#4a90e2" />
      
      {/* Camera controls */}
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        maxDistance={200}
        minDistance={5}
      />
      
      {/* Track nodes */}
      {enhancedTracks.map((track) => (
        <TrackNode
          key={track.id || track.trackid}
          track={track}
          position={track.position}
          isActive={activeTrack && (activeTrack.id === track.id || activeTrack.trackid === track.trackid)}
          isHighlighted={highlightedTracks.includes(track.id || track.trackid)}
          onClick={handleTrackClick}
          onDragStart={handleDragStart}
        />
      ))}
      
      {/* Connection lines between related tracks */}
      <ConnectionLines
        tracks={enhancedTracks}
        activeTrack={activeTrack}
        highlightedTracks={highlightedTracks.map(id => 
          enhancedTracks.find(t => (t.id || t.trackid) === id)
        ).filter(Boolean)}
        pathwayData={pathwayData}
      />
      
      {/* Background grid for spatial reference */}
      <gridHelper args={[100, 20, '#333333', '#333333']} position={[0, -25, 0]} />
      
      {/* Drag and drop visual feedback */}
      {isDragMode && (
        <mesh position={[25, 0, 0]}>
          <boxGeometry args={[2, 2, 2]} />
          <meshBasicMaterial color="#00ff88" transparent opacity={0.3} />
        </mesh>
      )}
    </>
  );
};

export default MusicCloud3D;