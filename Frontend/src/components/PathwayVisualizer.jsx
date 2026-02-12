// Frontend/src/components/PathwayVisualizer.jsx
import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

const PathwayVisualizer = ({ pathwayData, activeTrack }) => {
  const pathRef = useRef();
  const particlesRef = useRef();
  
  // Generate pathway geometry from data
  const pathGeometry = useMemo(() => {
    if (!pathwayData || !pathwayData.pathway_points) return null;
    
    const points = pathwayData.pathway_points.map(point => 
      new THREE.Vector3(point.x, point.y, point.z)
    );
    
    return new THREE.CatmullRomCurve3(points);
  }, [pathwayData]);
  
  // Animated particles along pathway
  const pathParticles = useMemo(() => {
    if (!pathGeometry) return null;
    
    const particleCount = 50;
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    
    for (let i = 0; i < particleCount; i++) {
      const t = i / particleCount;
      const point = pathGeometry.getPoint(t);
      
      positions[i * 3] = point.x;
      positions[i * 3 + 1] = point.y;
      positions[i * 3 + 2] = point.z;
      
      // Color gradient along path
      colors[i * 3] = 1 - t; // Red component
      colors[i * 3 + 1] = t;   // Green component
      colors[i * 3 + 2] = 0.5; // Blue component
    }
    
    return { positions, colors };
  }, [pathGeometry]);
  
  // Animate particles
  useFrame((state) => {
    if (particlesRef.current && pathGeometry) {
      const time = state.clock.elapsedTime;
      const positions = particlesRef.current.geometry.attributes.position.array;
      
      for (let i = 0; i < positions.length / 3; i++) {
        const t = (i / (positions.length / 3) + time * 0.1) % 1;
        const point = pathGeometry.getPoint(t);
        
        positions[i * 3] = point.x;
        positions[i * 3 + 1] = point.y;
        positions[i * 3 + 2] = point.z;
      }
      
      particlesRef.current.geometry.attributes.position.needsUpdate = true;
    }
  });
  
  if (!pathGeometry || !pathParticles) return null;
  
  return (
    <group>
      {/* Pathway line */}
      <mesh ref={pathRef}>
        <tubeGeometry args={[pathGeometry, 100, 0.1, 8, false]} />
        <meshBasicMaterial color="#00ff88" transparent opacity={0.6} />
      </mesh>
      
      {/* Animated particles */}
      <points ref={particlesRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={pathParticles.positions.length / 3}
            array={pathParticles.positions}
            itemSize={3}
          />
          <bufferAttribute
            attach="attributes-color"
            count={pathParticles.colors.length / 3}
            array={pathParticles.colors}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial size={0.3} vertexColors transparent opacity={0.8} />
      </points>
    </group>
  );
};

export default PathwayVisualizer;