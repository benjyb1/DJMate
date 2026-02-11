import React, { useState, useEffect, useCallback } from 'react';
import { Canvas } from '@react-three/fiber';
import { Suspense } from 'react';

import MusicCloud3D from './components/MusicCloud3D';
import CrateInspector from './components/CrateInspector';
import NaturalLanguageBar from './components/NaturalLanguageBar';
import PathwayVisualizer from './components/PathwayVisualizer';
import CompatibilityIndicator from './components/CompatibilityIndicator';

import { useRecommendationAPI } from './hooks/useRecommendationAPI';
import { useCrateManager } from './hooks/useCrateManager';

function App() {
  // Core state management
  const [musicLibrary, setMusicLibrary] = useState([]);
  const [activeTrack, setActiveTrack] = useState(null);
  const [highlightedTracks, setHighlightedTracks] = useState([]);
  const [pathwayData, setPathwayData] = useState(null);
  const [queryState, setQueryState] = useState({
    isLoading: false,
    lastQuery: '',
    interpretation: null
  });

  // Custom hooks for API management
  const recommendationAPI = useRecommendationAPI();
  const crateManager = useCrateManager();

  // Handle natural language queries
  const handleNaturalLanguageQuery = useCallback(async (queryText) => {
    setQueryState(prev => ({ ...prev, isLoading: true, lastQuery: queryText }));
    
    try {
      // Step 1: Parse intent
      const interpretation = await recommendationAPI.parseIntent({
        query: queryText,
        context: activeTrack ? { current_track: activeTrack } : null,
        session_id: crateManager.sessionId
      });

      // Step 2: Get recommendations
      const recommendations = await recommendationAPI.getIntelligentRecommendations({
        structured_query: interpretation.structured_query,
        context_track_id: activeTrack?.id
      });

      // Step 3: Update visualization
      setHighlightedTracks(recommendations.recommendations.map(r => r.track.id));
      setPathwayData(recommendations.pathway_data);
      setQueryState(prev => ({ 
        ...prev, 
        isLoading: false, 
        interpretation: interpretation 
      }));

      // Step 4: Provide visual feedback
      showQueryResults(recommendations);

    } catch (error) {
      console.error('Query processing failed:', error);
      setQueryState(prev => ({ ...prev, isLoading: false }));
    }
  }, [activeTrack, recommendationAPI, crateManager.sessionId]);

  // Handle track selection from 3D cloud
  const handleTrackSelection = useCallback(async (trackId) => {
    const track = musicLibrary.find(t => t.id === trackId);
    setActiveTrack(track);
    
    // Get automatic recommendations for context
    if (track) {
      const recommendations = await recommendationAPI.getSimilarTracks(trackId);
      setHighlightedTracks(recommendations.map(r => r.id));
    }
  }, [musicLibrary, recommendationAPI]);

  // Handle crate operations
  const handleCrateOperation = useCallback(async (operation, trackId) => {
    switch (operation) {
      case 'add':
        await crateManager.addTrack(trackId);
        break;
      case 'remove':
        await crateManager.removeTrack(trackId);
        break;
      case 'reorder':
        await crateManager.reorderTracks(trackId.newOrder);
        break;
    }
  }, [crateManager]);

  return (
    <div className="app-container">
      {/* Natural Language Interface */}
      <div className="top-bar">
        <NaturalLanguageBar
          onQuery={handleNaturalLanguageQuery}
          isLoading={queryState.isLoading}
          lastInterpretation={queryState.interpretation}
        />
      </div>

      <div className="main-content">
        {/* 3D Visualization */}
        <div className="visualization-panel">
          <Canvas camera={{ position: [0, 0, 50], fov: 75 }}>
            <Suspense fallback={null}>
              <MusicCloud3D
                tracks={musicLibrary}
                activeTrack={activeTrack}
                highlightedTracks={highlightedTracks}
                pathwayData={pathwayData}
                onTrackSelect={handleTrackSelection}
                onTrackDragToCrate={(trackId) => handleCrateOperation('add', trackId)}
              />
              
              {pathwayData && (
                <PathwayVisualizer
                  pathwayData={pathwayData}
                  activeTrack={activeTrack}
                />
              )}
            </Suspense>
          </Canvas>
        </div>

        {/* Crate Inspector Sidebar */}
        <div className="crate-panel">
          <CrateInspector
            crate={crateManager.currentCrate}
            onTrackRemove={(trackId) => handleCrateOperation('remove', trackId)}
            onTrackReorder={(newOrder) => handleCrateOperation('reorder', { newOrder })}
            onSequenceValidation={crateManager.validateSequence}
            compatibilityData={crateManager.compatibilityData}
          />
        </div>
      </div>

      {/* Status and compatibility indicators */}
      <div className="status-bar">
        <CompatibilityIndicator
          activeTrack={activeTrack}
          crateSequence={crateManager.currentCrate}
          validationResults={crateManager.validationResults}
        />
      </div>
    </div>
  );
}

export default App;