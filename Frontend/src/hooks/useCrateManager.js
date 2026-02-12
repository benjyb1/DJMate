// src/hooks/useCrateManager.js
import { useState, useCallback, useEffect, useRef } from 'react';
import { apiClient, APIError } from '../api/apiClient';

// Generate session ID
const generateSessionId = () => {
  return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

export const useCrateManager = () => {
  const [sessionId] = useState(() => generateSessionId());
  const [currentCrate, setCurrentCrate] = useState([]);
  const [compatibilityData, setCompatibilityData] = useState(null);
  const [validationResults, setValidationResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Track crate operations to prevent conflicts
  const operationInProgressRef = useRef(false);

  // Validate current sequence
  const validateSequence = useCallback(async (trackIds = null) => {
    const tracksToValidate = trackIds || currentCrate.map(track => track.id);
    
    if (tracksToValidate.length < 2) {
      setValidationResults(null);
      return null;
    }

    setIsLoading(true);
    setError(null);

    try {
      const operation = {
        session_id: sessionId,
        tracks: tracksToValidate,
        sequence_order: tracksToValidate.map((_, index) => index),
        metadata: {
          validation_timestamp: Date.now(),
          crate_size: tracksToValidate.length
        }
      };

      const result = await apiClient.post('/crate/operations', operation);
      
      const validationData = {
        ...result.compatibility_issues,
        sequence_score: result.sequence_score,
        last_validated: Date.now()
      };

      setValidationResults(validationData);
      return validationData;
    } catch (err) {
      setError(err.message);
      console.error('Sequence validation failed:', err);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [currentCrate, sessionId]);

  // Add track to crate
  const addTrack = useCallback(async (trackData) => {
    if (operationInProgressRef.current) {
      console.warn('Crate operation already in progress');
      return false;
    }

    operationInProgressRef.current = true;
    setError(null);

    try {
      // Ensure trackData has required fields
      const normalizedTrack = {
        id: trackData.id || trackData.trackid,
        title: trackData.title || 'Unknown Title',
        artist: trackData.artist || 'Unknown Artist',
        bpm: trackData.bpm || 120,
        key: trackData.key || 'Unknown',
        energy: trackData.energy || 0.5,
        tags: trackData.semantic_tags || trackData.tags || [],
        vibe: trackData.vibe || [],
        album_art: trackData.album_art || null,
        ...trackData
      };

      // Check for duplicates
      const isDuplicate = currentCrate.some(track => track.id === normalizedTrack.id);
      if (isDuplicate) {
        console.warn('Track already in crate:', normalizedTrack.title);
        return false;
      }

      const updatedCrate = [...currentCrate, normalizedTrack];
      setCurrentCrate(updatedCrate);

      // Auto-validate if we have multiple tracks
      if (updatedCrate.length > 1) {
        await validateSequence(updatedCrate.map(t => t.id));
      }

      return true;
    } catch (err) {
      setError(err.message);
      console.error('Failed to add track to crate:', err);
      return false;
    } finally {
      operationInProgressRef.current = false;
    }
  }, [currentCrate, validateSequence]);

  // Remove track from crate
  const removeTrack = useCallback(async (trackId) => {
    if (operationInProgressRef.current) {
      console.warn('Crate operation already in progress');
      return false;
    }

    operationInProgressRef.current = true;
    setError(null);

    try {
      const updatedCrate = currentCrate.filter(track => track.id !== trackId);
      setCurrentCrate(updatedCrate);

      // Revalidate sequence
      if (updatedCrate.length > 1) {
        await validateSequence(updatedCrate.map(t => t.id));
      } else {
        setValidationResults(null);
      }

      return true;
    } catch (err) {
      setError(err.message);
      console.error('Failed to remove track from crate:', err);
      return false;
    } finally {
      operationInProgressRef.current = false;
    }
  }, [currentCrate, validateSequence]);

  // Reorder tracks in crate
  const reorderTracks = useCallback(async (newOrder) => {
    if (operationInProgressRef.current) {
      console.warn('Crate operation already in progress');
      return false;
    }

    operationInProgressRef.current = true;
    setError(null);

    try {
      // newOrder should be an array of track objects in the desired sequence
      setCurrentCrate(newOrder);

      // Validate new sequence
      if (newOrder.length > 1) {
        await validateSequence(newOrder.map(t => t.id));
      }

      return true;
    } catch (err) {
      setError(err.message);
      console.error('Failed to reorder crate:', err);
      return false;
    } finally {
      operationInProgressRef.current = false;
    }
  }, [validateSequence]);

  // Clear entire crate
  const clearCrate = useCallback(() => {
    setCurrentCrate([]);
    setValidationResults(null);
    setCompatibilityData(null);
    setError(null);
  }, []);

  // Get crate statistics
  const getCrateStats = useCallback(() => {
    if (currentCrate.length === 0) {
      return {
        totalTracks: 0,
        totalDuration: 0,
        averageBPM: 0,
        keyDistribution: {},
        energyRange: [0, 0],
        genreDistribution: {}
      };
    }

    const totalTracks = currentCrate.length;
    const bpms = currentCrate.map(t => t.bpm || 120);
    const energies = currentCrate.map(t => t.energy || 0.5);
    
    // Calculate key distribution
    const keyDistribution = currentCrate.reduce((acc, track) => {
      const key = track.key || 'Unknown';
      acc[key] = (acc[key] || 0) + 1;
      return acc;
    }, {});

    // Calculate genre distribution from tags
    const genreDistribution = currentCrate.reduce((acc, track) => {
      const tags = track.tags || [];
      tags.forEach(tag => {
        acc[tag] = (acc[tag] || 0) + 1;
      });
      return acc;
    }, {});

    return {
      totalTracks,
      totalDuration: totalTracks * 4, // Rough estimate: 4 minutes per track
      averageBPM: Math.round(bpms.reduce((a, b) => a + b, 0) / bpms.length),
      keyDistribution,
      energyRange: [Math.min(...energies), Math.max(...energies)],
      genreDistribution,
      validationScore: validationResults?.sequence_score || 0
    };
  }, [currentCrate, validationResults]);

  // Export crate as playlist
  const exportPlaylist = useCallback((format = 'json') => {
    const playlist = {
      session_id: sessionId,
      created_at: new Date().toISOString(),
      tracks: currentCrate,
      validation_results: validationResults,
      stats: getCrateStats()
    };

    switch (format) {
      case 'json':
        return JSON.stringify(playlist, null, 2);
      case 'm3u':
        return currentCrate.map(track => 
          `#EXTINF:${track.duration || 240},${track.artist} - ${track.title}\n${track.filepath || ''}`
        ).join('\n');
      default:
        return playlist;
    }
  }, [sessionId, currentCrate, validationResults, getCrateStats]);

  // Auto-save crate to localStorage
  useEffect(() => {
    const crateData = {
      sessionId,
      currentCrate,
      validationResults,
      lastModified: Date.now()
    };
    
    localStorage.setItem('ai_dj_crate', JSON.stringify(crateData));
  }, [sessionId, currentCrate, validationResults]);

  // Load crate from localStorage on mount
  useEffect(() => {
    try {
      const savedCrate = localStorage.getItem('ai_dj_crate');
      if (savedCrate) {
        const crateData = JSON.parse(savedCrate);
        // Only restore if it's recent (less than 24 hours old)
        if (Date.now() - crateData.lastModified < 24 * 60 * 60 * 1000) {
          setCurrentCrate(crateData.currentCrate || []);
          setValidationResults(crateData.validationResults);
        }
      }
    } catch (err) {
      console.warn('Failed to restore crate from localStorage:', err);
    }
  }, []);

  return {
    // State
    sessionId,
    currentCrate,
    compatibilityData,
    validationResults,
    isLoading,
    error,
    
    // Operations
    addTrack,
    removeTrack,
    reorderTracks,
    clearCrate,
    validateSequence,
    
    // Utilities
    getCrateStats,
    exportPlaylist,
  };
};