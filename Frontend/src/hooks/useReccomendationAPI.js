// src/hooks/useRecommendationAPI.js
import { useState, useCallback, useRef } from 'react';
import { apiClient, APIError } from '../api/apiClient';

export const useRecommendationAPI = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastQuery, setLastQuery] = useState(null);
  
  // Use ref to track active requests to prevent race conditions
  const activeRequestRef = useRef(null);

  // Clear error state
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // Parse natural language intent
  const parseIntent = useCallback(async (queryData) => {
    const requestId = Date.now();
    activeRequestRef.current = requestId;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const result = await apiClient.post('/parse-intent', queryData);
      
      // Check if this is still the active request
      if (activeRequestRef.current !== requestId) {
        return null; // Request was superseded
      }
      
      setLastQuery(queryData.query);
      return result;
    } catch (err) {
      if (activeRequestRef.current === requestId) {
        setError(err.message);
        console.error('Intent parsing failed:', err);
      }
      throw err;
    } finally {
      if (activeRequestRef.current === requestId) {
        setIsLoading(false);
      }
    }
  }, []);

  // Get intelligent recommendations
  const getIntelligentRecommendations = useCallback(async (requestData) => {
    const requestId = Date.now();
    activeRequestRef.current = requestId;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const result = await apiClient.post('/intelligent-recommend', requestData);
      
      if (activeRequestRef.current !== requestId) {
        return null;
      }
      
      return result;
    } catch (err) {
      if (activeRequestRef.current === requestId) {
        setError(err.message);
        console.error('Intelligent recommendations failed:', err);
      }
      throw err;
    } finally {
      if (activeRequestRef.current === requestId) {
        setIsLoading(false);
      }
    }
  }, []);

  // Get similar tracks (simplified recommendation)
  const getSimilarTracks = useCallback(async (trackId, options = {}) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Create a basic structured query for similarity
      const structuredQuery = {
        tags: [],
        vibe_descriptors: [],
        exclude_tracks: [trackId], // Don't include the source track
        ...options
      };
      
      const result = await getIntelligentRecommendations({
        structured_query: structuredQuery,
        context_track_id: trackId
      });
      
      return result?.recommendations || [];
    } catch (err) {
      setError(err.message);
      console.error('Similar tracks fetch failed:', err);
      return [];
    } finally {
      setIsLoading(false);
    }
  }, [getIntelligentRecommendations]);

  // Get all tracks for visualization
  const getAllTracks = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const result = await apiClient.get('/tracks');
      return result.tracks || [];
    } catch (err) {
      setError(err.message);
      console.error('Failed to fetch all tracks:', err);
      return [];
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Get pathway visualization data
  const getPathwayVisualization = useCallback(async (fromTrack, toTracks) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const result = await apiClient.get('/visualization/pathway', {
        from_track: fromTrack,
        to_tracks: toTracks.join(',')
      });
      
      return result;
    } catch (err) {
      setError(err.message);
      console.error('Pathway visualization failed:', err);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Health check
  const healthCheck = useCallback(async () => {
    try {
      const result = await apiClient.get('/');
      return result.status === 'healthy';
    } catch (err) {
      console.error('Health check failed:', err);
      return false;
    }
  }, []);

  return {
    // Core methods
    parseIntent,
    getIntelligentRecommendations,
    getSimilarTracks,
    getAllTracks,
    getPathwayVisualization,
    healthCheck,
    
    // State
    isLoading,
    error,
    lastQuery,
    
    // Utilities
    clearError,
  };
};