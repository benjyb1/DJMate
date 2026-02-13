// hooks/useRecommendationAPI.js - Fixed version with pagination support
import { useState, useCallback } from 'react';
import { apiClient } from '../api/apiClient';

export const useRecommendationAPI = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  /**
   * Get all tracks with pagination support
   * IMPORTANT: Use this method instead of the old getAllTracks
   * 
   * @param {number} limit - Tracks per request (default 100)
   * @param {number} offset - Starting position (for manual pagination)
   * @returns {Promise<Array>} Array of track objects
   */
  const getAllTracks = useCallback(async (limit = 100, offset = 0) => {
    setIsLoading(true);
    setError(null);

    try {
      // Use pagination parameters to prevent memory issues
      const response = await apiClient.get('/tracks', { 
        limit: limit, 
        offset: offset 
      });
      
      setIsLoading(false);
      return response.tracks || [];
    } catch (err) {
      console.error('Failed to fetch all tracks:', err);
      setError(err);
      setIsLoading(false);
      throw err;
    }
  }, []);

  /**
   * Get ALL tracks using automatic pagination
   * This method handles pagination automatically by making multiple requests
   * Use this when you need the complete library
   * 
   * @param {number} batchSize - Tracks per batch (default 100)
   * @returns {Promise<Array>} Complete array of all tracks
   */
  const getAllTracksPaginated = useCallback(async (batchSize = 100) => {
    setIsLoading(true);
    setError(null);

    try {
      // Use the new paginated method from apiClient
      const tracks = await apiClient.getAllTracksPaginated(batchSize);
      setIsLoading(false);
      return tracks;
    } catch (err) {
      console.error('Failed to fetch paginated tracks:', err);
      setError(err);
      setIsLoading(false);
      throw err;
    }
  }, []);

  /**
   * Get track positions for 3D visualization
   * This endpoint is CPU-intensive, results should be cached
   */
  const getTrackPositions = useCallback(async (limit = 500) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await apiClient.get('/tracks/positions', { limit });
      setIsLoading(false);
      return response.positions || [];
    } catch (err) {
      console.error('Failed to fetch track positions:', err);
      setError(err);
      setIsLoading(false);
      throw err;
    }
  }, []);

  /**
   * Parse natural language query into structured parameters
   */
  const parseIntent = useCallback(async ({ query, context, session_id }) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await apiClient.post('/parse-intent', {
        query,
        context,
        session_id
      });
      setIsLoading(false);
      return response;
    } catch (err) {
      console.error('Failed to parse intent:', err);
      setError(err);
      setIsLoading(false);
      throw err;
    }
  }, []);

  /**
   * Get intelligent recommendations using structured query
   */
  const getIntelligentRecommendations = useCallback(async ({ 
    structured_query, 
    context_track_id 
  }) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await apiClient.post('/intelligent-recommend', {
        query: structured_query,
        context_track_id
      });
      setIsLoading(false);
      return response;
    } catch (err) {
      console.error('Failed to get recommendations:', err);
      setError(err);
      setIsLoading(false);
      throw err;
    }
  }, []);

  /**
   * Get similar tracks (neighbors) for a given track
   */
  const getSimilarTracks = useCallback(async (trackId, limit = 8) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await apiClient.get(`/tracks/${trackId}/neighbors`, { limit });
      setIsLoading(false);
      return response.neighbors || [];
    } catch (err) {
      console.error('Failed to fetch similar tracks:', err);
      setError(err);
      setIsLoading(false);
      throw err;
    }
  }, []);

  /**
   * Validate crate sequence and manage crate operations
   */
  const manageCrate = useCallback(async ({ session_id, tracks, sequence_order, metadata }) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await apiClient.post('/crate/operations', {
        session_id,
        tracks,
        sequence_order,
        metadata
      });
      setIsLoading(false);
      return response;
    } catch (err) {
      console.error('Failed to manage crate:', err);
      setError(err);
      setIsLoading(false);
      throw err;
    }
  }, []);

  /**
   * Get pathway visualization data
   */
  const getPathwayVisualization = useCallback(async (from_track, to_tracks) => {
    setIsLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams({
        from_track,
        to_tracks: to_tracks.join(',')
      });
      const response = await apiClient.get(`/visualization/pathway?${params}`);
      setIsLoading(false);
      return response;
    } catch (err) {
      console.error('Failed to get pathway visualization:', err);
      setError(err);
      setIsLoading(false);
      throw err;
    }
  }, []);

  /**
   * Clear API cache (useful for forced refresh)
   */
  const clearCache = useCallback(() => {
    apiClient.clearCache();
  }, []);

  return {
    // Methods
    getAllTracks,
    getAllTracksPaginated,
    getTrackPositions,
    parseIntent,
    getIntelligentRecommendations,
    getSimilarTracks,
    manageCrate,
    getPathwayVisualization,
    clearCache,
    
    // State
    isLoading,
    error
  };
};