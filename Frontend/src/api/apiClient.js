// src/api/apiClient.js
class APIError extends Error {
  constructor(message, status, response) {
    super(message);
    this.name = 'APIError';
    this.status = status;
    this.response = response;
  }
}

class APIClient {
  constructor(baseURL = 'http://localhost:8000') {
    this.baseURL = baseURL;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
    };
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: { ...this.defaultHeaders, ...options.headers },
      ...options,
    };

    // Add request logging for development
    if (process.env.NODE_ENV === 'development') {
      console.log(`🌐 API Request: ${config.method || 'GET'} ${url}`, {
        body: config.body ? JSON.parse(config.body) : undefined,
      });
    }

    try {
      const response = await fetch(url, config);
      const data = await response.json();

      if (!response.ok) {
        throw new APIError(
          data.detail || `HTTP ${response.status}: ${response.statusText}`,
          response.status,
          data
        );
      }

      // Log successful responses in development
      if (process.env.NODE_ENV === 'development') {
        console.log(`✅ API Response: ${config.method || 'GET'} ${url}`, data);
      }

      return data;
    } catch (error) {
      if (error instanceof APIError) {
        throw error;
      }
      
      // Handle network errors
      console.error(`❌ API Network Error: ${config.method || 'GET'} ${url}`, error);
      throw new APIError('Network error - please check your connection', 0, null);
    }
  }

  // GET request helper
  async get(endpoint, params = {}) {
    const queryString = new URLSearchParams(params).toString();
    const url = queryString ? `${endpoint}?${queryString}` : endpoint;
    return this.request(url);
  }

  // POST request helper
  async post(endpoint, data = {}) {
    return this.request(endpoint, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // PUT request helper
  async put(endpoint, data = {}) {
    return this.request(endpoint, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  // DELETE request helper
  async delete(endpoint) {
    return this.request(endpoint, {
      method: 'DELETE',
    });
  }
}

// Create singleton instance
export const apiClient = new APIClient();
export { APIError };

2. Recommendation API Hook

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

3. Crate Manager Hook

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

4. Enhanced CrateInspector Component

// src/components/CrateInspector.jsx
import React, { useState, useEffect, useCallback } from 'react';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';
import { 
  Play, 
  Pause, 
  SkipForward, 
  AlertTriangle, 
  CheckCircle, 
  Download,
  Trash2,
  BarChart3,
  Clock
} from 'lucide-react';

const CrateInspector = ({ 
  crate, 
  onTrackRemove, 
  onTrackReorder, 
  onSequenceValidation,
  compatibilityData,
  isLoading = false
}) => {
  const [isPlaying, setIsPlaying] = useState(null);
  const [validationResults, setValidationResults] = useState(null);
  const [showStats, setShowStats] = useState(false);
  const [audioElement, setAudioElement] = useState(null);

  // Initialize audio element
  useEffect(() => {
    const audio = new Audio();
    audio.addEventListener('ended', () => setIsPlaying(null));
    setAudioElement(audio);
    
    return () => {
      audio.pause();
      audio.removeEventListener('ended', () => setIsPlaying(null));
    };
  }, []);

  // Validate sequence whenever crate changes
  useEffect(() => {
    if (crate.length > 1 && onSequenceValidation) {
      validateCurrentSequence();
    } else {
      setValidationResults(null);
    }
  }, [crate, onSequenceValidation]);

  const validateCurrentSequence = async () => {
    try {
      const results = await onSequenceValidation(crate.map(track => track.id));
      setValidationResults(results);
    } catch (error) {
      console.error('Sequence validation failed:', error);
    }
  };

  const handleDragEnd = useCallback((result) => {
    if (!result.destination || isLoading) return;

    const newOrder = Array.from(crate);
    const [reorderedItem] = newOrder.splice(result.source.index, 1);
    newOrder.splice(result.destination.index, 0, reorderedItem);

    onTrackReorder(newOrder);
  }, [crate, onTrackReorder, isLoading]);

  const handlePlayPause = useCallback((track) => {
    if (!audioElement) return;

    if (isPlaying === track.id) {
      audioElement.pause();
      setIsPlaying(null);
    } else {
      // For demo purposes, we'll use a placeholder audio URL
      // In production, this would be the actual track file
      audioElement.src = track.preview_url || track.filepath || '';
      audioElement.play().catch(err => {
        console.warn('Audio playback failed:', err);
        // Fallback: just indicate it's "playing" without actual audio
        setIsPlaying(track.id);
        setTimeout(() => setIsPlaying(null), 3000); // Auto-stop after 3 seconds
      });
      setIsPlaying(track.id);
    }
  }, [audioElement, isPlaying]);

  const getTransitionIndicator = (index) => {
    if (!validationResults || index >= crate.length - 1) return null;
    
    const transitionScore = validationResults.transition_scores?.[index];
    const hasIssues = validationResults.issues?.some(issue => issue.position === index);
    
    if (hasIssues) {
      const severity = validationResults.issues.find(issue => issue.position === index)?.severity;
      return (
        <AlertTriangle 
          className={`transition-indicator ${severity}`} 
          size={16} 
          title="Mixing issue detected"
        />
      );
    } else if (transitionScore > 0.7) {
      return (
        <CheckCircle 
          className="transition-indicator good" 
          size={16} 
          title="Good transition"
        />
      );
    }
    return null;
  };

  const formatCompatibilityIssue = (issue) => {
    switch (issue.type) {
      case 'bpm_incompatibility':
        return `BPM jump: ${issue.details}`;
      case 'harmonic_clash':
        return `Key clash: ${issue.details}`;
      default:
        return issue.details;
    }
  };

  const calculateCrateStats = () => {
    if (crate.length === 0) return null;

    const totalDuration = crate.length * 4; // Estima
