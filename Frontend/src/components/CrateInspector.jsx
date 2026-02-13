// src/components/CrateInspector.jsx
import React, { useState, useEffect, useCallback } from 'react';
import { DragDropContext, Droppable, Draggable } from 'hello-pangea/dnd';
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

    const totalDuration = crate.length * 4; // Estimate 4 minutes per track
    const avgBPM = Math.round(
      crate.reduce((sum, track) => sum + (track.bpm || 120), 0) / crate.length
    );
    const energyLevels = crate.map(track => track.energy || 0.5);
    const avgEnergy = energyLevels.reduce((sum, energy) => sum + energy, 0) / energyLevels.length;

    return {
      totalDuration,
      avgBPM,
      avgEnergy,
      mixabilityScore: validationResults?.sequence_score || 0
    };
  };

  const stats = calculateCrateStats();

  const exportCrateAsPlaylist = () => {
    const playlist = {
      name: `AI DJ Crate - ${new Date().toLocaleDateString()}`,
      tracks: crate,
      created_at: new Date().toISOString(),
      validation_results: validationResults
    };

    const blob = new Blob([JSON.stringify(playlist, null, 2)], {
      type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ai_dj_crate_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const clearCrate = () => {
    if (window.confirm('Are you sure you want to clear the entire crate?')) {
      crate.forEach(() => onTrackRemove(crate[0]?.id));
    }
  };

  return (
    <div className="crate-inspector">
      {/* Header */}
      <div className="crate-header">
        <div className="header-main">
          <h2>Virtual Crate</h2>
          <div className="crate-controls">
            <button
              onClick={() => setShowStats(!showStats)}
              className="stats-toggle"
              title="Show/hide statistics"
            >
              <BarChart3 size={18} />
            </button>
            <button
              onClick={exportCrateAsPlaylist}
              className="export-button"
              disabled={crate.length === 0}
              title="Export playlist"
            >
              <Download size={18} />
            </button>
            <button
              onClick={clearCrate}
              className="clear-button"
              disabled={crate.length === 0}
              title="Clear crate"
            >
              <Trash2 size={18} />
            </button>
          </div>
        </div>

        <div className="crate-stats-summary">
          <span className="track-count">{crate.length} tracks</span>
          {stats && (
            <>
              <span className="duration">
                <Clock size={14} />
                {Math.floor(stats.totalDuration / 60)}h {stats.totalDuration % 60}m
              </span>
              {validationResults && (
                <span className={`compatibility-score ${validationResults.is_mixable ? 'good' : 'warning'}`}>
                  Mix: {Math.round(stats.mixabilityScore * 100)}%
                </span>
              )}
            </>
          )}
        </div>
      </div>

      {/* Detailed Stats Panel */}
      {showStats && stats && (
        <div className="crate-stats-detailed">
          <div className="stat-item">
            <label>Average BPM:</label>
            <span>{stats.avgBPM}</span>
          </div>
          <div className="stat-item">
            <label>Average Energy:</label>
            <span>{(stats.avgEnergy * 10).toFixed(1)}/10</span>
          </div>
          <div className="stat-item">
            <label>Mixability:</label>
            <span className={stats.mixabilityScore > 0.7 ? 'good' : 'warning'}>
              {Math.round(stats.mixabilityScore * 100)}%
            </span>
          </div>
        </div>
      )}

      {/* Track List */}
      <DragDropContext onDragEnd={handleDragEnd}>
        <Droppable droppableId="crate">
          {(provided, snapshot) => (
            <div
              {...provided.droppableProps}
              ref={provided.innerRef}
              className={`crate-list ${snapshot.isDraggingOver ? 'drag-over' : ''} ${isLoading ? 'loading' : ''}`}
            >
              {crate.length === 0 ? (
                <div className="empty-crate">
                  <p>Your crate is empty</p>
                  <small>Drag tracks from the 3D visualization or use the search bar to find tracks</small>
                </div>
              ) : (
                crate.map((track, index) => (
                  <Draggable key={track.id} draggableId={track.id} index={index}>
                    {(provided, snapshot) => (
                      <div
                        ref={provided.innerRef}
                        {...provided.draggableProps}
                        className={`crate-item ${snapshot.isDragging ? 'dragging' : ''}`}
                      >
                        {/* Drag handle */}
                        <div {...provided.dragHandleProps} className="drag-handle">
                          <div className="drag-dots">⋮⋮</div>
                        </div>

                        {/* Track info */}
                        <div className="track-info">
                          <img 
                            src={track.album_art || '/api/placeholder/40/40'} 
                            alt={track.title}
                            className="track-thumbnail"
                            onError={(e) => {
                              e.target.src = '/api/placeholder/40/40';
                            }}
                          />
                          <div className="track-details">
                            <div className="track-title" title={track.title}>
                              {track.title}
                            </div>
                            <div className="track-artist" title={track.artist}>
                              {track.artist}
                            </div>
                            <div className="track-metadata">
                              <span className="bpm" title="BPM">
                                {Math.round(track.bpm || 120)}
                              </span>
                              <span className="key" title="Key">
                                {track.key || '?'}
                              </span>
                              <span className="energy" title="Energy Level">
                                E{((track.energy || 0.5) * 10).toFixed(1)}
                              </span>
                            </div>
                          </div>
                        </div>

                        {/* Playback controls */}
                        <div className="track-controls">
                          <button
                            onClick={() => handlePlayPause(track)}
                            className="play-button"
                            title={isPlaying === track.id ? 'Pause' : 'Play preview'}
                          >
                            {isPlaying === track.id ? <Pause size={16} /> : <Play size={16} />}
                          </button>
                          <button
                            onClick={() => onTrackRemove(track.id)}
                            className="remove-button"
                            title="Remove from crate"
                          >
                            ×
                          </button>
                        </div>

                        {/* Transition indicator */}
                        <div className="transition-indicator-container">
                          {getTransitionIndicator(index)}
                        </div>
                      </div>
                    )}
                  </Draggable>
                ))
              )}
              {provided.placeholder}
            </div>
          )}
        </Droppable>
      </DragDropContext>

      {/* Compatibility Issues Panel */}
      {validationResults && validationResults.issues && validationResults.issues.length > 0 && (
        <div className="compatibility-issues">
          <h3>
            <AlertTriangle size={16} />
            Mixing Issues ({validationResults.issues.length})
          </h3>
          <div className="issues-list">
            {validationResults.issues.map((issue, index) => (
              <div key={index} className={`issue-item severity-${issue.severity}`}>
                <div className="issue-icon">
                  <AlertTriangle size={14} />
                </div>
                <div className="issue-content">
                  <div className="issue-description">
                    {formatCompatibilityIssue(issue)}
                  </div>
                  <div className="issue-position">
                    Between tracks {issue.position + 1} and {issue.position + 2}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Loading indicator */}
      {isLoading && (
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <span>Validating sequence...</span>
        </div>
      )}
    </div>
  );
};

export default CrateInspector;