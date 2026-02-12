// Frontend/src/components/CompatibilityIndicator.jsx
import React from 'react';
import { AlertTriangle, CheckCircle, Zap, Music } from 'lucide-react';

const CompatibilityIndicator = ({ 
  activeTrack, 
  crateSequence = [], 
  validationResults 
}) => {
  if (!activeTrack && crateSequence.length === 0) {
    return (
      <div className="compatibility-indicator empty">
        <Music size={16} />
        <span>Select a track or build a crate to see compatibility info</span>
      </div>
    );
  }

  const getCompatibilityScore = () => {
    if (validationResults?.overall_score !== undefined) {
      return Math.round(validationResults.overall_score * 100);
    }
    return null;
  };

  const getCompatibilityColor = (score) => {
    if (score >= 80) return 'excellent';
    if (score >= 60) return 'good';
    if (score >= 40) return 'fair';
    return 'poor';
  };

  const score = getCompatibilityScore();
  const colorClass = score !== null ? getCompatibilityColor(score) : 'neutral';

  return (
    <div className={`compatibility-indicator ${colorClass}`}>
      <div className="indicator-main">
        {/* Active track info */}
        {activeTrack && (
          <div className="active-track-info">
            <div className="track-details">
              <span className="track-title">{activeTrack.title}</span>
              <span className="track-meta">
                {Math.round(activeTrack.bpm || 120)} BPM • {activeTrack.key || '?'} • 
                E{((activeTrack.energy || 0.5) * 10).toFixed(1)}
              </span>
            </div>
          </div>
        )}

        {/* Crate compatibility */}
        {crateSequence.length > 0 && (
          <div className="crate-compatibility">
            <div className="compatibility-score">
              {score !== null ? (
                <>
                  {score >= 70 ? <CheckCircle size={16} /> : <AlertTriangle size={16} />}
                  <span>Mix Score: {score}%</span>
                </>
              ) : (
                <>
                  <Zap size={16} />
                  <span>{crateSequence.length} tracks in crate</span>
                </>
              )}
            </div>
            
            {validationResults?.issues?.length > 0 && (
              <div className="compatibility-issues-summary">
                <AlertTriangle size={14} />
                <span>{validationResults.issues.length} mixing issues</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Detailed breakdown */}
      {validationResults && (
        <div className="compatibility-details">
          <div className="detail-item">
            <span className="label">Transitions:</span>
            <span className="value">
              {validationResults.transition_scores?.length || 0} analyzed
            </span>
          </div>
          
          {validationResults.is_mixable !== undefined && (
            <div className="detail-item">
              <span className="label">Mixable:</span>
              <span className={`value ${validationResults.is_mixable ? 'positive' : 'negative'}`}>
                {validationResults.is_mixable ? 'Yes' : 'Issues detected'}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default CompatibilityIndicator;