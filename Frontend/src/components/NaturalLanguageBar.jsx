// Frontend/src/components/NaturalLanguageBar.jsx
import React, { useState, useRef, useEffect } from 'react';
import { Search, Loader, Brain, Music, Sparkles } from 'lucide-react';

const NaturalLanguageBar = ({ onQuery, isLoading, lastInterpretation }) => {
  const [queryText, setQueryText] = useState('');
  const [showInterpretation, setShowInterpretation] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const inputRef = useRef(null);

  // DJ-specific query examples
  const exampleQueries = [
    "Give me something dark but bouncy around 126 BPM",
    "I need a melodic bridge to get from house to techno", 
    "Find tracks similar to this but with more energy",
    "Build me a late-night minimal set",
    "Something hypnotic and deep for the comedown",
    "Uplifting progressive tracks around 128 BPM",
    "Dark underground techno with good basslines",
    "Groovy tech house for peak time"
  ];

  const [currentExamples, setCurrentExamples] = useState(
    exampleQueries.slice(0, 3)
  );

  // Rotate examples periodically
  useEffect(() => {
    const interval = setInterval(() => {
      const startIndex = Math.floor(Math.random() * (exampleQueries.length - 3));
      setCurrentExamples(exampleQueries.slice(startIndex, startIndex + 3));
    }, 10000); // Change every 10 seconds

    return () => clearInterval(interval);
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (queryText.trim() && !isLoading) {
      onQuery(queryText.trim());
      setShowSuggestions(false);
    }
  };

  const handleExampleClick = (example) => {
    setQueryText(example);
    inputRef.current?.focus();
    setShowSuggestions(false);
  };

  const handleInputFocus = () => {
    setShowSuggestions(true);
  };

  const handleInputBlur = () => {
    // Delay hiding suggestions to allow clicks
    setTimeout(() => setShowSuggestions(false), 200);
  };

  const interpretationSummary = lastInterpretation ? {
    tags: lastInterpretation.structured_query?.tags || [],
    vibes: lastInterpretation.structured_query?.vibe_descriptors || [],
    bmp_range: lastInterpretation.structured_query?.bpm_range,
    energy_range: lastInterpretation.structured_query?.energy_range,
    direction: lastInterpretation.structured_query?.direction,
    confidence: lastInterpretation.confidence || 0
  } : null;

  return (
    <div className="natural-language-bar">
      {/* Main query interface */}
      <div className="query-section">
        <form onSubmit={handleSubmit} className="query-form">
          <div className="input-container">
            <Search className="search-icon" size={20} />
            <input
              ref={inputRef}
              type="text"
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
              onFocus={handleInputFocus}
              onBlur={handleInputBlur}
              placeholder="Describe the vibe you're looking for... (e.g., 'dark minimal around 124 BPM')"
              className="query-input"
              disabled={isLoading}
            />
            {isLoading && <Loader className="loading-icon spinning" size={20} />}
            <button 
              type="submit" 
              disabled={!queryText.trim() || isLoading}
              className="query-submit"
            >
              {isLoading ? 'Thinking...' : 'Search'}
            </button>
          </div>
        </form>

        {/* Query suggestions dropdown */}
        {showSuggestions && !lastInterpretation && (
          <div className="suggestions-dropdown">
            <div className="suggestions-header">
              <Sparkles size={16} />
              <span>Try these DJ queries:</span>
            </div>
            <div className="suggestions-list">
              {currentExamples.map((example, index) => (
                <button
                  key={index}
                  onClick={() => handleExampleClick(example)}
                  className="suggestion-item"
                >
                  <Music size={14} />
                  <span>"{example}"</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* AI interpretation display */}
      {interpretationSummary && (
        <div className="interpretation-panel">
          <button
            onClick={() => setShowInterpretation(!showInterpretation)}
            className="interpretation-toggle"
          >
            <Brain size={16} />
            <span>AI Interpretation</span>
            <span className={`confidence-badge ${interpretationSummary.confidence > 0.7 ? 'high' : 'medium'}`}>
              {Math.round(interpretationSummary.confidence * 100)}%
            </span>
            <span className={`expand-icon ${showInterpretation ? 'expanded' : ''}`}>
              ▼
            </span>
          </button>

          {showInterpretation && (
            <div className="interpretation-details">
              {interpretationSummary.tags.length > 0 && (
                <div className="interpretation-section">
                  <label>Style Tags:</label>
                  <div className="tag-list">
                    {interpretationSummary.tags.map(tag => (
                      <span key={tag} className="tag style-tag">{tag}</span>
                    ))}
                  </div>
                </div>
              )}

              {interpretationSummary.vibes.length > 0 && (
                <div className="interpretation-section">
                  <label>Vibe:</label>
                  <div className="tag-list">
                    {interpretationSummary.vibes.map(vibe => (
                      <span key={vibe} className="tag vibe-tag">{vibe}</span>
                    ))}
                  </div>
                </div>
              )}

              <div className="interpretation-ranges">
                {interpretationSummary.bpm_range && (
                  <div className="range-item">
                    <label>BPM:</label>
                    <span className="range-value">
                      {interpretationSummary.bpm_range[0]} - {interpretationSummary.bpm_range[1]}
                    </span>
                  </div>
                )}

                {interpretationSummary.energy_range && (
                  <div className="range-item">
                    <label>Energy:</label>
                    <span className="range-value">
                      {(interpretationSummary.energy_range[0] * 10).toFixed(1)} - {(interpretationSummary.energy_range[1] * 10).toFixed(1)}
                    </span>
                  </div>
                )}

                {interpretationSummary.direction && (
                  <div className="range-item">
                    <label>Direction:</label>
                    <span className="direction-value">
                      {interpretationSummary.direction.replace('_', ' ')}
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default NaturalLanguageBar;