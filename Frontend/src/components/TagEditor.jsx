// src/components/TagEditor.jsx
import React, { useState, useRef, useEffect } from 'react';
import { m, AnimatePresence } from 'framer-motion';
import { apiClient } from '../api/apiClient';
import GlassPanel from './ui/GlassPanel';

const IconClose = () => (
  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true">
    <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
  </svg>
);

// ── Removable pill ────────────────────────────────────────────────────────
function TagPill({ label, variant = 'genre', onRemove, index = 0 }) {
  const isVibe = variant === 'vibe';
  return (
    <m.span
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.8 }}
      transition={{ type: 'spring', damping: 22, stiffness: 400, delay: index * 0.02 }}
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 4,
        background: isVibe ? 'rgba(0,212,255,0.07)' : 'rgba(124,58,237,0.1)',
        color: isVibe ? 'rgba(0,212,255,0.8)' : 'rgba(168,85,247,0.8)',
        border: `1px solid ${isVibe ? 'rgba(0,212,255,0.22)' : 'rgba(124,58,237,0.22)'}`,
        borderRadius: 'var(--radius-pill)',
        padding: '3px 8px 3px 10px',
        fontSize: 10,
        fontFamily: "'JetBrains Mono', monospace",
        letterSpacing: '0.04em',
      }}
    >
      {label}
      <m.button
        onClick={onRemove}
        whileHover={{ scale: 1.2, color: '#ef4444' }}
        whileTap={{ scale: 0.85 }}
        style={{
          background: 'none', border: 'none', cursor: 'pointer',
          color: 'inherit', opacity: 0.6, display: 'flex', alignItems: 'center',
          padding: 0, marginLeft: 2,
        }}
        aria-label={`Remove ${label}`}
      >
        <IconClose />
      </m.button>
    </m.span>
  );
}

// ── Autocomplete input ────────────────────────────────────────────────────
function TagInput({ placeholder, value, onChange, suggestions, onAdd }) {
  const [showDropdown, setShowDropdown] = useState(false);
  const inputRef = useRef(null);

  const filtered = value.trim()
    ? suggestions.filter(s => s.toLowerCase().includes(value.toLowerCase())).slice(0, 8)
    : [];

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && value.trim()) {
      e.preventDefault();
      onAdd(value.trim().toLowerCase());
      onChange('');
    }
    if (e.key === 'Escape') {
      setShowDropdown(false);
      inputRef.current?.blur();
    }
  };

  const handleSelect = (tag) => {
    onAdd(tag);
    onChange('');
    setShowDropdown(false);
    inputRef.current?.focus();
  };

  return (
    <div style={{ position: 'relative' }}>
      <input
        ref={inputRef}
        value={value}
        onChange={e => { onChange(e.target.value); setShowDropdown(true); }}
        onFocus={() => setShowDropdown(true)}
        onBlur={() => setTimeout(() => setShowDropdown(false), 150)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        style={{
          width: '100%', boxSizing: 'border-box',
          background: 'rgba(8,8,20,0.6)',
          border: '1px solid rgba(124,58,237,0.18)',
          borderRadius: 'var(--radius-sm)',
          color: '#e2e8f0',
          fontSize: 11,
          padding: '7px 10px',
          fontFamily: "'JetBrains Mono', monospace",
          outline: 'none',
          caretColor: '#7c3aed',
          transition: '200ms ease',
        }}
      />
      <AnimatePresence>
        {showDropdown && filtered.length > 0 && (
          <m.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.12 }}
            style={{
              position: 'absolute', top: '100%', left: 0, right: 0,
              zIndex: 10,
              background: 'rgba(8,8,20,0.95)',
              border: '1px solid rgba(124,58,237,0.25)',
              borderRadius: 'var(--radius-sm)',
              marginTop: 4,
              maxHeight: 140,
              overflowY: 'auto',
              scrollbarWidth: 'thin',
              scrollbarColor: 'rgba(124,58,237,0.3) transparent',
            }}
          >
            {filtered.map(tag => (
              <div
                key={tag}
                onMouseDown={(e) => { e.preventDefault(); handleSelect(tag); }}
                style={{
                  padding: '6px 10px',
                  fontSize: 10,
                  color: '#94a3b8',
                  cursor: 'pointer',
                  fontFamily: "'JetBrains Mono', monospace",
                  letterSpacing: '0.04em',
                  borderBottom: '1px solid rgba(124,58,237,0.06)',
                  transition: '100ms ease',
                }}
                onMouseEnter={e => { e.target.style.background = 'rgba(124,58,237,0.12)'; e.target.style.color = '#e2e8f0'; }}
                onMouseLeave={e => { e.target.style.background = 'transparent'; e.target.style.color = '#94a3b8'; }}
              >
                {tag}
              </div>
            ))}
          </m.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ── Main TagEditor component ──────────────────────────────────────────────
export default function TagEditor({ track, onClose, onSave, availableTags = [], availableVibes = [] }) {
  const [tags, setTags]         = useState(track.semantic_tags || []);
  const [vibes, setVibes]       = useState(track.vibe_descriptors || []);
  const [energy, setEnergy]     = useState(track.energy ?? 5);
  const [tagInput, setTagInput] = useState('');
  const [vibeInput, setVibeInput] = useState('');
  const [saving, setSaving]     = useState(false);

  const addTag = (tag) => {
    if (tag && !tags.includes(tag)) setTags(prev => [...prev, tag]);
  };
  const removeTag = (tag) => setTags(prev => prev.filter(t => t !== tag));

  const addVibe = (vibe) => {
    if (vibe && !vibes.includes(vibe)) setVibes(prev => [...prev, vibe]);
  };
  const removeVibe = (vibe) => setVibes(prev => prev.filter(v => v !== vibe));

  const handleSave = async () => {
    setSaving(true);
    try {
      await apiClient.put(`/tags/${track.trackid}`, {
        semantic_tags: tags,
        vibe: vibes,
        energy: Math.round(energy),
      });
      onSave?.({ semantic_tags: tags, vibe_descriptors: vibes, energy });
      onClose();
    } catch (err) {
      console.error('Failed to save tags:', err);
    } finally {
      setSaving(false);
    }
  };

  // Close on escape
  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  const filteredAvailableTags = availableTags.filter(t => !tags.includes(t));
  const filteredAvailableVibes = availableVibes.filter(v => !vibes.includes(v));

  return (
    <m.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.15 }}
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 2000,
        background: 'rgba(0,0,0,0.6)',
        backdropFilter: 'blur(8px)',
        WebkitBackdropFilter: 'blur(8px)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
    >
      <GlassPanel
        depth={3}
        initial={{ opacity: 0, scale: 0.95 }}
        animateProps={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        transition={{ type: 'spring', damping: 26, stiffness: 350 }}
        onClick={e => e.stopPropagation()}
        style={{
          width: 380,
          maxHeight: '80vh',
          overflowY: 'auto',
          borderRadius: 'var(--radius-lg)',
          fontFamily: "'Inter', system-ui, sans-serif",
          scrollbarWidth: 'thin',
          scrollbarColor: 'rgba(124,58,237,0.3) transparent',
        }}
      >
        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between',
          padding: '16px 18px 12px',
          borderBottom: '1px solid rgba(124,58,237,0.1)',
        }}>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{
              fontSize: 8, letterSpacing: '0.25em', color: 'rgba(124,58,237,0.5)',
              fontFamily: "'JetBrains Mono', monospace", marginBottom: 6,
            }}>
              EDIT TAGS
            </div>
            <div style={{
              fontSize: 14, fontWeight: 700, color: '#e2e8f0',
              whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
            }}>
              {track.title || track.name || 'Unknown'}
            </div>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>
              {track.artist || 'Unknown'}
            </div>
          </div>
          <m.button
            onClick={onClose}
            whileHover={{ color: '#ef4444', scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            style={{
              background: 'none', border: 'none', cursor: 'pointer',
              color: '#475569', display: 'flex', padding: 4, marginTop: 2,
            }}
            aria-label="Close editor"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </m.button>
        </div>

        {/* Genres section */}
        <div style={{ padding: '14px 18px', borderBottom: '1px solid rgba(124,58,237,0.06)' }}>
          <div style={{
            fontSize: 8, letterSpacing: '0.2em', color: 'rgba(168,85,247,0.6)',
            fontFamily: "'JetBrains Mono', monospace", marginBottom: 10,
          }}>
            GENRES
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginBottom: 10 }}>
            <AnimatePresence>
              {tags.map((tag, i) => (
                <TagPill key={tag} label={tag} variant="genre" onRemove={() => removeTag(tag)} index={i} />
              ))}
            </AnimatePresence>
            {tags.length === 0 && (
              <span style={{ fontSize: 10, color: '#2d3748', fontFamily: "'JetBrains Mono', monospace", fontStyle: 'italic' }}>
                No genres
              </span>
            )}
          </div>
          <TagInput
            placeholder="Type to add genre..."
            value={tagInput}
            onChange={setTagInput}
            suggestions={filteredAvailableTags}
            onAdd={addTag}
          />
        </div>

        {/* Vibes section */}
        <div style={{ padding: '14px 18px', borderBottom: '1px solid rgba(124,58,237,0.06)' }}>
          <div style={{
            fontSize: 8, letterSpacing: '0.2em', color: 'rgba(0,212,255,0.6)',
            fontFamily: "'JetBrains Mono', monospace", marginBottom: 10,
          }}>
            VIBES
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginBottom: 10 }}>
            <AnimatePresence>
              {vibes.map((vibe, i) => (
                <TagPill key={vibe} label={vibe} variant="vibe" onRemove={() => removeVibe(vibe)} index={i} />
              ))}
            </AnimatePresence>
            {vibes.length === 0 && (
              <span style={{ fontSize: 10, color: '#2d3748', fontFamily: "'JetBrains Mono', monospace", fontStyle: 'italic' }}>
                No vibes
              </span>
            )}
          </div>
          <TagInput
            placeholder="Type to add vibe..."
            value={vibeInput}
            onChange={setVibeInput}
            suggestions={filteredAvailableVibes}
            onAdd={addVibe}
          />
        </div>

        {/* Energy slider */}
        <div style={{ padding: '14px 18px', borderBottom: '1px solid rgba(124,58,237,0.06)' }}>
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            marginBottom: 10,
          }}>
            <span style={{
              fontSize: 8, letterSpacing: '0.2em', color: 'rgba(124,58,237,0.5)',
              fontFamily: "'JetBrains Mono', monospace",
            }}>
              ENERGY
            </span>
            <span style={{
              fontSize: 12, color: '#00d4ff', fontWeight: 700,
              fontFamily: "'JetBrains Mono', monospace",
            }}>
              {Math.round(energy)}
            </span>
          </div>
          <input
            type="range"
            min="1"
            max="10"
            step="1"
            value={energy}
            onChange={e => setEnergy(parseInt(e.target.value))}
            style={{
              width: '100%',
              height: 4,
              appearance: 'none',
              WebkitAppearance: 'none',
              background: `linear-gradient(to right, #7c3aed ${energy * 10}%, rgba(124,58,237,0.15) ${energy * 10}%)`,
              borderRadius: 'var(--radius-pill)',
              outline: 'none',
              cursor: 'pointer',
            }}
          />
        </div>

        {/* Actions */}
        <div style={{
          display: 'flex', gap: 8, justifyContent: 'flex-end',
          padding: '14px 18px',
        }}>
          <m.button
            onClick={onClose}
            whileHover={{ scale: 1.02, borderColor: 'rgba(124,58,237,0.4)' }}
            whileTap={{ scale: 0.97 }}
            style={{
              padding: '8px 20px',
              background: 'transparent',
              border: '1px solid rgba(124,58,237,0.2)',
              borderRadius: 'var(--radius-sm)',
              color: '#94a3b8',
              cursor: 'pointer',
              fontSize: 11,
              fontWeight: 600,
              letterSpacing: '0.08em',
              fontFamily: "'Inter', system-ui, sans-serif",
            }}
          >
            CANCEL
          </m.button>
          <m.button
            onClick={handleSave}
            disabled={saving}
            whileHover={saving ? {} : { scale: 1.02 }}
            whileTap={saving ? {} : { scale: 0.97 }}
            style={{
              padding: '8px 24px',
              background: saving
                ? 'rgba(8,8,20,0.5)'
                : 'linear-gradient(135deg, rgba(124,58,237,0.5), rgba(0,212,255,0.3))',
              border: `1px solid ${saving ? 'rgba(124,58,237,0.1)' : 'rgba(124,58,237,0.45)'}`,
              borderRadius: 'var(--radius-sm)',
              color: saving ? '#475569' : '#e2e8f0',
              cursor: saving ? 'default' : 'pointer',
              fontSize: 11,
              fontWeight: 600,
              letterSpacing: '0.08em',
              fontFamily: "'Inter', system-ui, sans-serif",
              display: 'flex', alignItems: 'center', gap: 6,
            }}
          >
            {saving && (
              <div style={{
                width: 12, height: 12,
                border: '1.5px solid rgba(124,58,237,0.2)',
                borderTop: '1.5px solid #7c3aed',
                borderRadius: '50%',
                animation: 'spin 0.8s linear infinite',
              }} />
            )}
            SAVE
          </m.button>
        </div>
      </GlassPanel>
    </m.div>
  );
}
