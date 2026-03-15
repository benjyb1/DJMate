import React, { useState, useRef } from 'react';
import { m } from 'framer-motion';

const IconSend = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13" /><polygon points="22,2 15,22 11,13 2,9" />
  </svg>
);

const QUICK_PROMPTS = [
  { label: 'HIGHER ENERGY', prompt: 'Give me a higher energy crate' },
  { label: 'DIFFERENT GENRE', prompt: 'Give me a crate with a different genre' },
  { label: 'SLOW IT DOWN', prompt: 'Slow it down with a lower energy crate' },
  { label: 'BRIDGE TRACKS', prompt: 'Suggest bridge tracks between these crates' },
];

export default function CrateChat({ onSubmit, isLoading = false, selectedCrateLabel = '' }) {
  const [input, setInput] = useState('');
  const inputRef = useRef(null);

  const handleSubmit = () => {
    if (!input.trim() || isLoading) return;
    onSubmit(input.trim());
    setInput('');
  };

  return (
    <div style={{ padding: '12px 16px', borderTop: '1px solid rgba(124,58,237,0.08)' }}>
      {/* Context indicator */}
      {selectedCrateLabel && (
        <div style={{
          fontSize: 8, letterSpacing: '0.15em', color: 'rgba(124,58,237,0.4)',
          fontFamily: "'JetBrains Mono', monospace", marginBottom: 8,
        }}>
          BRANCHING FROM: {selectedCrateLabel.toUpperCase()}
        </div>
      )}

      {/* Quick prompts */}
      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 10 }}>
        {QUICK_PROMPTS.map(qp => (
          <m.button
            key={qp.label}
            onClick={() => !isLoading && onSubmit(qp.prompt)}
            whileHover={{ scale: 1.04, borderColor: 'rgba(124,58,237,0.4)' }}
            whileTap={{ scale: 0.96 }}
            style={{
              padding: '3px 10px',
              background: 'rgba(124,58,237,0.06)',
              border: '1px solid rgba(124,58,237,0.15)',
              borderRadius: 'var(--radius-pill)',
              color: '#94a3b8',
              cursor: isLoading ? 'default' : 'pointer',
              fontSize: 8,
              fontFamily: "'JetBrains Mono', monospace",
              letterSpacing: '0.08em',
              opacity: isLoading ? 0.4 : 1,
            }}
          >
            {qp.label}
          </m.button>
        ))}
      </div>

      {/* Input */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <input
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSubmit()}
          placeholder="Describe the crate you want..."
          disabled={isLoading}
          style={{
            flex: 1,
            background: 'rgba(8,8,20,0.6)',
            border: '1px solid rgba(124,58,237,0.18)',
            borderRadius: 'var(--radius-sm)',
            color: '#e2e8f0',
            fontSize: 11,
            padding: '9px 12px',
            fontFamily: "'Inter', system-ui, sans-serif",
            outline: 'none',
            caretColor: '#7c3aed',
            opacity: isLoading ? 0.5 : 1,
          }}
        />
        <m.button
          onClick={handleSubmit}
          disabled={isLoading || !input.trim()}
          whileHover={isLoading ? {} : { scale: 1.08 }}
          whileTap={isLoading ? {} : { scale: 0.92 }}
          style={{
            width: 34, height: 34,
            borderRadius: 'var(--radius-sm)',
            background: isLoading ? 'rgba(124,58,237,0.1)' : 'linear-gradient(135deg, rgba(124,58,237,0.4), rgba(0,212,255,0.25))',
            border: `1px solid ${isLoading ? 'rgba(124,58,237,0.1)' : 'rgba(124,58,237,0.4)'}`,
            color: isLoading ? '#475569' : '#e2e8f0',
            cursor: isLoading ? 'default' : 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0,
          }}
        >
          {isLoading ? (
            <div style={{
              width: 12, height: 12,
              border: '1.5px solid rgba(124,58,237,0.15)',
              borderTop: '1.5px solid #7c3aed',
              borderRadius: '50%',
              animation: 'spin 0.8s linear infinite',
            }} />
          ) : (
            <IconSend />
          )}
        </m.button>
      </div>
    </div>
  );
}
