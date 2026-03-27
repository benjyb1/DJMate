// playlist/PlaylistChatBar.jsx — Fixed bottom chat bar for LLM interaction
import React from 'react';
import { m } from 'framer-motion';

const quickPrompts = [
  'Find deep house tracks',
  'Dark minimal techno',
  'Tracks around 128 BPM',
  'High energy peak time',
  'Melodic and atmospheric',
];

export default function PlaylistChatBar({
  query,
  onQueryChange,
  status,       // 'idle' | 'thinking' | 'done'
  feedback,     // string — status message
  onSubmit,     // (query) => void
}) {
  const isThinking = status === 'thinking';

  return (
    <div style={{
      borderTop: '1px solid var(--glass-border)',
      padding: '10px 20px 12px',
      flexShrink: 0,
      background: 'rgba(8,8,20,0.5)',
    }}>
      {/* Quick prompts */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 8, flexWrap: 'wrap' }}>
        {quickPrompts.map(prompt => (
          <m.button
            key={prompt}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => !isThinking && onSubmit(prompt)}
            disabled={isThinking}
            style={{
              padding: '4px 10px',
              background: 'rgba(124,58,237,0.06)',
              border: '1px solid rgba(124,58,237,0.2)',
              borderRadius: 'var(--radius-pill)',
              color: isThinking ? '#334155' : '#94a3b8',
              fontSize: 9, fontWeight: 700,
              letterSpacing: '0.05em', cursor: isThinking ? 'wait' : 'pointer',
              fontFamily: 'var(--font-ui)',
            }}
          >
            {prompt}
          </m.button>
        ))}
      </div>

      {/* Input bar */}
      <div style={{ display: 'flex', gap: 8 }}>
        <input
          value={query}
          onChange={e => onQueryChange(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && query.trim() && !isThinking) onSubmit(query); }}
          placeholder="e.g. Find deep house tracks..."
          disabled={isThinking}
          style={{
            flex: 1, padding: '9px 14px',
            background: 'rgba(8,8,20,0.5)',
            border: '1px solid var(--glass-border)',
            borderRadius: 'var(--radius-sm)',
            color: '#e2e8f0', fontSize: 12,
            fontFamily: 'var(--font-ui)',
            outline: 'none',
          }}
        />
        <m.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.97 }}
          onClick={() => { if (query.trim() && !isThinking) onSubmit(query); }}
          disabled={isThinking || !query.trim()}
          style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            width: 38, height: 38,
            background: isThinking ? 'rgba(100,116,139,0.1)' : 'rgba(0,212,255,0.08)',
            border: '1px solid ' + (isThinking ? 'rgba(100,116,139,0.2)' : 'rgba(0,212,255,0.3)'),
            borderRadius: 'var(--radius-sm)',
            color: isThinking ? '#475569' : '#00d4ff',
            cursor: isThinking ? 'wait' : 'pointer',
          }}
        >
          {isThinking ? (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style={{ animation: 'spin 1s linear infinite' }}>
              <circle cx="7" cy="7" r="5.5" stroke="currentColor" strokeWidth="1.5" strokeDasharray="20 10" strokeLinecap="round"/>
            </svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
            </svg>
          )}
        </m.button>
      </div>

      {/* Status feedback */}
      {(isThinking || feedback) && (
        <m.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          style={{
            marginTop: 6, fontSize: 10, color: '#94a3b8',
            fontFamily: 'var(--font-mono)',
          }}
        >
          {isThinking ? 'Thinking...' : feedback}
        </m.div>
      )}
    </div>
  );
}
