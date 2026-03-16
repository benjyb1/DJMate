import React from 'react';
import { m, AnimatePresence } from 'framer-motion';
import GlassPanel from './ui/GlassPanel';

const IconClose = () => (
  <svg
    width="8"
    height="8"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="3"
    strokeLinecap="round"
  >
    <line x1="18" y1="6" x2="6" y2="18" />
    <line x1="6" y1="6" x2="18" y2="18" />
  </svg>
);

export default function CrateDetail({
  crate,
  onRemoveTrack,
  onRequestBridge,
}) {
  if (!crate) {
    return (
      <div
        style={{
          padding: 20,
          color: '#2d3748',
          fontSize: 10,
          fontFamily: "'JetBrains Mono', monospace",
          letterSpacing: '0.1em',
        }}
      >
        SELECT A CRATE TO VIEW DETAILS
      </div>
    );
  }

  const meta = crate.metadata || {};
  const tracks = crate.tracks || [];
  const bridge = meta.bridge_warning;

  return (
    <div style={{ padding: 16, overflow: 'auto', flex: 1 }}>
      {/* Header */}
      <div style={{ marginBottom: 14 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#e2e8f0', marginBottom: 4 }}>
          {crate.label || 'Crate'}
        </div>
        <div
          style={{
            display: 'flex',
            gap: 8,
            fontSize: 9,
            fontFamily: "'JetBrains Mono', monospace",
            color: '#94a3b8',
          }}
        >
          <span>{tracks.length} tracks</span>
          {meta.avg_bpm && <span style={{ color: '#00d4ff' }}>{meta.avg_bpm} BPM</span>}
          {meta.dominant_key && <span style={{ color: '#a855f7' }}>{meta.dominant_key}</span>}
        </div>
      </div>

      {/* Energy bar */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
          <span
            style={{
              fontSize: 8,
              letterSpacing: '0.2em',
              color: 'rgba(124,58,237,0.5)',
              fontFamily: "'JetBrains Mono', monospace",
            }}
          >
            ENERGY
          </span>
          <span
            style={{
              fontSize: 10,
              color: '#00d4ff',
              fontFamily: "'JetBrains Mono', monospace",
              fontWeight: 700,
            }}
          >
            {meta.avg_energy != null ? meta.avg_energy.toFixed(2) : '\u2014'}
          </span>
        </div>
        <div
          style={{
            height: 4,
            background: 'rgba(124,58,237,0.12)',
            borderRadius: 'var(--radius-pill)',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              height: '100%',
              width: `${(meta.avg_energy || 0) * 100}%`,
              background: 'linear-gradient(90deg, #7c3aed, #00d4ff)',
              borderRadius: 'var(--radius-pill)',
            }}
          />
        </div>
      </div>

      {/* Tags */}
      {meta.dominant_tags?.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 14 }}>
          {meta.dominant_tags.map((tag) => (
            <span
              key={tag}
              style={{
                fontSize: 9,
                color: 'rgba(168,85,247,0.8)',
                background: 'rgba(124,58,237,0.1)',
                border: '1px solid rgba(124,58,237,0.22)',
                borderRadius: 'var(--radius-pill)',
                padding: '2px 8px',
                fontFamily: "'JetBrains Mono', monospace",
                letterSpacing: '0.04em',
              }}
            >
              {tag}
            </span>
          ))}
        </div>
      )}

      {/* Track list */}
      <div style={{ marginBottom: 12 }}>
        <div
          style={{
            fontSize: 8,
            letterSpacing: '0.2em',
            color: 'rgba(124,58,237,0.4)',
            fontFamily: "'JetBrains Mono', monospace",
            marginBottom: 8,
          }}
        >
          TRACKS
        </div>
        <AnimatePresence>
          {tracks.map((t, i) => (
            <m.div
              key={t.trackid || i}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 8 }}
              transition={{ type: 'spring', damping: 24, stiffness: 300, delay: i * 0.03 }}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                padding: '8px 10px',
                background: 'var(--bg-card)',
                border: '1px solid var(--glass-border)',
                borderRadius: 'var(--radius-sm)',
                marginBottom: 3,
              }}
            >
              <span
                style={{
                  fontSize: 10,
                  color: 'rgba(124,58,237,0.2)',
                  fontWeight: 900,
                  fontFamily: "'JetBrains Mono', monospace",
                  minWidth: 16,
                }}
              >
                {String(i + 1).padStart(2, '0')}
              </span>

              <div style={{ flex: 1, minWidth: 0 }}>
                <div
                  style={{
                    fontSize: 11,
                    fontWeight: 600,
                    color: '#e2e8f0',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}
                >
                  {t.title || 'Unknown'}
                </div>
                <div style={{ fontSize: 9, color: '#94a3b8' }}>{t.artist || ''}</div>
              </div>

              <div
                style={{
                  display: 'flex',
                  gap: 6,
                  fontSize: 8,
                  fontFamily: "'JetBrains Mono', monospace",
                  color: '#475569',
                  flexShrink: 0,
                }}
              >
                {t.bpm && <span>{Math.round(t.bpm)}</span>}
                {t.key && <span style={{ color: '#a855f7' }}>{t.key}</span>}
              </div>

              {onRemoveTrack && (
                <m.button
                  onClick={() => onRemoveTrack(t.trackid)}
                  whileHover={{ scale: 1.2, color: '#ef4444' }}
                  whileTap={{ scale: 0.85 }}
                  style={{
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    color: '#2d3748',
                    display: 'flex',
                    padding: 2,
                  }}
                >
                  <IconClose />
                </m.button>
              )}
            </m.div>
          ))}
        </AnimatePresence>

        {tracks.length === 0 && (
          <div
            style={{
              padding: 12,
              fontSize: 9,
              color: '#2d3748',
              fontFamily: "'JetBrains Mono', monospace",
              fontStyle: 'italic',
            }}
          >
            No tracks in this crate
          </div>
        )}
      </div>

      {/* Bridge warning */}
      {bridge?.needs_bridge && (
        <GlassPanel
          depth={1}
          animate={false}
          style={{
            padding: '10px 14px',
            borderColor: 'rgba(239,68,68,0.3)',
            marginBottom: 12,
          }}
        >
          <div
            style={{
              fontSize: 9,
              color: 'rgba(239,68,68,0.8)',
              fontFamily: "'JetBrains Mono', monospace",
              letterSpacing: '0.06em',
              marginBottom: 6,
            }}
          >
            BRIDGE NEEDED — GAP SCORE {bridge.gap_score?.toFixed(2)}
          </div>
          <div style={{ fontSize: 9, color: '#94a3b8', marginBottom: 8, lineHeight: 1.5 }}>
            BPM gap: {bridge.bpm_gap} · Energy gap: {bridge.energy_gap} · Key:{' '}
            {bridge.key_compatible ? 'compatible' : 'incompatible'}
          </div>
          {onRequestBridge && (
            <m.button
              onClick={onRequestBridge}
              whileHover={{ scale: 1.02, borderColor: 'rgba(0,212,255,0.5)' }}
              whileTap={{ scale: 0.97 }}
              style={{
                padding: '6px 14px',
                background: 'rgba(0,212,255,0.08)',
                border: '1px solid rgba(0,212,255,0.25)',
                borderRadius: 'var(--radius-sm)',
                color: '#00d4ff',
                cursor: 'pointer',
                fontSize: 9,
                fontWeight: 600,
                letterSpacing: '0.08em',
                fontFamily: "'JetBrains Mono', monospace",
              }}
            >
              SUGGEST BRIDGE TRACKS
            </m.button>
          )}
        </GlassPanel>
      )}
    </div>
  );
}
