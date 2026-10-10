// Small presentational pieces for the Next Track screen.
import React, { useState } from 'react';
import { m } from 'framer-motion';
import { nextApi } from './nextApi';

const mono = "'JetBrains Mono', monospace";

function hashHue(s = '') {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return 190 + (h % 100); // cyan → violet only
}

/** Album art from the backend, with a cool generative fallback. */
export function Art({ track, size = 56, radius = 'var(--radius-sm)', glow = false }) {
  const [failed, setFailed] = useState(false);
  const id = track?.id;
  const hue = hashHue(id || track?.title);
  const initials = (track?.artist || track?.title || '?').split(/\s+/).slice(0, 2).map(w => w[0]).join('').toUpperCase();
  const common = {
    width: size, height: size, borderRadius: radius, flexShrink: 0,
    boxShadow: glow ? '0 20px 60px rgba(0,0,0,0.6), 0 0 80px rgba(124,58,237,0.25)' : '0 4px 14px rgba(0,0,0,0.45)',
  };
  if (!id || failed || String(id).startsWith('x')) {
    return (
      <div style={{
        ...common,
        background: `radial-gradient(circle at 30% 25%, hsl(${hue} 90% 62% / 0.55), transparent 55%),
                     radial-gradient(circle at 75% 80%, hsl(${hue + 40} 85% 55% / 0.5), transparent 60%),
                     linear-gradient(135deg, #12122a, #0a0a14)`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: 'rgba(226,232,240,0.75)', fontWeight: 700, fontSize: Math.max(10, size * 0.22),
        letterSpacing: '0.06em', border: '1px solid rgba(124,58,237,0.2)',
      }}>{initials}</div>
    );
  }
  return (
    <img
      src={nextApi.artUrl(id)} alt="" draggable={false}
      onError={() => setFailed(true)}
      style={{ ...common, objectFit: 'cover', background: '#10101e', display: 'block' }}
    />
  );
}

export function Chip({ children, color, title, strong }) {
  return (
    <span title={title} style={{
      fontFamily: mono, fontSize: 10, letterSpacing: '0.04em', whiteSpace: 'nowrap',
      padding: '3px 8px', borderRadius: 'var(--radius-pill)',
      color: color || 'var(--text-secondary)',
      background: strong ? `${color}1f` : 'rgba(255,255,255,0.04)',
      border: `1px solid ${strong ? `${color}55` : 'rgba(148,163,184,0.12)'}`,
    }}>{children}</span>
  );
}

/** Energy on the 1–5 star scale, drawn as five bars. */
export function EnergyBars({ energy, color = '#a855f7', height = 12 }) {
  if (energy == null) return null;
  return (
    <span title={`Energy ${energy.toFixed(1)} / 5`} style={{ display: 'inline-flex', gap: 2, alignItems: 'flex-end', height }}>
      {[1, 2, 3, 4, 5].map(i => {
        const fill = Math.max(0, Math.min(1, energy - (i - 1)));
        return (
          <span key={i} style={{
            width: 4, height: 4 + (height - 4) * (i / 5), borderRadius: 1,
            background: `linear-gradient(to top, ${color} ${fill * 100}%, rgba(148,163,184,0.15) ${fill * 100}%)`,
          }} />
        );
      })}
    </span>
  );
}

const AXIS_LABELS = [
  ['energy', 'Energy', '#a855f7'],
  ['dark', 'Dark', '#818cf8'],
  ['vocal', 'Vocal', '#c084fc'],
  ['deep', 'Deep', '#2dd4bf'],
];

/** Where this track sits in your library on each axis (percentiles). */
export function AxisBars({ pct }) {
  if (!pct) return null;
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '5px 10px', alignItems: 'center', width: '100%' }}>
      {AXIS_LABELS.map(([k, label, color]) => (
        <React.Fragment key={k}>
          <span style={{ fontSize: 9, fontFamily: mono, color: 'var(--text-muted)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>{label}</span>
          <div style={{ height: 4, borderRadius: 2, background: 'rgba(148,163,184,0.1)', position: 'relative', overflow: 'hidden' }}>
            <m.div
              initial={{ width: 0 }} animate={{ width: `${Math.round((pct[k] ?? 0) * 100)}%` }}
              transition={{ type: 'spring', damping: 24, stiffness: 180 }}
              style={{ position: 'absolute', inset: 0, right: 'auto', background: `linear-gradient(90deg, ${color}55, ${color})`, borderRadius: 2 }}
            />
          </div>
        </React.Fragment>
      ))}
    </div>
  );
}

export function IconButton({ onClick, title, children, active, disabled, label }) {
  return (
    <m.button
      onClick={onClick} title={title} disabled={disabled}
      whileHover={disabled ? undefined : { scale: 1.02 }} whileTap={disabled ? undefined : { scale: 0.97 }}
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 7,
        height: 34, padding: label ? '0 14px' : 0, width: label ? 'auto' : 34, justifyContent: 'center',
        borderRadius: 'var(--radius-pill)', cursor: disabled ? 'default' : 'pointer',
        background: active ? 'var(--gradient-accent-soft)' : 'rgba(255,255,255,0.04)',
        border: `1px solid ${active ? 'rgba(124,58,237,0.45)' : 'rgba(124,58,237,0.16)'}`,
        color: disabled ? 'var(--text-muted)' : 'var(--text-primary)',
        fontSize: 12, fontWeight: 600, fontFamily: 'var(--font-ui)', letterSpacing: '0.02em',
        opacity: disabled ? 0.5 : 1,
      }}
    >
      {children}{label && <span>{label}</span>}
    </m.button>
  );
}

export const Icon = {
  mic: (s = 15) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3" /></svg>,
  file: (s = 15) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 18V5l12-2v13" /><circle cx="6" cy="18" r="3" /><circle cx="18" cy="16" r="3" /></svg>,
  search: (s = 14) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round"><circle cx="11" cy="11" r="7" /><path d="M21 21l-4.3-4.3" /></svg>,
  live: (s = 14) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="12" cy="12" r="2.5" /><path d="M16.2 7.8a6 6 0 0 1 0 8.4M7.8 16.2a6 6 0 0 1 0-8.4M19 5a10 10 0 0 1 0 14M5 19A10 10 0 0 1 5 5" /></svg>,
  folder: (s = 13) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" /></svg>,
  back: (s = 13) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><path d="M15 6l-6 6 6 6" /></svg>,
  refresh: (s = 13) => <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12a9 9 0 1 1-2.6-6.4M21 4v5h-5" /></svg>,
};

export function fmtBpm(bpm) {
  if (!bpm) return '—';
  return Math.abs(bpm - Math.round(bpm)) < 0.05 ? `${Math.round(bpm)}` : bpm.toFixed(1);
}
