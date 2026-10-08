// src/components/NextTrack.jsx
//
// The Next Track screen. Shows the track on air in rekordbox in the middle of
// a compass, with one suggestion per direction around it (more energy,
// darker, faster, more vocal...). Click a suggestion once you've mixed into it
// and the compass re-centres on it; when rekordbox moves on, it follows.
//
// Starting points other than rekordbox: search the library, drop an audio
// file, or let the mic listen to whatever's playing in the room.
import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { m, AnimatePresence, useReducedMotion } from 'framer-motion';
import { nextApi, recordClip } from './next/nextApi';
import { DIRECTION_META, ICONS, SOURCE_LABEL } from './next/directions';
import { Art, AxisBars, Chip, EnergyBars, Icon, IconButton, fmtBpm } from './next/parts';

const mono = "'JetBrains Mono', monospace";
const POLL_MS = 2500;
const STALE_SECONDS = 6 * 3600;
const LISTEN_SECONDS = 15;

function useElementSize(ref) {
  const [size, setSize] = useState({ w: 0, h: 0 });
  useLayoutEffect(() => {
    if (!ref.current) return undefined;
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, [ref]);
  return size;
}

// ── Centre: the track on air ────────────────────────────────────────────────
function CentreTrack({ track, source, pct, compact, nearest, onJump, analysing, roomy = true }) {
  const reduce = useReducedMotion();
  const artSize = compact ? 132 : roomy ? 188 : 160;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14, textAlign: 'center', maxWidth: 320 }}>
      <div style={{
        display: 'inline-flex', alignItems: 'center', gap: 7, padding: '4px 11px',
        borderRadius: 'var(--radius-pill)', background: 'rgba(0,212,255,0.06)',
        border: '1px solid var(--border-cyan)', fontSize: 9.5, fontFamily: mono,
        letterSpacing: '0.14em', textTransform: 'uppercase',
        color: source === 'rekordbox' ? 'var(--cyan)' : source === 'rekordbox_stale' ? 'var(--text-secondary)' : 'var(--purple-light)',
      }}>
        <span style={{
          width: 6, height: 6, borderRadius: '50%',
          background: source === 'rekordbox' ? 'var(--cyan)' : source === 'rekordbox_stale' ? 'var(--text-muted)' : 'var(--purple-bright)',
          boxShadow: source === 'rekordbox_stale' ? 'none' : `0 0 8px ${source === 'rekordbox' ? 'var(--cyan)' : 'var(--purple-bright)'}`,
          animation: source === 'rekordbox' && !reduce ? 'statusPulse 2s ease-in-out infinite' : 'none',
        }} />
        {analysing ? 'Analysing new track' : SOURCE_LABEL[source] || 'Now'}
      </div>

      <div style={{ position: 'relative', width: artSize, height: artSize }}>
        {!reduce && (
          <m.div
            aria-hidden
            animate={{ rotate: 360 }}
            transition={{ duration: 18, repeat: Infinity, ease: 'linear' }}
            style={{
              position: 'absolute', inset: -10, borderRadius: 26,
              background: 'conic-gradient(from 0deg, rgba(0,212,255,0.0), rgba(0,212,255,0.45), rgba(124,58,237,0.0), rgba(168,85,247,0.5), rgba(0,212,255,0.0))',
              filter: 'blur(14px)', opacity: 0.8,
            }}
          />
        )}
        <AnimatePresence mode="wait">
          <m.div
            key={track?.id || 'none'}
            initial={{ opacity: 0, scale: 0.92 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.96 }}
            transition={{ type: 'spring', damping: 22, stiffness: 220 }}
            style={{ position: 'relative' }}
          >
            <Art track={track} size={artSize} radius="var(--radius-xl)" glow />
            {analysing && (
              <div style={{
                position: 'absolute', inset: 0, borderRadius: 'var(--radius-xl)',
                background: 'rgba(5,5,12,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <div className="nt-spinner" />
              </div>
            )}
          </m.div>
        </AnimatePresence>
      </div>

      <div style={{ minWidth: 0, width: '100%' }}>
        <div style={{
          fontSize: compact ? 17 : 21, fontWeight: 700, color: 'var(--text-primary)', lineHeight: 1.2,
          overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical',
        }}>{track?.title || 'Nothing playing'}</div>
        <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 4, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {track?.artist || ' '}
        </div>
      </div>

      {track && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, justifyContent: 'center', alignItems: 'center' }}>
          <Chip color="#00d4ff" strong title={track.bpm_estimated ? 'Estimated from the audio' : 'From rekordbox'}>
            {fmtBpm(track.bpm)} BPM{track.bpm_estimated ? '*' : ''}
          </Chip>
          <Chip color="#a855f7" strong title={track.key_estimated ? 'Estimated from the audio' : 'Camelot key'}>
            {track.key || '—'}{track.key_estimated ? '*' : ''}
          </Chip>
          <Chip><span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>ENERGY <EnergyBars energy={track.energy} height={10} /></span></Chip>
        </div>
      )}
      {track?.genres?.length > 0 && (
        <div style={{ fontSize: 11, color: 'var(--text-muted)', letterSpacing: '0.02em' }}>
          {track.genres.slice(0, 3).join('  ·  ')}
        </div>
      )}
      {pct && !compact && !(nearest && !roomy) && (
        <div style={{ width: 230, marginTop: 2 }}><AxisBars pct={pct} /></div>
      )}
      {nearest && (
        <m.button
          onClick={() => onJump(nearest.track)}
          whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.97 }}
          style={{
            display: 'flex', alignItems: 'center', gap: 10, padding: '7px 12px 7px 7px',
            borderRadius: 'var(--radius-md)', cursor: 'pointer', textAlign: 'left',
            background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(124,58,237,0.2)', color: 'inherit',
            maxWidth: 300,
          }}
        >
          <Art track={nearest.track} size={32} />
          <span style={{ minWidth: 0 }}>
            <span style={{ display: 'block', fontSize: 9, fontFamily: mono, letterSpacing: '0.12em', color: 'var(--text-muted)' }}>
              CLOSEST IN YOUR LIBRARY
            </span>
            <span style={{ display: 'block', fontSize: 12, color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {nearest.track.artist} – {nearest.track.title}
            </span>
          </span>
        </m.button>
      )}
    </div>
  );
}

// ── One direction on the compass ────────────────────────────────────────────
function DirectionCard({ dir, onPick, onReveal, cardRef, popUp, compact }) {
  const meta = DIRECTION_META[dir.id] || { color: '#94a3b8' };
  const IconC = ICONS[dir.id];
  const [hover, setHover] = useState(false);
  const [main, ...alts] = dir.tracks || [];
  const color = meta.color;

  return (
    <div
      ref={cardRef}
      onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}
      style={{ position: 'relative', width: '100%', maxWidth: compact ? 'none' : 290, zIndex: hover ? 30 : 1 }}
    >
      <m.div
        layout
        whileHover={main ? { y: -2 } : undefined}
        onClick={() => main && onPick(main)}
        role={main ? 'button' : undefined}
        tabIndex={main ? 0 : -1}
        onKeyDown={(e) => { if (main && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); onPick(main); } }}
        style={{
          cursor: main ? 'pointer' : 'default',
          background: hover && main ? 'var(--bg-card-hover)' : 'var(--glass-bg)',
          backdropFilter: 'blur(var(--glass-blur))', WebkitBackdropFilter: 'blur(var(--glass-blur))',
          border: `1px solid ${hover && main ? `${color}66` : 'var(--glass-border)'}`,
          borderRadius: 'var(--radius-lg)', padding: 12,
          boxShadow: hover && main ? `var(--shadow-card-hover), 0 0 28px ${color}22` : 'var(--shadow-card)',
          transition: 'background 200ms, border-color 200ms, box-shadow 200ms',
          outline: 'none',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: main ? 10 : 4 }}>
          <span style={{
            width: 24, height: 24, borderRadius: 7, display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
            color, background: `${color}1a`, border: `1px solid ${color}40`,
          }}>{IconC ? <IconC size={13} /> : null}</span>
          <span style={{ fontSize: 10.5, fontWeight: 700, letterSpacing: '0.13em', textTransform: 'uppercase', color }}>{dir.label}</span>
          <span style={{ marginLeft: 'auto', fontSize: 10, color: 'var(--text-muted)' }}>{meta.blurb}</span>
        </div>

        {main ? (
          <div style={{ display: 'flex', gap: 11, alignItems: 'center', minWidth: 0 }}>
            <Art track={main} size={52} />
            <div style={{ minWidth: 0, flex: 1 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{main.title}</div>
              <div style={{ fontSize: 11.5, color: 'var(--text-secondary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', marginTop: 1 }}>{main.artist}</div>
              <div style={{ display: 'flex', gap: 4, marginTop: 7, flexWrap: 'nowrap', overflow: 'hidden' }}>
                {(main.reasons || []).slice(0, 3).map((r, i) => {
                  const lead = i === 0 && dir.id !== 'closest';
                  return <Chip key={i} color={lead ? color : undefined} strong={lead}>{r}</Chip>;
                })}
              </div>
            </div>
          </div>
        ) : (
          <div style={{ fontSize: 11.5, color: 'var(--text-muted)', padding: '6px 0 2px' }}>{dir.empty_reason || 'Nothing that way'}</div>
        )}
      </m.div>

      {/* Alternates + actions on hover */}
      <AnimatePresence>
        {hover && main && !compact && (
          <m.div
            initial={{ opacity: 0, y: popUp ? 6 : -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: popUp ? 6 : -6 }}
            transition={{ duration: 0.15 }}
            style={{
              position: 'absolute', left: 0, right: 0, [popUp ? 'bottom' : 'top']: 'calc(100% + 6px)',
              background: 'var(--bg-glass-heavy)', backdropFilter: 'blur(24px)', WebkitBackdropFilter: 'blur(24px)',
              border: '1px solid var(--glass-border)', borderRadius: 'var(--radius-md)', padding: 8,
              boxShadow: 'var(--shadow-float)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 4px 6px' }}>
              <span style={{ fontSize: 9, fontFamily: mono, letterSpacing: '0.12em', color: 'var(--text-muted)' }}>CLICK WHEN YOU'VE MIXED IN</span>
              <button
                onClick={(e) => { e.stopPropagation(); onReveal(main); }}
                title="Show in Finder, then drag onto a deck"
                style={{ display: 'inline-flex', alignItems: 'center', gap: 5, background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: 10.5, padding: 2 }}
              >{Icon.folder(12)} Finder</button>
            </div>
            {alts.length === 0 && <div style={{ fontSize: 11, color: 'var(--text-muted)', padding: '2px 4px 4px' }}>No other options this way.</div>}
            {alts.map(t => (
              <button
                key={t.id}
                onClick={(e) => { e.stopPropagation(); onPick(t); }}
                style={{
                  display: 'flex', alignItems: 'center', gap: 9, width: '100%', padding: 5, borderRadius: 8,
                  background: 'none', border: 'none', cursor: 'pointer', textAlign: 'left', color: 'inherit',
                }}
                onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(124,58,237,0.1)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = 'none'; }}
              >
                <Art track={t} size={30} />
                <span style={{ minWidth: 0, flex: 1 }}>
                  <span style={{ display: 'block', fontSize: 12, color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{t.title}</span>
                  <span style={{ display: 'block', fontSize: 10.5, color: 'var(--text-muted)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{t.artist} · {(t.reasons || []).slice(0, 2).join(' · ')}</span>
                </span>
              </button>
            ))}
          </m.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ── Animated arrows from the centre out to each card ───────────────────────
function ArrowLayer({ stageRef, centreRef, cardRefs, dirs, size, reduce }) {
  const [lines, setLines] = useState([]);
  useLayoutEffect(() => {
    // Measure now, and again once the cards' entry animation has settled.
    measure();
    const t = setTimeout(measure, 320);
    return () => clearTimeout(t);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dirs, size.w, size.h]);

  function measure() {
    const stage = stageRef.current, centre = centreRef.current;
    if (!stage || !centre) return;
    const s = stage.getBoundingClientRect();
    const c = centre.getBoundingClientRect();
    const cx = c.left + c.width / 2 - s.left, cy = c.top + c.height / 2 - s.top;
    const rCentre = Math.min(c.width, c.height) / 2 + 6;
    const out = [];
    for (const d of dirs) {
      const el = cardRefs.current[d.id];
      if (!el) continue;
      const r = el.getBoundingClientRect();
      const bx = Math.max(r.left, Math.min(cx + s.left, r.right)) - s.left;
      const by = Math.max(r.top, Math.min(cy + s.top, r.bottom)) - s.top;
      const dx = bx - cx, dy = by - cy;
      const len = Math.hypot(dx, dy);
      if (len < rCentre + 20) continue;
      const ux = dx / len, uy = dy / len;
      out.push({
        id: d.id, color: (DIRECTION_META[d.id] || {}).color || '#94a3b8',
        x1: cx + ux * rCentre, y1: cy + uy * rCentre, x2: bx - ux * 8, y2: by - uy * 8,
        empty: !d.tracks?.length,
      });
    }
    setLines(out);
  }

  return (
    <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', overflow: 'visible', zIndex: 0 }}>
      <defs>
        {lines.map(l => (
          <linearGradient key={l.id} id={`nt-g-${l.id}`} gradientUnits="userSpaceOnUse" x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2}>
            <stop offset="0" stopColor={l.color} stopOpacity="0.05" />
            <stop offset="1" stopColor={l.color} stopOpacity={l.empty ? 0.15 : 0.75} />
          </linearGradient>
        ))}
      </defs>
      {lines.map(l => {
        const path = `M${l.x1},${l.y1} L${l.x2},${l.y2}`;
        const ang = Math.atan2(l.y2 - l.y1, l.x2 - l.x1);
        const head = 7;
        const hx = l.x2, hy = l.y2;
        const p1 = `${hx - head * Math.cos(ang - 0.45)},${hy - head * Math.sin(ang - 0.45)}`;
        const p2 = `${hx - head * Math.cos(ang + 0.45)},${hy - head * Math.sin(ang + 0.45)}`;
        return (
          <g key={l.id}>
            <path d={path} stroke={`url(#nt-g-${l.id})`} strokeWidth="1.5" strokeDasharray={l.empty ? '2 6' : '5 7'}
              className={reduce || l.empty ? undefined : 'nt-dash'} fill="none" />
            {!l.empty && <polyline points={`${p1} ${hx},${hy} ${p2}`} fill="none" stroke={l.color} strokeOpacity="0.8" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />}
            {!l.empty && !reduce && (
              <circle r="2.6" fill={l.color} opacity="0.9">
                <animateMotion dur="2.6s" repeatCount="indefinite" path={path} keyPoints="0;1" keyTimes="0;1" calcMode="spline" keySplines="0.4 0 0.2 1" />
                <animate attributeName="opacity" values="0;1;0" dur="2.6s" repeatCount="indefinite" />
              </circle>
            )}
          </g>
        );
      })}
    </svg>
  );
}

// ── Library search (pick any starting track) ───────────────────────────────
function SearchBox({ onPick, disabled }) {
  const [q, setQ] = useState('');
  const [results, setResults] = useState([]);
  const [open, setOpen] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === '/' && document.activeElement?.tagName !== 'INPUT') { e.preventDefault(); inputRef.current?.focus(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  useEffect(() => {
    if (!q.trim()) return undefined;
    const t = setTimeout(() => {
      nextApi.search(q).then(setResults).catch(() => setResults([]));
    }, 160);
    return () => clearTimeout(t);
  }, [q]);

  const onType = (value) => {
    setQ(value);
    setOpen(true);
    if (!value.trim()) setResults([]);
  };

  return (
    <div style={{ position: 'relative' }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8, height: 34, padding: '0 12px', width: 240,
        borderRadius: 'var(--radius-pill)', background: 'rgba(255,255,255,0.04)',
        border: '1px solid rgba(124,58,237,0.16)', color: 'var(--text-muted)',
      }}>
        {Icon.search()}
        <input
          ref={inputRef} value={q} disabled={disabled}
          onChange={(e) => onType(e.target.value)}
          onFocus={() => setOpen(true)} onBlur={() => setTimeout(() => setOpen(false), 150)}
          onKeyDown={(e) => { if (e.key === 'Enter' && results[0]) { onPick(results[0]); onType(''); setOpen(false); e.currentTarget.blur(); } }}
          placeholder="Start from any track   /"
          style={{ flex: 1, minWidth: 0, background: 'none', border: 'none', outline: 'none', color: 'var(--text-primary)', fontSize: 12.5, fontFamily: 'var(--font-ui)' }}
        />
      </div>
      <AnimatePresence>
        {open && results.length > 0 && (
          <m.div
            initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }}
            style={{
              position: 'absolute', top: 40, right: 0, width: 320, maxHeight: 360, overflowY: 'auto', zIndex: 60,
              background: 'var(--bg-glass-heavy)', backdropFilter: 'blur(24px)', WebkitBackdropFilter: 'blur(24px)',
              border: '1px solid var(--glass-border)', borderRadius: 'var(--radius-md)', padding: 6, boxShadow: 'var(--shadow-float)',
            }}
          >
            {results.map(t => (
              <button
                key={t.id}
                onMouseDown={(e) => { e.preventDefault(); onPick(t); onType(''); setOpen(false); }}
                style={{ display: 'flex', alignItems: 'center', gap: 10, width: '100%', padding: 6, borderRadius: 8, background: 'none', border: 'none', cursor: 'pointer', textAlign: 'left', color: 'inherit' }}
                onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(124,58,237,0.1)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = 'none'; }}
              >
                <Art track={t} size={32} />
                <span style={{ minWidth: 0, flex: 1 }}>
                  <span style={{ display: 'block', fontSize: 12.5, color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{t.title}</span>
                  <span style={{ display: 'block', fontSize: 11, color: 'var(--text-muted)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{t.artist}</span>
                </span>
                <span style={{ fontFamily: mono, fontSize: 10, color: 'var(--text-muted)' }}>{fmtBpm(t.bpm)} · {t.key || '—'}</span>
              </button>
            ))}
          </m.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ── Listening overlay ───────────────────────────────────────────────────────
function ListenOverlay({ progress, level, phase, onCancel }) {
  const R = 74, C = 2 * Math.PI * R;
  return (
    <m.div
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      style={{
        position: 'absolute', inset: 0, zIndex: 80, display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'rgba(5,5,12,0.72)', backdropFilter: 'blur(10px)', WebkitBackdropFilter: 'blur(10px)',
      }}
      onClick={onCancel}
    >
      <div onClick={(e) => e.stopPropagation()} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 18 }}>
        <div style={{ position: 'relative', width: 180, height: 180 }}>
          <m.div
            animate={{ scale: 1 + level * 0.35, opacity: 0.25 + level * 0.6 }}
            transition={{ duration: 0.08 }}
            style={{ position: 'absolute', inset: 30, borderRadius: '50%', background: 'radial-gradient(circle, rgba(0,212,255,0.55), rgba(124,58,237,0.15) 70%, transparent)' }}
          />
          <svg width="180" height="180" style={{ position: 'absolute', inset: 0, transform: 'rotate(-90deg)' }}>
            <circle cx="90" cy="90" r={R} stroke="rgba(148,163,184,0.12)" strokeWidth="3" fill="none" />
            <circle cx="90" cy="90" r={R} stroke="url(#nt-listen)" strokeWidth="3" fill="none" strokeLinecap="round"
              strokeDasharray={C} strokeDashoffset={C * (1 - progress)} style={{ transition: 'stroke-dashoffset 120ms linear' }} />
            <defs>
              <linearGradient id="nt-listen" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#00d4ff" /><stop offset="1" stopColor="#a855f7" /></linearGradient>
            </defs>
          </svg>
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-primary)' }}>
            {phase === 'analysing' ? <div className="nt-spinner" /> : Icon.mic(34)}
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>
            {phase === 'analysing' ? 'Working out what that was' : 'Listening to the room'}
          </div>
          <div style={{ fontSize: 12.5, color: 'var(--text-secondary)', marginTop: 6 }}>
            {phase === 'analysing'
              ? 'Matching it against your library'
              : `Hold the laptop near a speaker · ${Math.ceil(LISTEN_SECONDS * (1 - progress))}s`}
          </div>
        </div>
        {phase !== 'analysing' && <IconButton onClick={onCancel} label="Cancel" />}
      </div>
    </m.div>
  );
}

// ── First-run: analyse the library ──────────────────────────────────────────
function Onboarding({ status, onBuild }) {
  const b = status?.index?.build || {};
  const running = b.running;
  const pctDone = b.total ? b.done / b.total : 0;
  return (
    <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24 }}>
      <div style={{
        width: 'min(520px, 100%)', padding: 32, borderRadius: 'var(--radius-xl)',
        background: 'var(--glass-bg)', border: '1px solid var(--glass-border)', boxShadow: 'var(--shadow-panel)',
        backdropFilter: 'blur(24px)', WebkitBackdropFilter: 'blur(24px)', textAlign: 'center',
      }}>
        <div style={{ fontSize: 10, fontFamily: mono, letterSpacing: '0.16em', color: 'var(--cyan)', marginBottom: 12 }}>NEXT TRACK</div>
        <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--text-primary)' }}>
          {running ? 'Listening to your library' : 'First, let it hear your music'}
        </div>
        <p style={{ fontSize: 13.5, color: 'var(--text-secondary)', lineHeight: 1.6, margin: '12px 0 22px' }}>
          Every track in rekordbox and your Mixing folder gets analysed once for energy, darkness,
          vocals and texture. It takes about two seconds a track and runs on this Mac.
        </p>
        {running ? (
          <>
            <div style={{ height: 6, borderRadius: 3, background: 'rgba(148,163,184,0.12)', overflow: 'hidden' }}>
              <m.div animate={{ width: `${Math.round(pctDone * 100)}%` }} style={{ height: '100%', background: 'var(--gradient-accent)' }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 10, fontFamily: mono, fontSize: 10.5, color: 'var(--text-muted)' }}>
              <span>{b.done} / {b.total}</span>
              <span style={{ maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{b.current}</span>
            </div>
          </>
        ) : (
          <m.button
            onClick={onBuild} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.97 }}
            style={{
              padding: '12px 26px', borderRadius: 'var(--radius-pill)', border: 'none', cursor: 'pointer',
              background: 'var(--gradient-accent)', color: 'white', fontWeight: 700, fontSize: 13.5, letterSpacing: '0.02em',
              boxShadow: 'var(--shadow-glow-purple)',
            }}
          >Analyse my library</m.button>
        )}
        {b.error && <div style={{ marginTop: 14, fontSize: 12, color: '#f0abfc' }}>{b.error}</div>}
      </div>
    </div>
  );
}

function Notice({ title, body }) {
  return (
    <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24 }}>
      <div style={{ maxWidth: 440, textAlign: 'center' }}>
        <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-primary)' }}>{title}</div>
        <p style={{ fontSize: 13.5, color: 'var(--text-secondary)', lineHeight: 1.6, marginTop: 10 }}>{body}</p>
      </div>
    </div>
  );
}

// ── The screen ──────────────────────────────────────────────────────────────
export default function NextTrack({ topInset = 84 }) {
  const reduce = useReducedMotion();
  const [status, setStatus] = useState(null);
  const [backendDown, setBackendDown] = useState(false);
  const [live, setLive] = useState(null);                // last /now response
  const [centre, setCentre] = useState(null);            // { track, source }
  const [suggestions, setSuggestions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [trail, setTrail] = useState([]);                // tracks stepped into by hand
  const [nearest, setNearest] = useState(null);
  const [listen, setListen] = useState(null);            // { phase, progress, level }
  const [toast, setToast] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const lastLiveId = useRef(null);
  const listenRef = useRef(null);
  const stageRef = useRef(null);
  const centreRef = useRef(null);
  const cardRefs = useRef({});
  const fileInput = useRef(null);
  const stageSize = useElementSize(stageRef);
  const compact = stageSize.w > 0 && (stageSize.w < 900 || stageSize.h < 600);

  const say = useCallback((msg) => {
    setToast(msg);
    setTimeout(() => setToast(t => (t === msg ? null : t)), 3600);
  }, []);

  // ── status / index progress ──
  const refreshStatus = useCallback(() => (
    nextApi.status().then(s => { setStatus(s); setBackendDown(false); return s; })
      .catch(() => { setBackendDown(true); return null; })
  ), []);
  useEffect(() => { refreshStatus(); }, [refreshStatus]);
  const building = status?.index?.build?.running;
  const analysed = status?.index?.analysed || 0;
  useEffect(() => {
    if (!building && analysed > 0) return undefined;
    const t = setInterval(refreshStatus, 2500);
    return () => clearInterval(t);
  }, [building, analysed, refreshStatus]);

  // ── suggestions for the current centre ──
  // rekordbox's newest history row can be from weeks ago. Older than this and
  // it's "last played", not live, and that old session doesn't exclude tracks.
  const stale = (live?.age_seconds ?? 0) > STALE_SECONDS;

  // A ref, not a dependency: the played list changes on every poll and the
  // polling effect must not restart each time.
  const playedIdsRef = useRef([]);
  playedIdsRef.current = stale ? [] : (live?.played_ids || []);
  const requestSeq = useRef(0);
  const loadSuggestions = useCallback((track, extraExclude = []) => {
    if (!track?.id) return;
    const seq = ++requestSeq.current;
    setLoading(true);
    const exclude = [...new Set([...playedIdsRef.current, ...extraExclude])].filter(id => id !== track.id);
    nextApi.suggest(track.id, exclude)
      .then(res => {
        if (seq !== requestSeq.current) return;   // a newer centre won
        setSuggestions(res);
        setCentre(c => (c && c.track.id === track.id ? { ...c, track: { ...c.track, ...res.current } } : c));
      })
      .catch(err => { if (seq === requestSeq.current) { setSuggestions(null); say(err.message); } })
      .finally(() => { if (seq === requestSeq.current) setLoading(false); });
  }, [say]);

  const goTo = useCallback((track, source, opts = {}) => {
    setCentre({ track, source });
    if (!opts.keepNearest) setNearest(null);
    loadSuggestions(track, opts.exclude || []);
  }, [loadSuggestions]);

  // ── follow rekordbox ──
  useEffect(() => {
    if (backendDown || !analysed) return undefined;
    let stopped = false;
    // Poll even when the window is hidden behind rekordbox, so the compass is
    // already up to date when you glance across at it.
    const poll = () => {
      nextApi.now().then(n => {
        if (stopped) return;
        setLive(n);
        const t = n?.track;
        if (!n?.available || !t?.id || n.state !== 'ready') return;
        if (t.id !== lastLiveId.current) {
          lastLiveId.current = t.id;
          setTrail([]);
          const old = (n.age_seconds ?? 0) > STALE_SECONDS;
          goTo(t, old ? 'rekordbox_stale' : 'rekordbox', { exclude: old ? [] : n.played_ids });
        }
      }).catch(() => {});
    };
    poll();
    const iv = setInterval(poll, POLL_MS);
    return () => { stopped = true; clearInterval(iv); };
  }, [backendDown, analysed, goTo]);

  const isLiveCentre = centre?.source?.startsWith('rekordbox');
  const liveSource = stale ? 'rekordbox_stale' : 'rekordbox';
  const liveTrack = live?.state === 'ready' ? live?.track : null;
  const liveAnalysing = live?.state === 'analysing';

  const stepInto = useCallback((t) => {
    setTrail(tr => [...tr, centre?.track].filter(Boolean).slice(-8));
    goTo(t, 'explore', { exclude: [centre?.track?.id, ...trail.map(x => x.id)].filter(Boolean) });
  }, [centre, trail, goTo]);

  const backToLive = useCallback(() => {
    if (!liveTrack) return;
    setTrail([]);
    goTo(liveTrack, liveSource);
  }, [liveTrack, liveSource, goTo]);

  const backTo = useCallback((i) => {
    const t = trail[i];
    setTrail(tr => tr.slice(0, i));
    goTo(t, i === 0 && lastLiveId.current === t.id ? liveSource : 'explore');
  }, [trail, liveSource, goTo]);

  // ── analyse external audio ──
  const analyseBlob = useCallback(async (blob, name, kind) => {
    try {
      const res = await nextApi.analyse(blob, name, kind);
      setTrail(centre?.track ? [centre.track] : []);
      if (res.nearest?.confident) {
        // It's one of yours: centre on your copy, with rekordbox's BPM and key.
        const t = res.nearest.track;
        setCentre({ track: t, source: kind });
        setNearest(null);
        loadSuggestions(t);
        say(`Recognised: ${t.artist} – ${t.title}`);
      } else {
        setCentre({ track: res.track, source: kind });
        setNearest(res.nearest);
        loadSuggestions(res.track);
      }
    } catch (err) {
      say(err.message);
    }
  }, [centre, loadSuggestions, say]);

  const startListening = useCallback(() => {
    if (!navigator.mediaDevices?.getUserMedia) { say('This browser can\'t use the microphone'); return; }
    setListen({ phase: 'recording', progress: 0, level: 0 });
    const rec = recordClip(LISTEN_SECONDS, {
      onLevel: (level) => setListen(l => (l ? { ...l, level } : l)),
      onProgress: (progress) => setListen(l => (l ? { ...l, progress } : l)),
    });
    listenRef.current = rec;
    rec.promise
      .then(async (blob) => {
        setListen({ phase: 'analysing', progress: 1, level: 0 });
        await analyseBlob(blob, 'mic.wav', 'mic');
      })
      .catch(err => { if (err.message !== 'cancelled') say(err.name === 'NotAllowedError' ? 'Microphone access was blocked' : err.message); })
      .finally(() => { setListen(null); listenRef.current = null; });
  }, [analyseBlob, say]);

  const cancelListening = useCallback(() => {
    listenRef.current?.cancel();
    setListen(null);
  }, []);

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') cancelListening(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [cancelListening]);

  const onFiles = useCallback(async (files) => {
    const f = files?.[0];
    if (!f) return;
    if (!/^audio\//.test(f.type) && !/\.(mp3|wav|flac|m4a|aiff?|ogg|aac)$/i.test(f.name)) { say('That isn\'t an audio file'); return; }
    setListen({ phase: 'analysing', progress: 1, level: 0 });
    try { await analyseBlob(f, f.name, 'file'); } finally { setListen(null); }
  }, [analyseBlob, say]);

  const reveal = useCallback((t) => {
    nextApi.reveal(t.id).then(() => say('Shown in Finder. Drag it onto a deck')).catch(err => say(err.message));
  }, [say]);

  const dirs = suggestions?.directions || [];
  const centrePct = suggestions?.current?.pct;

  // ── render ──
  if (backendDown) {
    return <div style={{ position: 'absolute', inset: 0, paddingTop: topInset }}>
      <Notice title="Next Track runs on your Mac" body="Open the DJMate desktop app (or start the backend on port 8000). It reads rekordbox and your audio files locally, so it can't run from the website alone." />
    </div>;
  }
  if (status && !analysed) {
    return <div style={{ position: 'absolute', inset: 0, paddingTop: topInset }}>
      <Onboarding status={status} onBuild={() => nextApi.build().then(refreshStatus).catch(err => say(err.message))} />
    </div>;
  }

  const rb = status?.rekordbox;
  const showFollow = liveTrack && !isLiveCentre && liveTrack.id !== centre?.track?.id;

  return (
    <div
      style={{ position: 'absolute', inset: 0, paddingTop: topInset, display: 'flex', flexDirection: 'column', fontFamily: 'var(--font-ui)' }}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={(e) => { if (e.currentTarget === e.target) setDragOver(false); }}
      onDrop={(e) => { e.preventDefault(); setDragOver(false); onFiles(e.dataTransfer.files); }}
    >
      {/* Ambient background */}
      <div aria-hidden style={{
        position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 0,
        background: 'radial-gradient(ellipse 60% 50% at 50% 50%, rgba(124,58,237,0.10), transparent 70%), radial-gradient(ellipse 40% 35% at 50% 55%, rgba(0,212,255,0.06), transparent 70%)',
      }} />

      {/* Toolbar */}
      <div style={{ position: 'relative', zIndex: 40, display: 'flex', alignItems: 'center', gap: 10, padding: '6px 24px 0', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, color: 'var(--text-muted)', fontFamily: mono, letterSpacing: '0.06em' }}>
          <span style={{
            width: 7, height: 7, borderRadius: '50%',
            background: rb?.available ? (liveTrack || liveAnalysing ? 'var(--cyan)' : '#475569') : '#64748b',
            boxShadow: rb?.available && liveTrack ? '0 0 8px var(--cyan)' : 'none',
          }} />
          {rb?.available
            ? (stale && live?.started_at
              ? `REKORDBOX · LAST SESSION ${new Date(live.started_at).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' }).toUpperCase()}`
              : liveTrack || liveAnalysing ? 'REKORDBOX CONNECTED' : 'REKORDBOX · WAITING FOR A PLAY')
            : 'REKORDBOX NOT FOUND'}
          {building && <span style={{ color: 'var(--purple-light)' }}>· ANALYSING {status.index.build.done}/{status.index.build.total}</span>}
        </div>
        <div style={{ flex: 1 }} />
        <AnimatePresence>
          {showFollow && (
            <m.div initial={{ opacity: 0, x: 8 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }}>
              <IconButton onClick={backToLive} active label={`Back to live: ${liveTrack.title}`}>{Icon.live()}</IconButton>
            </m.div>
          )}
        </AnimatePresence>
        <SearchBox onPick={(t) => { setTrail(centre?.track ? [centre.track] : []); goTo(t, 'search'); }} />
        <IconButton onClick={startListening} label="Listen" title="Identify what's playing in the room">{Icon.mic()}</IconButton>
        <IconButton onClick={() => fileInput.current?.click()} label="Analyse file" title="Analyse any audio file, even one that isn't yours">{Icon.file()}</IconButton>
        <input ref={fileInput} type="file" accept="audio/*" style={{ display: 'none' }} onChange={(e) => { onFiles(e.target.files); e.target.value = ''; }} />
      </div>

      {/* Stage */}
      <div ref={stageRef} style={{ position: 'relative', zIndex: 1, flex: 1, minHeight: 0, padding: compact ? '14px 16px' : '14px 28px 10px' }}>
        {!centre && (!status || (analysed && live === null)) && (
          <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div className="nt-spinner" />
          </div>
        )}
        {!centre && status && live !== null && (
          <Notice
            title={liveAnalysing ? 'Analysing the track on air'
              : live?.available === false ? 'Can\'t see rekordbox on this Mac' : 'Play something in rekordbox'}
            body={liveAnalysing
              ? 'It isn\'t in your analysed library yet, so it\'s being listened to now. A few seconds.'
              : live?.available === false
                ? 'You can still search for a track, drop an audio file here, or press Listen to hear the room.'
                : 'The compass follows whatever rekordbox plays. Or search for a track, drop an audio file here, or press Listen to hear the room.'}
          />
        )}

        {centre && !compact && (
          <>
            <div style={{
              position: 'relative', zIndex: 1, height: '100%', display: 'grid',
              gridTemplateColumns: 'minmax(220px, 1fr) minmax(300px, 1.15fr) minmax(220px, 1fr)',
              gridTemplateRows: 'auto 1fr auto', gap: '18px 36px',
            }}>
              {dirs.map(d => {
                const meta = DIRECTION_META[d.id] || {};
                const justify = meta.col === 1 ? 'flex-start' : meta.col === 3 ? 'flex-end' : 'center';
                const align = meta.row === 1 ? 'flex-start' : meta.row === 3 ? 'flex-end' : 'center';
                return (
                  <m.div
                    key={d.id}
                    initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: loading ? 0.55 : 1, scale: 1 }}
                    transition={{ duration: 0.25 }}
                    style={{ gridRow: meta.row, gridColumn: meta.col, display: 'flex', justifyContent: justify, alignItems: align, minWidth: 0 }}
                  >
                    <DirectionCard
                      dir={d} onPick={stepInto} onReveal={reveal} popUp={meta.row === 3}
                      cardRef={(el) => { cardRefs.current[d.id] = el; }}
                    />
                  </m.div>
                );
              })}
              <div style={{ gridRow: 2, gridColumn: 2, display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 0 }}>
                <div ref={centreRef}>
                  <CentreTrack track={centre.track} source={centre.source} pct={centrePct} nearest={nearest}
                    analysing={false} onJump={(t) => stepInto(t)}
                    roomy={stageSize.h >= (nearest ? 860 : 700)} />
                </div>
              </div>
            </div>
            {/* After the cards so their refs exist when the arrows measure;
                zIndex keeps the arrows underneath. */}
            <ArrowLayer stageRef={stageRef} centreRef={centreRef} cardRefs={cardRefs} dirs={dirs} size={stageSize} reduce={reduce} />
          </>
        )}

        {centre && compact && (
          <div style={{ height: '100%', overflowY: 'auto', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 18, paddingBottom: 24 }}>
            <div ref={centreRef}><CentreTrack track={centre.track} source={centre.source} compact nearest={nearest} onJump={stepInto} /></div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: 10, width: '100%' }}>
              {dirs.map(d => (
                <DirectionCard key={d.id} dir={d} onPick={stepInto} onReveal={reveal} compact cardRef={() => {}} />
              ))}
            </div>
          </div>
        )}

        <AnimatePresence>
          {dragOver && (
            <m.div
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              style={{
                position: 'absolute', inset: 16, zIndex: 70, borderRadius: 'var(--radius-xl)', pointerEvents: 'none',
                border: '2px dashed rgba(0,212,255,0.5)', background: 'rgba(0,212,255,0.05)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 15, fontWeight: 600, color: 'var(--cyan)',
              }}
            >Drop to find where it fits in your library</m.div>
          )}
        </AnimatePresence>
      </div>

      {/* Footer: your path tonight */}
      {(trail.length > 0 || (live?.played?.length || 0) > 1) && (
        <div style={{ position: 'relative', zIndex: 2, display: 'flex', alignItems: 'center', gap: 10, padding: '8px 24px 16px', overflowX: 'auto', whiteSpace: 'nowrap' }}>
          {trail.length > 0 ? (
            <>
              <span style={{ fontSize: 9.5, fontFamily: mono, letterSpacing: '0.14em', color: 'var(--text-muted)' }}>YOUR PATH</span>
              {trail.map((t, i) => (
                <button key={`${t.id}-${i}`} onClick={() => backTo(i)} title="Go back to this track"
                  style={{ display: 'inline-flex', alignItems: 'center', gap: 7, padding: '4px 10px 4px 4px', borderRadius: 'var(--radius-pill)', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(124,58,237,0.15)', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: 11.5 }}>
                  <Art track={t} size={20} radius="50%" />{t.title}
                </button>
              ))}
              <span style={{ color: 'var(--text-muted)' }}>→</span>
              <span style={{ fontSize: 11.5, color: 'var(--text-primary)', fontWeight: 600 }}>{centre?.track?.title}</span>
            </>
          ) : (
            <>
              <span style={{ fontSize: 9.5, fontFamily: mono, letterSpacing: '0.14em', color: 'var(--text-muted)' }}>{stale ? 'LAST SESSION' : 'PLAYED TONIGHT'} · {live.played.length}</span>
              {live.played.slice(0, -1).slice(-10).map(p => (
                <span key={p.id + p.played_at} title={`${p.artist} – ${p.title}`} style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text-muted)', opacity: 0.8 }}>
                  <Art track={p} size={20} radius="50%" />
                  <span style={{ maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis' }}>{p.title}</span>
                </span>
              ))}
            </>
          )}
        </div>
      )}

      <AnimatePresence>
        {listen && <ListenOverlay {...listen} onCancel={cancelListening} />}
      </AnimatePresence>

      <AnimatePresence>
        {toast && (
          <m.div
            initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 10 }}
            style={{
              position: 'absolute', bottom: 22, left: '50%', transform: 'translateX(-50%)', zIndex: 90,
              padding: '9px 16px', borderRadius: 'var(--radius-pill)', fontSize: 12.5, color: 'var(--text-primary)',
              background: 'var(--bg-glass-heavy)', border: '1px solid var(--glass-border)', boxShadow: 'var(--shadow-float)',
            }}
          >{toast}</m.div>
        )}
      </AnimatePresence>
    </div>
  );
}
