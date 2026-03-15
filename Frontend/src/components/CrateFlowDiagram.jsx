import React, { useMemo } from 'react';
import { m } from 'framer-motion';

const NODE_W = 180;
const NODE_H = 100;
const X_STEP = 220;
const Y_STEP = 130;

function buildLayout(crates) {
  const byParent = {};
  const roots = [];

  for (const c of crates) {
    if (!c.parent_crate_id) {
      roots.push(c);
    } else {
      if (!byParent[c.parent_crate_id]) byParent[c.parent_crate_id] = [];
      byParent[c.parent_crate_id].push(c);
    }
  }

  const positions = {};
  let yCounter = 0;

  function dfs(node, depth) {
    const children = byParent[node.id] || [];
    if (children.length === 0) {
      positions[node.id] = { x: depth * X_STEP, y: yCounter * Y_STEP };
      yCounter++;
    } else {
      const startY = yCounter;
      for (const child of children) dfs(child, depth + 1);
      const endY = yCounter - 1;
      positions[node.id] = {
        x: depth * X_STEP,
        y: ((startY + endY) / 2) * Y_STEP,
      };
    }
  }

  for (const root of roots) dfs(root, 0);
  return positions;
}

export default function CrateFlowDiagram({
  crates = [],
  selectedCrateId,
  onSelectCrate,
  onAddBranch,
}) {
  const positions = useMemo(() => buildLayout(crates), [crates]);

  const maxX = Math.max(0, ...Object.values(positions).map((p) => p.x)) + X_STEP;
  const maxY = Math.max(0, ...Object.values(positions).map((p) => p.y)) + Y_STEP + 10;

  return (
    <div style={{ overflow: 'auto', flex: 1, padding: 16, position: 'relative' }}>
      <svg
        width={maxX}
        height={maxY}
        style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
      >
        {crates
          .filter((c) => c.parent_crate_id && positions[c.id] && positions[c.parent_crate_id])
          .map((c) => {
            const from = positions[c.parent_crate_id];
            const to = positions[c.id];
            const hasBridgeWarning = c.metadata?.bridge_warning?.needs_bridge;
            const x1 = from.x + NODE_W;
            const y1 = from.y + NODE_H / 2;
            const x2 = to.x;
            const y2 = to.y + NODE_H / 2;
            const cx1 = x1 + 40;
            const cx2 = x2 - 40;

            return (
              <path
                key={`line-${c.id}`}
                d={`M ${x1} ${y1} C ${cx1} ${y1}, ${cx2} ${y2}, ${x2} ${y2}`}
                fill="none"
                stroke={hasBridgeWarning ? 'rgba(239,68,68,0.5)' : 'rgba(124,58,237,0.3)'}
                strokeWidth={hasBridgeWarning ? 2.5 : 2}
                strokeDasharray={hasBridgeWarning ? '6,4' : 'none'}
              />
            );
          })}
      </svg>

      {crates.map((c) => {
        const pos = positions[c.id];
        if (!pos) return null;
        const isSelected = selectedCrateId === c.id;
        const meta = c.metadata || {};
        const energy = meta.avg_energy ?? 0.5;

        return (
          <m.div
            key={c.id}
            onClick={() => onSelectCrate(c.id)}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.98 }}
            style={{
              position: 'absolute',
              left: pos.x,
              top: pos.y,
              width: NODE_W,
              height: NODE_H,
              background: 'var(--glass-bg)',
              backdropFilter: 'blur(20px)',
              WebkitBackdropFilter: 'blur(20px)',
              border: `1.5px solid ${isSelected ? 'rgba(168,85,247,0.7)' : 'var(--glass-border)'}`,
              borderRadius: 'var(--radius-md)',
              boxShadow: isSelected
                ? '0 0 20px rgba(124,58,237,0.3)'
                : 'var(--shadow-card)',
              cursor: 'pointer',
              padding: '10px 12px',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              overflow: 'visible',
            }}
          >
            <div
              style={{
                fontSize: 11,
                fontWeight: 700,
                color: isSelected ? '#c084fc' : '#e2e8f0',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}
            >
              {c.label || 'Crate'}
            </div>

            <div
              style={{
                display: 'flex',
                gap: 8,
                alignItems: 'center',
                fontSize: 9,
                fontFamily: "'JetBrains Mono', monospace",
                color: '#94a3b8',
              }}
            >
              <span>{meta.track_count || c.tracks?.length || 0} tracks</span>
              {meta.avg_bpm && <span style={{ color: '#00d4ff' }}>{meta.avg_bpm} BPM</span>}
            </div>

            <div
              style={{
                height: 3,
                background: 'rgba(124,58,237,0.15)',
                borderRadius: 'var(--radius-pill)',
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  height: '100%',
                  width: `${energy * 100}%`,
                  background: 'linear-gradient(90deg, #7c3aed, #00d4ff)',
                  borderRadius: 'var(--radius-pill)',
                  transition: 'width 300ms ease',
                }}
              />
            </div>

            {meta.dominant_tags?.length > 0 && (
              <div style={{ display: 'flex', gap: 3, overflow: 'hidden' }}>
                {meta.dominant_tags.slice(0, 2).map((tag) => (
                  <span
                    key={tag}
                    style={{
                      fontSize: 7,
                      color: 'rgba(168,85,247,0.6)',
                      background: 'rgba(124,58,237,0.08)',
                      borderRadius: 'var(--radius-pill)',
                      padding: '1px 5px',
                      fontFamily: "'JetBrains Mono', monospace",
                      letterSpacing: '0.03em',
                    }}
                  >
                    {tag}
                  </span>
                ))}
              </div>
            )}

            <m.button
              onClick={(e) => {
                e.stopPropagation();
                onAddBranch(c.id);
              }}
              whileHover={{ scale: 1.2, background: 'rgba(0,212,255,0.2)' }}
              whileTap={{ scale: 0.9 }}
              style={{
                position: 'absolute',
                right: -12,
                top: '50%',
                transform: 'translateY(-50%)',
                width: 24,
                height: 24,
                borderRadius: '50%',
                background: 'rgba(0,212,255,0.1)',
                border: '1px solid rgba(0,212,255,0.3)',
                color: '#00d4ff',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 14,
                fontWeight: 300,
              }}
            >
              +
            </m.button>
          </m.div>
        );
      })}

      {crates.length === 0 && (
        <div
          style={{
            padding: 40,
            textAlign: 'center',
            color: '#2d3748',
            fontSize: 10,
            fontFamily: "'JetBrains Mono', monospace",
            letterSpacing: '0.1em',
          }}
        >
          NO CRATES YET — USE THE PROMPT TO GENERATE ONE
        </div>
      )}
    </div>
  );
}
