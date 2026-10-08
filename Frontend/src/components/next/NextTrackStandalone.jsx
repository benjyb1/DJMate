// Next Track without the rest of DJMate: no sign-in, just the compass.
// Reached via #next (and from the sign-in screen).
import React from 'react';
import NextTrack from '../NextTrack';
import { IconWaveform } from '../icons';

export default function NextTrackStandalone() {
  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'var(--bg-1)', color: 'var(--text-primary)',
      fontFamily: 'var(--font-ui)', overflow: 'hidden',
    }}>
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: 64, zIndex: 50,
        display: 'flex', alignItems: 'center', gap: 10, padding: '0 24px',
      }}>
        <div style={{
          width: 26, height: 26, borderRadius: 'var(--radius-sm)',
          background: 'linear-gradient(135deg, rgba(124,58,237,0.25), rgba(0,212,255,0.15))',
          border: '1px solid rgba(124,58,237,0.3)', display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}><IconWaveform /></div>
        <span style={{ fontSize: 14, fontWeight: 800, letterSpacing: '0.12em' }}>
          <span style={{ color: '#e2e8f0' }}>DJ</span><span style={{ color: '#a855f7' }}>MATE</span>
        </span>
        <span style={{ width: 1, height: 20, background: 'var(--border-panel)', margin: '0 6px' }} />
        <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.16em', color: 'var(--text-secondary)' }}>NEXT TRACK</span>
        <div style={{ flex: 1 }} />
        <a href="#" style={{ fontSize: 11.5, color: 'var(--text-muted)', textDecoration: 'none' }}>Full DJMate</a>
      </div>
      <NextTrack topInset={64} />
    </div>
  );
}
