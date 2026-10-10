// The Needle shell: no sign-in, just the compass.
import React from 'react';
import NextTrack from '../NextTrack';

// In the desktop app the window has no title bar: the traffic-light buttons
// sit over the top-left corner, and the header is what you drag the window by.
const IN_ELECTRON = typeof navigator !== 'undefined' && /Electron/i.test(navigator.userAgent);

export default function NextTrackStandalone() {
  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'var(--bg-1)', color: 'var(--text-primary)',
      fontFamily: 'var(--font-ui)', overflow: 'hidden',
    }}>
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: 64, zIndex: 50,
        display: 'flex', alignItems: 'center', gap: 10,
        padding: IN_ELECTRON ? '0 24px 0 88px' : '0 24px',
        WebkitAppRegion: IN_ELECTRON ? 'drag' : undefined,
      }}>
        <img src="/needle.svg" alt="" width={30} height={30} draggable={false}
          style={{ display: 'block', margin: '0 -2px' }} />
        <span style={{ fontSize: 15, fontWeight: 800, letterSpacing: '0.2em', color: '#e2e8f0' }}>
          NEEDLE
        </span>
      </div>
      <NextTrack topInset={64} />
    </div>
  );
}
