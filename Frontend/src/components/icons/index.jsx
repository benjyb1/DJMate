import React from 'react';

// ── Play / Pause ──────────────────────────────────────────────────────────────

/** Play icon (filled triangle). App.jsx uses size=14, DJChatbox uses size=10, LiveMode uses size=11. */
export const IconPlay = ({ size = 14, ...props }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" {...props}>
    <polygon points="5,3 19,12 5,21" />
  </svg>
);

/** Pause icon (two bars). App.jsx uses size=14, DJChatbox uses size=10, LiveMode uses size=11. */
export const IconPause = ({ size = 14, ...props }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" {...props}>
    <rect x="6" y="4" width="4" height="16" rx="1" /><rect x="14" y="4" width="4" height="16" rx="1" />
  </svg>
);

// ── Search / Similar ──────────────────────────────────────────────────────────

/** Search magnifying glass. App.jsx uses size=13, DJChatbox uses size=11, LiveMode uses size=11. */
export const IconSearch = ({ size = 13, ...props }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true" {...props}>
    <circle cx="11" cy="11" r="7" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>
);

// ── Close / X ─────────────────────────────────────────────────────────────────

/** Close X icon. DJChatbox uses size=12, LiveMode uses size=11. */
export const IconClose = ({ size = 12, ...props }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true" {...props}>
    <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
  </svg>
);

// ── Send ──────────────────────────────────────────────────────────────────────

/** Send arrow icon. DJChatbox uses size=14, LiveMode uses size=13. */
export const IconSend = ({ size = 14, ...props }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>
    <line x1="22" y1="2" x2="11" y2="13" /><polygon points="22,2 15,22 11,13 2,9" />
  </svg>
);

// ── Edit (pencil) ─────────────────────────────────────────────────────────────

/** Pencil edit icon. DJChatbox & LiveMode use size=10. */
export const IconEdit = ({ size = 10, ...props }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>
    <path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z" />
  </svg>
);

// ── Chevrons ──────────────────────────────────────────────────────────────────

/** Chevron up. DJChatbox uses size=12. */
export const IconUp = ({ size = 12, ...props }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true" {...props}>
    <polyline points="18,15 12,9 6,15" />
  </svg>
);

/** Chevron down. DJChatbox uses size=12. */
export const IconDown = ({ size = 12, ...props }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true" {...props}>
    <polyline points="6,9 12,15 18,9" />
  </svg>
);

// ── Plus ──────────────────────────────────────────────────────────────────────

/** Plus icon. LiveMode uses size=13. */
export const IconPlus = ({ size = 13, ...props }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden="true" {...props}>
    <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
  </svg>
);

// ── Waveform ──────────────────────────────────────────────────────────────────

/** Waveform / soundbar icon. App.jsx uses size=16. */
export const IconWaveform = ({ size = 16, ...props }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true" {...props}>
    <line x1="12" y1="2" x2="12" y2="22" /><line x1="8" y1="6" x2="8" y2="18" />
    <line x1="16" y1="6" x2="16" y2="18" /><line x1="4" y1="10" x2="4" y2="14" /><line x1="20" y1="10" x2="20" y2="14" />
  </svg>
);

// ── Mic ───────────────────────────────────────────────────────────────────────

/** Microphone icon. LiveMode uses size=16. */
export const IconMic = ({ size = 16, ...props }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true" {...props}>
    <rect x="9" y="2" width="6" height="11" rx="3" />
    <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
    <line x1="12" y1="19" x2="12" y2="23" />
    <line x1="8" y1="23" x2="16" y2="23" />
  </svg>
);

// ── Vector / Arrow right ──────────────────────────────────────────────────────

/** Right arrow / vector icon. LiveMode uses size=14. */
export const IconVector = ({ size = 14, ...props }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true" {...props}>
    <path d="M5 12h14" /><polyline points="12,5 19,12 12,19" />
  </svg>
);
