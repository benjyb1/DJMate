// Direction metadata for the Next Track compass: colour, icon and where each
// direction sits around the track on air. Cool hues only, per the design system.

const stroke = { fill: 'none', stroke: 'currentColor', strokeWidth: 2, strokeLinecap: 'round', strokeLinejoin: 'round' };

const Svg = ({ size = 16, children }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true" {...stroke}>{children}</svg>
);

export const ICONS = {
  energy_up: (p) => <Svg {...p}><path d="M13 2 4 14h7l-1 8 10-13h-7z" /></Svg>,
  energy_down: (p) => <Svg {...p}><path d="M12 4v15M6 13l6 6 6-6" /></Svg>,
  darker: (p) => <Svg {...p}><path d="M20.5 13.5A8.5 8.5 0 1 1 10.5 3.5a6.5 6.5 0 0 0 10 10z" /></Svg>,
  lighter: (p) => <Svg {...p}><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></Svg>,
  faster: (p) => <Svg {...p}><path d="M5 6l6 6-6 6M13 6l6 6-6 6" /></Svg>,
  slower: (p) => <Svg {...p}><path d="M19 6l-6 6 6 6M11 6l-6 6 6 6" /></Svg>,
  electronic: (p) => <Svg {...p}><path d="M2 12h3V6h4v12h4V6h4v12h3v-6h2" /></Svg>,
  organic: (p) => <Svg {...p}><path d="M11 20A7 7 0 0 1 4 13c0-6 7-10 16-10 0 9-4 16-9 17z" /><path d="M4 21c4-4 7-7 10-10" /></Svg>,
  vocal: (p) => <Svg {...p}><rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3" /></Svg>,
  instrumental: (p) => <Svg {...p}><path d="M3 12h2l2-5 3 10 3-14 3 12 2-3h3" /></Svg>,
  closest: (p) => <Svg {...p}><circle cx="12" cy="12" r="9" /><circle cx="12" cy="12" r="5" /><circle cx="12" cy="12" r="1.2" /></Svg>,
};

// grid row/column on the 3x3 compass; the centre cell is the track on air.
export const DIRECTION_META = {
  energy_up:    { color: '#a855f7', row: 1, col: 2, blurb: 'Lift the room' },
  darker:       { color: '#818cf8', row: 1, col: 3, blurb: 'Moodier, minor, heavier' },
  faster:       { color: '#00d4ff', row: 2, col: 3, blurb: 'Push the tempo' },
  electronic:   { color: '#2dd4bf', row: 3, col: 3, blurb: 'More machines' },
  energy_down:  { color: '#60a5fa', row: 3, col: 2, blurb: 'Let it breathe' },
  closest:      { color: '#cbd5e1', row: 3, col: 1, blurb: 'The safest blend' },
  slower:       { color: '#38bdf8', row: 2, col: 1, blurb: 'Ease the tempo' },
  vocal:        { color: '#c084fc', row: 1, col: 1, blurb: 'Bring in a voice' },
  lighter:      { color: '#e0e7ff', row: 1, col: 3, blurb: 'Brighter, happier' },
  instrumental: { color: '#94a3b8', row: 1, col: 1, blurb: 'Strip the vocals' },
  organic:      { color: '#5eead4', row: 3, col: 3, blurb: 'Real instruments' },
};

export const SOURCE_LABEL = {
  rekordbox: 'Live from rekordbox',
  explore: 'Exploring',
  mic: 'Heard on the mic',
  file: 'Dropped file',
  search: 'Picked by hand',
};
