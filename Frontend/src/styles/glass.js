/**
 * Shared glassmorphism style objects — sourced from index.css design tokens.
 *
 * CSS variable references:
 *   --glass-bg: rgba(8, 8, 20, 0.72)
 *   --glass-blur: 24px
 *   --glass-border: rgba(124, 58, 237, 0.15)
 *   --radius-md: 12px, --radius-lg: 16px, --radius-xl: 20px, --radius-pill: 9999px
 *   --shadow-float, --shadow-panel
 */

export const glassPanel = {
  background: 'var(--glass-bg)',
  backdropFilter: 'blur(24px)',
  WebkitBackdropFilter: 'blur(24px)',
  border: '1px solid var(--glass-border)',
  borderRadius: 'var(--radius-lg)',
};

export const glassCard = {
  ...glassPanel,
  backdropFilter: 'blur(16px)',
  WebkitBackdropFilter: 'blur(16px)',
  borderRadius: 'var(--radius-md)',
};

export const glassPill = {
  ...glassPanel,
  backdropFilter: 'blur(20px)',
  WebkitBackdropFilter: 'blur(20px)',
  borderRadius: 'var(--radius-pill)',
};

export const glassFloat = {
  ...glassPanel,
  backdropFilter: 'blur(28px)',
  WebkitBackdropFilter: 'blur(28px)',
  borderRadius: 'var(--radius-xl)',
  boxShadow: 'var(--shadow-float)',
};
