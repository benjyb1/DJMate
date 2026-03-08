import React from 'react';
import { m, AnimatePresence } from 'framer-motion';

/**
 * Reusable glassmorphism container with framer-motion entry/exit.
 * depth: 1 (subtle), 2 (default), 3 (prominent)
 */
export default function GlassPanel({
  children,
  style = {},
  depth = 2,
  animate = true,
  className,
  onClick,
  initial,
  animateProps,
  exit,
  transition,
  layoutId,
  ...rest
}) {
  const shadows = {
    1: 'var(--shadow-card)',
    2: 'var(--shadow-panel)',
    3: 'var(--shadow-float)',
  };

  const blurs = {
    1: 16,
    2: 24,
    3: 32,
  };

  const panelStyle = {
    background: 'var(--glass-bg)',
    backdropFilter: `blur(${blurs[depth]}px)`,
    WebkitBackdropFilter: `blur(${blurs[depth]}px)`,
    border: '1px solid var(--glass-border)',
    borderRadius: 'var(--radius-lg)',
    boxShadow: shadows[depth],
    ...style,
  };

  if (!animate) {
    return (
      <div style={panelStyle} className={className} onClick={onClick} {...rest}>
        {children}
      </div>
    );
  }

  return (
    <m.div
      style={panelStyle}
      className={className}
      onClick={onClick}
      layoutId={layoutId}
      initial={initial || { opacity: 0, y: 12, scale: 0.97 }}
      animate={animateProps || { opacity: 1, y: 0, scale: 1 }}
      exit={exit || { opacity: 0, y: 12, scale: 0.97 }}
      transition={transition || { type: 'spring', damping: 28, stiffness: 320 }}
      {...rest}
    >
      {children}
    </m.div>
  );
}
