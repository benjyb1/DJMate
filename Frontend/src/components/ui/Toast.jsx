import { useEffect } from 'react';
import { m, AnimatePresence } from 'framer-motion';

export default function Toast({ message, visible, onDismiss, duration = 4000 }) {
  useEffect(() => {
    if (visible && onDismiss) {
      const t = setTimeout(onDismiss, duration);
      return () => clearTimeout(t);
    }
  }, [visible, onDismiss, duration]);

  return (
    <AnimatePresence>
      {visible && (
        <m.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 30 }}
          transition={{ type: 'spring', damping: 25, stiffness: 300 }}
          style={{
            position: 'fixed',
            bottom: 32,
            left: '50%',
            transform: 'translateX(-50%)',
            background: 'rgba(8,8,20,0.85)',
            backdropFilter: 'blur(16px)',
            WebkitBackdropFilter: 'blur(16px)',
            border: '1px solid rgba(124,58,237,0.2)',
            borderRadius: 8,
            padding: '10px 20px',
            color: '#e2e8f0',
            fontFamily: "'Inter', system-ui, sans-serif",
            fontSize: 13,
            boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
            zIndex: 9999,
            pointerEvents: 'none',
            whiteSpace: 'nowrap',
          }}
        >
          {message}
        </m.div>
      )}
    </AnimatePresence>
  );
}
