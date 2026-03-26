import { useState, useEffect, useCallback } from 'react';
import { m, AnimatePresence } from 'framer-motion';
import { createClient } from '@supabase/supabase-js';

async function testConnection(url, key) {
  const client = createClient(url, key);
  const { data, error } = await client.from('tracks').select('id').limit(1);
  if (error) throw new Error(error.message);
  return true;
}

export default function SupabaseLinkModal({
  open,
  onClose,
  onLink,
  defaultUrl = '',
  defaultKey = '',
}) {
  const [url, setUrl] = useState(defaultUrl);
  const [key, setKey] = useState(defaultKey);
  const [testing, setTesting] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    if (open) {
      setUrl(defaultUrl);
      setKey(defaultKey);
      setError('');
      setSuccess(false);
    }
  }, [open, defaultUrl, defaultKey]);

  const handleTest = useCallback(async () => {
    if (!url.trim() || !key.trim()) {
      setError('Both fields are required.');
      return;
    }
    setTesting(true);
    setError('');
    setSuccess(false);
    try {
      await testConnection(url.trim(), key.trim());
      setSuccess(true);
      await onLink(url.trim(), key.trim());
    } catch (e) {
      setError(e.message || 'Connection failed.');
    } finally {
      setTesting(false);
    }
  }, [url, key, onLink]);

  const handleOverlayClick = useCallback(
    (e) => {
      if (e.target === e.currentTarget) onClose();
    },
    [onClose],
  );

  return (
    <AnimatePresence>
      {open && (
        <m.div
          key="supabase-link-overlay"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          onClick={handleOverlayClick}
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.6)',
            backdropFilter: 'blur(4px)',
            WebkitBackdropFilter: 'blur(4px)',
            zIndex: 1000,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <m.div
            key="supabase-link-panel"
            initial={{ opacity: 0, y: 20, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.96 }}
            transition={{ type: 'spring', damping: 28, stiffness: 320 }}
            style={{
              position: 'relative',
              width: '100%',
              maxWidth: 440,
              margin: '0 16px',
              background: 'rgba(8,8,20,0.9)',
              backdropFilter: 'blur(24px)',
              WebkitBackdropFilter: 'blur(24px)',
              border: '1px solid rgba(124,58,237,0.2)',
              borderRadius: 16,
              padding: '28px 24px 24px',
              boxShadow:
                '0 8px 40px rgba(0,0,0,0.5), 0 0 60px rgba(124,58,237,0.08)',
            }}
          >
            {/* Close button */}
            <button
              onClick={onClose}
              style={{
                position: 'absolute',
                top: 14,
                right: 14,
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                padding: 4,
                color: '#64748b',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
              aria-label="Close"
            >
              <svg
                width="18"
                height="18"
                viewBox="0 0 18 18"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
              >
                <line x1="4" y1="4" x2="14" y2="14" />
                <line x1="14" y1="4" x2="4" y2="14" />
              </svg>
            </button>

            {/* Title */}
            <div
              style={{
                fontSize: 16,
                fontWeight: 700,
                color: '#e2e8f0',
                fontFamily: "'Inter', system-ui, sans-serif",
                marginBottom: 20,
              }}
            >
              Link Your Supabase
            </div>

            {/* URL input */}
            <label
              style={{
                display: 'block',
                fontSize: 11,
                fontWeight: 500,
                color: '#94a3b8',
                fontFamily: "'Inter', system-ui, sans-serif",
                marginBottom: 6,
              }}
            >
              Project URL
            </label>
            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://xyzcompany.supabase.co"
              spellCheck={false}
              autoComplete="off"
              style={{
                width: '100%',
                boxSizing: 'border-box',
                background: 'rgba(255,255,255,0.06)',
                border: '1px solid rgba(124,58,237,0.15)',
                borderRadius: 8,
                padding: '10px 12px',
                color: '#e2e8f0',
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 12,
                outline: 'none',
                marginBottom: 16,
                transition: 'border-color 0.15s',
              }}
              onFocus={(e) =>
                (e.target.style.borderColor = 'rgba(124,58,237,0.4)')
              }
              onBlur={(e) =>
                (e.target.style.borderColor = 'rgba(124,58,237,0.15)')
              }
            />

            {/* Key input */}
            <label
              style={{
                display: 'block',
                fontSize: 11,
                fontWeight: 500,
                color: '#94a3b8',
                fontFamily: "'Inter', system-ui, sans-serif",
                marginBottom: 6,
              }}
            >
              Anon Key
            </label>
            <input
              type="text"
              value={key}
              onChange={(e) => setKey(e.target.value)}
              placeholder="eyJhbGciOiJIUzI1NiIs..."
              spellCheck={false}
              autoComplete="off"
              style={{
                width: '100%',
                boxSizing: 'border-box',
                background: 'rgba(255,255,255,0.06)',
                border: '1px solid rgba(124,58,237,0.15)',
                borderRadius: 8,
                padding: '10px 12px',
                color: '#e2e8f0',
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 12,
                outline: 'none',
                marginBottom: 20,
                transition: 'border-color 0.15s',
              }}
              onFocus={(e) =>
                (e.target.style.borderColor = 'rgba(124,58,237,0.4)')
              }
              onBlur={(e) =>
                (e.target.style.borderColor = 'rgba(124,58,237,0.15)')
              }
            />

            {/* Error / success message */}
            {error && (
              <div
                style={{
                  fontSize: 12,
                  color: '#f87171',
                  fontFamily: "'Inter', system-ui, sans-serif",
                  marginBottom: 14,
                  lineHeight: 1.4,
                }}
              >
                {error}
              </div>
            )}
            {success && (
              <div
                style={{
                  fontSize: 12,
                  color: '#34d399',
                  fontFamily: "'Inter', system-ui, sans-serif",
                  marginBottom: 14,
                }}
              >
                Connected successfully.
              </div>
            )}

            {/* Test + Link button */}
            <m.button
              onClick={handleTest}
              disabled={testing}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.97 }}
              style={{
                width: '100%',
                padding: '10px 0',
                background:
                  'linear-gradient(135deg, rgba(124,58,237,0.3), rgba(168,85,247,0.2))',
                border: '1px solid rgba(124,58,237,0.3)',
                borderRadius: 8,
                color: '#e2e8f0',
                fontFamily: "'Inter', system-ui, sans-serif",
                fontSize: 13,
                fontWeight: 600,
                cursor: testing ? 'wait' : 'pointer',
                opacity: testing ? 0.6 : 1,
                transition: 'opacity 0.15s',
              }}
            >
              {testing ? 'Testing connection...' : 'Test Connection & Link'}
            </m.button>
          </m.div>
        </m.div>
      )}
    </AnimatePresence>
  );
}
