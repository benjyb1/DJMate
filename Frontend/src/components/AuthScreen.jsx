import { useState } from 'react';
import { m, AnimatePresence } from 'framer-motion';
import { useAuthStore } from '../stores/authStore';
import { IconWaveform } from './icons';

// ── Shared styles ────────────────────────────────────────────────────────────

const inputStyle = {
  width: '100%',
  padding: '12px 14px',
  background: 'rgba(255,255,255,0.06)',
  border: '1px solid rgba(124,58,237,0.15)',
  borderRadius: 8,
  color: '#e2e8f0',
  fontSize: 14,
  fontFamily: "'Inter', system-ui, sans-serif",
  outline: 'none',
  transition: 'border-color 0.2s, box-shadow 0.2s',
  boxSizing: 'border-box',
};

const inputFocusStyle = {
  borderColor: 'rgba(124,58,237,0.5)',
  boxShadow: '0 0 0 3px rgba(124,58,237,0.15)',
};

const labelStyle = {
  display: 'block',
  fontSize: 12,
  fontWeight: 500,
  color: '#64748b',
  marginBottom: 6,
  fontFamily: "'Inter', system-ui, sans-serif",
};

// ── Spinner SVG ──────────────────────────────────────────────────────────────

function Spinner() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" style={{ animation: 'auth-spin 0.8s linear infinite' }}>
      <circle cx="12" cy="12" r="10" stroke="rgba(255,255,255,0.25)" strokeWidth="3" />
      <path d="M12 2a10 10 0 0 1 10 10" stroke="#fff" strokeWidth="3" strokeLinecap="round" />
      <style>{`@keyframes auth-spin { to { transform: rotate(360deg); } }`}</style>
    </svg>
  );
}

// ── Focusable input wrapper ──────────────────────────────────────────────────

function Input({ label, ...props }) {
  const [focused, setFocused] = useState(false);
  return (
    <div style={{ marginBottom: 16 }}>
      {label && <label style={labelStyle}>{label}</label>}
      <input
        {...props}
        style={{
          ...inputStyle,
          ...(focused ? inputFocusStyle : {}),
        }}
        onFocus={(e) => { setFocused(true); props.onFocus?.(e); }}
        onBlur={(e) => { setFocused(false); props.onBlur?.(e); }}
      />
    </div>
  );
}

// ── Animation variants ───────────────────────────────────────────────────────

const formVariants = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.25, ease: 'easeOut' } },
  exit: { opacity: 0, y: -12, transition: { duration: 0.15, ease: 'easeIn' } },
};

// ── AuthScreen ───────────────────────────────────────────────────────────────

export default function AuthScreen() {
  const [mode, setMode] = useState('signin');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [username, setUsername] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const { signIn, signUp } = useAuthStore();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      if (mode === 'signup') {
        await signUp(email, password, username);
      } else {
        await signIn(email, password);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const toggleMode = () => {
    setError('');
    setMode(m => (m === 'signin' ? 'signup' : 'signin'));
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: '#0a0a14',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: 24,
      fontFamily: "'Inter', system-ui, sans-serif",
    }}>
      <m.div
        initial={{ opacity: 0, scale: 0.96 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.35, ease: 'easeOut' }}
        style={{
          width: '100%',
          maxWidth: 400,
          background: 'rgba(8,8,20,0.72)',
          backdropFilter: 'blur(24px)',
          WebkitBackdropFilter: 'blur(24px)',
          border: '1px solid rgba(124,58,237,0.15)',
          borderRadius: 12,
          padding: '36px 32px 32px',
        }}
      >
        {/* ── Branding ─────────────────────────────────────── */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 12,
          marginBottom: 32,
        }}>
          <div style={{
            width: 40, height: 40, borderRadius: 10,
            background: 'linear-gradient(135deg, rgba(124,58,237,0.25), rgba(0,212,255,0.15))',
            border: '1px solid rgba(124,58,237,0.3)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <IconWaveform size={22} style={{ color: '#a855f7' }} />
          </div>
          <span style={{ fontSize: 20, fontWeight: 800, letterSpacing: '0.12em' }}>
            <span style={{ color: '#e2e8f0' }}>DJ</span>
            <span style={{ color: '#a855f7' }}>MATE</span>
          </span>
        </div>

        {/* ── Form ─────────────────────────────────────────── */}
        <AnimatePresence mode="wait">
          <m.form
            key={mode}
            variants={formVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            onSubmit={handleSubmit}
          >
            {mode === 'signup' && (
              <Input
                label="Username"
                type="text"
                placeholder="your_dj_name"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoComplete="username"
                required
              />
            )}

            <Input
              label="Email"
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="email"
              required
            />

            <Input
              label="Password"
              type="password"
              placeholder={mode === 'signup' ? 'Min 6 characters' : 'Your password'}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete={mode === 'signup' ? 'new-password' : 'current-password'}
              minLength={6}
              required
            />

            {/* ── Error ──────────────────────────────────────── */}
            {error && (
              <m.p
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                style={{
                  color: '#f87171',
                  fontSize: 13,
                  margin: '0 0 16px',
                  lineHeight: 1.4,
                }}
              >
                {error}
              </m.p>
            )}

            {/* ── Submit ─────────────────────────────────────── */}
            <m.button
              type="submit"
              disabled={loading}
              whileHover={loading ? {} : { scale: 1.02 }}
              whileTap={loading ? {} : { scale: 0.97 }}
              style={{
                width: '100%',
                padding: '12px 0',
                border: 'none',
                borderRadius: 8,
                background: loading
                  ? 'rgba(124,58,237,0.4)'
                  : 'linear-gradient(135deg, #7c3aed, #a855f7)',
                color: '#fff',
                fontSize: 14,
                fontWeight: 600,
                fontFamily: "'Inter', system-ui, sans-serif",
                cursor: loading ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 8,
                transition: 'background 0.2s',
              }}
            >
              {loading && <Spinner />}
              {mode === 'signup' ? 'Create Account' : 'Sign In'}
            </m.button>
          </m.form>
        </AnimatePresence>

        {/* ── Toggle link ──────────────────────────────────── */}
        <p style={{
          textAlign: 'center',
          marginTop: 20,
          fontSize: 13,
          color: '#64748b',
        }}>
          {mode === 'signin' ? "Don't have an account? " : 'Already have an account? '}
          <span
            role="button"
            tabIndex={0}
            onClick={toggleMode}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') toggleMode(); }}
            style={{
              color: '#a855f7',
              cursor: 'pointer',
              fontWeight: 500,
            }}
          >
            {mode === 'signin' ? 'Sign up' : 'Sign in'}
          </span>
        </p>
      </m.div>
    </div>
  );
}
