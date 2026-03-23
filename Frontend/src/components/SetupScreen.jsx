import React, { useState } from 'react';
import { m, AnimatePresence } from 'framer-motion';
import { createClient } from '@supabase/supabase-js';
import { saveCredentials } from '../stores/credentialStore';
import { isFileSystemAccessSupported, pickMusicFolder } from '../utils/localAudio';
import GlassPanel from './ui/GlassPanel';

const spring = { type: 'spring', damping: 28, stiffness: 320 };

function StepIndicator({ step, total }) {
  return (
    <div style={{ display: 'flex', gap: 6, justifyContent: 'center', marginBottom: 32 }}>
      {Array.from({ length: total }, (_, i) => (
        <div
          key={i}
          style={{
            width: i === step ? 24 : 8,
            height: 8,
            borderRadius: 'var(--radius-pill)',
            background: i === step
              ? 'var(--gradient-accent)'
              : i < step
                ? 'var(--purple)'
                : 'rgba(124,58,237,0.2)',
            transition: 'all 300ms ease',
          }}
        />
      ))}
    </div>
  );
}

function InputField({ label, value, onChange, placeholder, type = 'text', mono = false }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <label style={{
        display: 'block',
        fontSize: 12,
        fontWeight: 500,
        color: 'var(--text-secondary)',
        marginBottom: 6,
        letterSpacing: '0.03em',
      }}>
        {label}
      </label>
      <input
        type={type}
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
        style={{
          width: '100%',
          padding: '10px 14px',
          background: 'rgba(8,8,20,0.6)',
          border: '1px solid var(--glass-border)',
          borderRadius: 'var(--radius-sm)',
          color: 'var(--text-primary)',
          fontSize: 13,
          fontFamily: mono ? 'var(--font-mono)' : 'var(--font-ui)',
          outline: 'none',
          transition: 'border-color 200ms ease',
          boxSizing: 'border-box',
        }}
        onFocus={e => e.target.style.borderColor = 'var(--purple)'}
        onBlur={e => e.target.style.borderColor = 'var(--glass-border)'}
      />
    </div>
  );
}

function ActionButton({ onClick, disabled, loading, children, variant = 'primary' }) {
  const isPrimary = variant === 'primary';
  return (
    <m.button
      onClick={onClick}
      disabled={disabled || loading}
      whileHover={!disabled ? { scale: 1.02 } : {}}
      whileTap={!disabled ? { scale: 0.97 } : {}}
      style={{
        width: '100%',
        padding: '12px 24px',
        background: isPrimary
          ? 'var(--gradient-accent)'
          : 'rgba(124,58,237,0.15)',
        border: isPrimary
          ? 'none'
          : '1px solid var(--glass-border)',
        borderRadius: 'var(--radius-sm)',
        color: 'var(--text-primary)',
        fontSize: 14,
        fontWeight: 600,
        fontFamily: 'var(--font-ui)',
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.5 : 1,
        transition: 'opacity 200ms ease',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
      }}
    >
      {loading && (
        <m.div
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
          style={{
            width: 14, height: 14,
            border: '2px solid rgba(255,255,255,0.3)',
            borderTopColor: 'white',
            borderRadius: '50%',
          }}
        />
      )}
      {children}
    </m.button>
  );
}

export default function SetupScreen({ onConnected }) {
  const [step, setStep] = useState(0);

  // Step 1 state: Supabase creds
  const [supabaseUrl, setSupabaseUrl] = useState('');
  const [supabaseKey, setSupabaseKey] = useState('');
  const [testing, setTesting] = useState(false);
  const [error, setError] = useState(null);

  // Step 2 state: Music folder
  const [folderName, setFolderName] = useState(null);
  const fsSupported = isFileSystemAccessSupported();

  async function handleConnect() {
    setTesting(true);
    setError(null);

    try {
      const url = supabaseUrl.trim();
      const key = supabaseKey.trim();

      if (!url.includes('supabase')) {
        setError('That doesn\'t look like a Supabase URL. It should be something like https://abcdefg.supabase.co');
        setTesting(false);
        return;
      }

      // Test the connection by querying the tracks table
      const client = createClient(url, key);
      const { data, error: sbError } = await client
        .from('tracks')
        .select('trackid', { count: 'exact', head: true });

      if (sbError) {
        setError(`Connection failed: ${sbError.message}`);
        setTesting(false);
        return;
      }

      // Save creds and move to next step
      saveCredentials(url, key);
      setStep(1);
    } catch (err) {
      setError(`Could not connect: ${err.message}`);
    } finally {
      setTesting(false);
    }
  }

  async function handlePickFolder() {
    try {
      const handle = await pickMusicFolder();
      setFolderName(handle.name);
    } catch (err) {
      // User cancelled the picker — that's fine
      if (err.name !== 'AbortError') {
        console.error('Folder picker error:', err);
      }
    }
  }

  function handleFinish() {
    onConnected();
  }

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      background: 'var(--bg-0)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 9999,
      fontFamily: 'var(--font-ui)',
    }}>
      {/* Subtle gradient backdrop */}
      <div style={{
        position: 'absolute',
        inset: 0,
        background: 'radial-gradient(ellipse at 50% 30%, rgba(124,58,237,0.12) 0%, transparent 60%)',
        pointerEvents: 'none',
      }} />

      <GlassPanel
        depth={3}
        style={{
          width: '100%',
          maxWidth: 440,
          padding: 36,
          margin: 16,
          position: 'relative',
        }}
      >
        {/* Logo / title */}
        <div style={{ textAlign: 'center', marginBottom: 8 }}>
          <h1 style={{
            fontSize: 28,
            fontWeight: 700,
            color: 'var(--text-primary)',
            margin: 0,
            letterSpacing: '-0.02em',
          }}>
            DJMate
          </h1>
          <p style={{
            fontSize: 13,
            color: 'var(--text-secondary)',
            margin: '6px 0 0',
          }}>
            Connect your music library
          </p>
        </div>

        <StepIndicator step={step} total={2} />

        <AnimatePresence mode="wait">
          {step === 0 && (
            <m.div
              key="step-0"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={spring}
            >
              <p style={{
                fontSize: 13,
                color: 'var(--text-secondary)',
                lineHeight: 1.5,
                marginBottom: 20,
              }}>
                Enter your Supabase project credentials. You can find these in your
                Supabase dashboard under <span style={{ color: 'var(--cyan)', fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                  Project Settings &gt; API
                </span>.
              </p>

              <InputField
                label="Supabase URL"
                value={supabaseUrl}
                onChange={setSupabaseUrl}
                placeholder="https://your-project.supabase.co"
                mono
              />

              <InputField
                label="Anon / Public Key"
                value={supabaseKey}
                onChange={setSupabaseKey}
                placeholder="eyJhbGciOiJIUzI1NiIs..."
                mono
              />

              {error && (
                <m.div
                  initial={{ opacity: 0, y: -4 }}
                  animate={{ opacity: 1, y: 0 }}
                  style={{
                    padding: '10px 14px',
                    background: 'rgba(239,68,68,0.12)',
                    border: '1px solid rgba(239,68,68,0.3)',
                    borderRadius: 'var(--radius-sm)',
                    color: '#fca5a5',
                    fontSize: 12,
                    marginBottom: 16,
                    lineHeight: 1.4,
                  }}
                >
                  {error}
                </m.div>
              )}

              <ActionButton
                onClick={handleConnect}
                disabled={!supabaseUrl.trim() || !supabaseKey.trim()}
                loading={testing}
              >
                Connect
              </ActionButton>

              <a
                href="https://github.com/benjyb1/DJMate/blob/main/SETUP.md"
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  display: 'block',
                  textAlign: 'center',
                  marginTop: 16,
                  fontSize: 12,
                  color: 'var(--text-muted)',
                  textDecoration: 'none',
                }}
              >
                Need help setting up? Read the setup guide
              </a>
            </m.div>
          )}

          {step === 1 && (
            <m.div
              key="step-1"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={spring}
            >
              <p style={{
                fontSize: 13,
                color: 'var(--text-secondary)',
                lineHeight: 1.5,
                marginBottom: 20,
              }}>
                Grant access to your music folder so DJMate can play your tracks
                directly from your computer. Your music stays on your machine.
              </p>

              {fsSupported ? (
                <>
                  <ActionButton
                    onClick={handlePickFolder}
                    variant={folderName ? 'secondary' : 'primary'}
                  >
                    {folderName ? `Selected: ${folderName}` : 'Select Music Folder'}
                  </ActionButton>

                  {folderName && (
                    <m.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      style={{
                        marginTop: 12,
                        padding: '10px 14px',
                        background: 'rgba(0,212,255,0.08)',
                        border: '1px solid rgba(0,212,255,0.2)',
                        borderRadius: 'var(--radius-sm)',
                        color: 'var(--cyan)',
                        fontSize: 12,
                        lineHeight: 1.4,
                      }}
                    >
                      Chrome will ask to confirm access when you return. One click to re-grant.
                    </m.div>
                  )}
                </>
              ) : (
                <div style={{
                  padding: '10px 14px',
                  background: 'rgba(245,158,11,0.1)',
                  border: '1px solid rgba(245,158,11,0.25)',
                  borderRadius: 'var(--radius-sm)',
                  color: '#fbbf24',
                  fontSize: 12,
                  lineHeight: 1.4,
                  marginBottom: 16,
                }}>
                  Audio playback from local files needs Chrome, Edge, or Brave.
                  You can still browse and search your library without it.
                </div>
              )}

              <div style={{ marginTop: 20, display: 'flex', gap: 10 }}>
                <m.button
                  onClick={() => setStep(0)}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.97 }}
                  style={{
                    flex: '0 0 auto',
                    padding: '12px 20px',
                    background: 'transparent',
                    border: '1px solid var(--glass-border)',
                    borderRadius: 'var(--radius-sm)',
                    color: 'var(--text-secondary)',
                    fontSize: 13,
                    fontFamily: 'var(--font-ui)',
                    cursor: 'pointer',
                  }}
                >
                  Back
                </m.button>
                <div style={{ flex: 1 }}>
                  <ActionButton onClick={handleFinish}>
                    {folderName ? 'Start Exploring' : 'Skip & Continue'}
                  </ActionButton>
                </div>
              </div>
            </m.div>
          )}
        </AnimatePresence>
      </GlassPanel>
    </div>
  );
}
