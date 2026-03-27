import { useState } from 'react';
import { m } from 'framer-motion';
import { useAuthStore } from '../stores/authStore';
import { syncFromProfile } from '../stores/credentialStore';
import { isFileSystemAccessSupported, pickMusicFolder } from '../utils/localAudio';
import { apiClient } from '../api/apiClient';
import SupabaseLinkModal from './ui/SupabaseLinkModal';

const labelStyle = {
  fontSize: 10,
  letterSpacing: '0.1em',
  color: '#64748b',
  fontFamily: 'JetBrains Mono, monospace',
  textTransform: 'uppercase',
  margin: 0,
};

const valueStyle = {
  fontSize: 14,
  color: '#e2e8f0',
  fontFamily: 'Inter, sans-serif',
  margin: 0,
};

const panelStyle = {
  background: 'rgba(8,8,20,0.72)',
  backdropFilter: 'blur(24px)',
  WebkitBackdropFilter: 'blur(24px)',
  border: '1px solid rgba(124,58,237,0.15)',
  borderRadius: 12,
  padding: 24,
};

const primaryBtnStyle = {
  background: 'linear-gradient(135deg, #7c3aed, #a855f7)',
  border: '1px solid rgba(124,58,237,0.4)',
  color: '#fff',
  borderRadius: 8,
  padding: '8px 18px',
  fontSize: 13,
  fontFamily: 'Inter, sans-serif',
  cursor: 'pointer',
  fontWeight: 500,
};

const secondaryBtnStyle = {
  background: 'transparent',
  border: '1px solid rgba(124,58,237,0.15)',
  color: '#e2e8f0',
  borderRadius: 8,
  padding: '8px 18px',
  fontSize: 13,
  fontFamily: 'Inter, sans-serif',
  cursor: 'pointer',
  fontWeight: 500,
};

const destructiveBtnStyle = {
  background: 'transparent',
  border: '1px solid rgba(248,113,113,0.3)',
  color: '#f87171',
  borderRadius: 8,
  padding: '8px 18px',
  fontSize: 13,
  fontFamily: 'Inter, sans-serif',
  cursor: 'pointer',
  fontWeight: 500,
};

const inputStyle = {
  background: 'rgba(8,8,20,0.5)',
  border: '1px solid rgba(124,58,237,0.25)',
  borderRadius: 8,
  padding: '8px 12px',
  fontSize: 14,
  color: '#e2e8f0',
  fontFamily: 'Inter, sans-serif',
  outline: 'none',
  width: '100%',
  boxSizing: 'border-box',
};

const dotStyle = (connected) => ({
  width: 8,
  height: 8,
  borderRadius: '50%',
  background: connected ? '#4ade80' : '#f87171',
  flexShrink: 0,
});

const hoverTap = { whileHover: { scale: 1.02 }, whileTap: { scale: 0.97 } };

export default function ProfileTab({ trackCount = 0 }) {
  const { profile, updateUsername, linkSupabase, signOut } = useAuthStore();
  const [editingName, setEditingName] = useState(false);
  const [nameInput, setNameInput] = useState('');
  const [nameSaving, setNameSaving] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);

  const handleStartEdit = () => {
    setNameInput(profile?.username || '');
    setEditingName(true);
  };

  const handleSaveName = async () => {
    if (!nameInput.trim()) return;
    setNameSaving(true);
    try {
      await updateUsername(nameInput.trim());
    } catch { /* ignore */ }
    setNameSaving(false);
    setEditingName(false);
  };

  const handleCancelEdit = () => {
    setEditingName(false);
  };

  const [scanStatus, setScanStatus] = useState(null); // null | 'picking' | 'done' | 'error'

  const handleRescan = async () => {
    if (!isFileSystemAccessSupported()) {
      setScanStatus('error');
      return;
    }
    try {
      setScanStatus('picking');
      await pickMusicFolder();
      setScanStatus('done');
    } catch {
      setScanStatus(null); // user cancelled
    }
  };

  const handleLink = async (url, key) => {
    await linkSupabase(url, key);
    syncFromProfile({ ...profile, supabase_url: url, supabase_key: key });
    setModalOpen(false);
  };

  const supabaseConnected = !!profile?.supabase_url;
  const libraryImported = trackCount > 0;

  const maskedUrl = profile?.supabase_url
    ? profile.supabase_url.replace(/^(https?:\/\/[^.]+)(.*)$/, (_, prefix) => prefix + '.***')
    : null;

  return (
    <div style={{ paddingTop: 70, maxWidth: 600, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 16, padding: '70px 16px 32px' }}>

      {/* Username */}
      <div style={panelStyle}>
        <p style={{ ...labelStyle, marginBottom: 12 }}>Username</p>
        {editingName ? (
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <input
              style={inputStyle}
              value={nameInput}
              onChange={(e) => setNameInput(e.target.value)}
              autoFocus
              onKeyDown={(e) => e.key === 'Enter' && handleSaveName()}
            />
            <m.button style={primaryBtnStyle} {...hoverTap} onClick={handleSaveName} disabled={nameSaving}>
              {nameSaving ? 'Saving...' : 'Save'}
            </m.button>
            <m.button style={secondaryBtnStyle} {...hoverTap} onClick={handleCancelEdit}>
              Cancel
            </m.button>
          </div>
        ) : (
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <p style={valueStyle}>{profile?.username || 'Unknown'}</p>
            <m.button style={secondaryBtnStyle} {...hoverTap} onClick={handleStartEdit}>
              Change
            </m.button>
          </div>
        )}
      </div>

      {/* Supabase Connection */}
      <div style={panelStyle}>
        <p style={{ ...labelStyle, marginBottom: 12 }}>Supabase Connection</p>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
          <div style={dotStyle(supabaseConnected)} />
          <span style={valueStyle}>{supabaseConnected ? 'Connected' : 'Not linked'}</span>
        </div>
        {supabaseConnected && maskedUrl && (
          <p style={{ ...valueStyle, fontSize: 12, color: '#94a3b8', marginBottom: 12, fontFamily: 'JetBrains Mono, monospace' }}>
            {maskedUrl}
          </p>
        )}
        <m.button style={primaryBtnStyle} {...hoverTap} onClick={() => setModalOpen(true)}>
          {supabaseConnected ? 'Update Credentials' : 'Link Supabase'}
        </m.button>
      </div>

      {/* Library */}
      <div style={panelStyle}>
        <p style={{ ...labelStyle, marginBottom: 12 }}>Library</p>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
          <div style={dotStyle(libraryImported)} />
          <span style={valueStyle}>
            {libraryImported ? `${trackCount} tracks synced` : 'No library imported yet'}
          </span>
        </div>
        <m.button style={secondaryBtnStyle} {...hoverTap} onClick={handleRescan} disabled={scanStatus === 'picking'}>
          {scanStatus === 'picking' ? 'Selecting folder...' : 'Re-scan Library'}
        </m.button>
        {scanStatus === 'done' && (
          <p style={{ fontSize: 11, color: '#4ade80', margin: '8px 0 0', fontFamily: 'Inter, sans-serif' }}>
            Folder linked for local playback
          </p>
        )}
        {scanStatus === 'error' && (
          <p style={{ fontSize: 11, color: '#f87171', margin: '8px 0 0', fontFamily: 'Inter, sans-serif' }}>
            Folder picker requires Chrome, Edge, or Arc
          </p>
        )}
      </div>

      {/* Account */}
      <div style={panelStyle}>
        <p style={{ ...labelStyle, marginBottom: 12 }}>Account</p>
        <m.button style={destructiveBtnStyle} {...hoverTap} onClick={signOut}>
          Sign Out
        </m.button>
      </div>

      {/* Supabase Link Modal */}
      <SupabaseLinkModal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        onLink={handleLink}
        defaultUrl={profile?.supabase_url || ''}
        defaultKey={profile?.supabase_key || ''}
      />
    </div>
  );
}
