import React from 'react';
import { useAlbumArt } from '../../hooks/useAlbumArt';

/**
 * Shared album art component with Supabase -> iTunes -> generative fallback.
 * Matches the hash/hue logic used in DJChatbox and LiveMode.
 */
export function AlbumArt({ artist, title, directUrl, trackid, size = 48, radius = 'var(--radius-sm)', style = {} }) {
  const { url, handleImgError } = useAlbumArt(artist, title, directUrl, trackid);

  if (url) {
    return (
      <img
        src={url}
        alt={`${title} artwork`}
        width={size}
        height={size}
        onError={handleImgError}
        style={{
          width: size, height: size, objectFit: 'cover', display: 'block',
          flexShrink: 0, borderRadius: radius, ...style,
        }}
      />
    );
  }

  // Generative fallback — hash-based hue with initials
  const str = (title || '') + (artist || '');
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash) + str.charCodeAt(i);
    hash |= 0;
  }
  const hue = 200 + (Math.abs(hash) % 100);
  const initials = (str || '?').replace(/[^a-zA-Z0-9]/g, '').slice(0, 2).toUpperCase() || '??';

  return (
    <div style={{
      width: size, height: size, flexShrink: 0,
      background: `linear-gradient(135deg, hsl(${hue},50%,8%), hsl(${hue},70%,5%))`,
      border: `1px solid hsla(${hue},70%,50%,0.3)`,
      borderRadius: radius,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: "'JetBrains Mono', monospace",
      fontSize: Math.round(size * 0.28), fontWeight: 600,
      color: `hsla(${hue},80%,70%,0.9)`, letterSpacing: '0.02em',
      ...style,
    }}>
      {initials}
    </div>
  );
}
