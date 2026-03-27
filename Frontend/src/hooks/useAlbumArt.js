import { useState, useMemo, useCallback } from 'react';
import { makeSupabaseCoverUrl } from '../utils/coverUrl';

/**
 * Resolves album art URL with fallback chain:
 *   directUrl -> Supabase storage ({trackid}.jpg) -> iTunes search -> null.
 *
 * @param {string} artist
 * @param {string} title
 * @param {string|null} directUrl - Pre-existing URL (e.g. album_art_url from DB)
 * @param {string|number|null} trackid - Used as fallback filename in storage
 * @returns {{ url: string|null, error: boolean, handleImgError: () => void }}
 */
export function useAlbumArt(artist, title, directUrl, trackid) {
  const initialUrl = useMemo(() => {
    if (directUrl) return directUrl;
    if (trackid) return makeSupabaseCoverUrl(trackid);
    return null;
  }, [directUrl, trackid]);

  const [fallbackUrl, setFallbackUrl] = useState(null);
  const [error, setError] = useState(false);

  const url = error ? null : (fallbackUrl || initialUrl);

  const handleImgError = useCallback(() => {
    if (initialUrl && initialUrl.includes('supabase')) {
      const term = encodeURIComponent(`${artist || ''} ${title || ''}`.trim());
      if (!term) { setError(true); return; }
      fetch(`https://itunes.apple.com/search?term=${term}&entity=song&limit=1`)
        .then(r => r.json())
        .then(data => {
          const raw = data.results?.[0]?.artworkUrl100;
          if (raw) setFallbackUrl(raw.replace('100x100bb', '300x300bb'));
          else setError(true);
        })
        .catch(() => setError(true));
      return;
    }
    setError(true);
  }, [initialUrl, artist, title]);

  return { url, error, handleImgError };
}
