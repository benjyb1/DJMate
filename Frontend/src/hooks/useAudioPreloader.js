// hooks/useAudioPreloader.js
// Preloads audio URLs for a list of tracks when a playlist is opened.
// Uses File System Access API first, falls back to backend streaming endpoint.

import { useRef, useCallback, useEffect, useState } from 'react';
import { getLocalAudioUrl, hasMusicFolder } from '../utils/localAudio';

const API_BASE = import.meta.env.VITE_API_URL
  || (import.meta.env.DEV ? 'http://localhost:8000' : 'https://djmate.onrender.com');

/**
 * Resolves a playable URL for a single track.
 * Priority: local File System Access → backend /tracks/{id}/audio
 */
async function resolveTrackUrl(track, useLocal) {
  // Try local filesystem first (instant playback from disk)
  if (useLocal && track.filepath) {
    const blobUrl = await getLocalAudioUrl(track.filepath);
    if (blobUrl) return { url: blobUrl, isBlob: true };
  }
  // Fallback: backend stream (no preload fetch needed, just the URL)
  const id = track.trackid || track.id;
  return { url: `${API_BASE}/tracks/${id}/audio`, isBlob: false };
}

/**
 * Hook that preloads audio URLs for an array of tracks.
 * Returns a Map<trackId, url> that grows as URLs resolve.
 */
export function useAudioPreloader(tracks) {
  const [cache, setCache] = useState(new Map());
  const blobUrls = useRef(new Set());
  const abortRef = useRef(null);

  // Preload whenever tracks change
  useEffect(() => {
    if (!tracks || tracks.length === 0) {
      setCache(new Map());
      return;
    }

    // Abort previous preload batch
    if (abortRef.current) abortRef.current.aborted = true;
    const controller = { aborted: false };
    abortRef.current = controller;

    let mounted = true;

    (async () => {
      const hasLocal = await hasMusicFolder();

      // Resolve all tracks concurrently
      const promises = tracks.map(async (track) => {
        const id = track.trackid || track.id;
        if (!id) return null;
        try {
          const result = await resolveTrackUrl(track, hasLocal);
          if (controller.aborted || !mounted) return null;
          if (result.isBlob) blobUrls.current.add(result.url);
          return { id, url: result.url };
        } catch {
          return null;
        }
      });

      // Update cache as each resolves (batched via Promise.allSettled)
      const results = await Promise.allSettled(promises);
      if (controller.aborted || !mounted) return;

      const newCache = new Map();
      for (const r of results) {
        if (r.status === 'fulfilled' && r.value) {
          newCache.set(r.value.id, r.value.url);
        }
      }
      setCache(newCache);
    })();

    return () => {
      mounted = false;
      controller.aborted = true;
    };
  }, [tracks]);

  // Cleanup blob URLs on unmount
  useEffect(() => {
    const urls = blobUrls.current;
    return () => {
      for (const url of urls) {
        URL.revokeObjectURL(url);
      }
      urls.clear();
    };
  }, []);

  /** Get a preloaded URL, or null if not yet resolved */
  const getUrl = useCallback((trackId) => cache.get(trackId) || null, [cache]);

  return { getUrl, preloadedCount: cache.size };
}
