// Frontend/src/utils/resolveAudio.js
// Centralised audio source resolution with blob URL caching.
//
// Resolution order:
//   1. In-memory cache (instant — already resolved blob or backend URL)
//   2. Local File System Access API (Chromium — reads from user's music folder)
//   3. Backend streaming endpoint /tracks/{id}/audio (works everywhere)
//
// The old Supabase Storage "audio-files" bucket has been removed.

import { getLocalAudioUrl } from './localAudio';

const API_BASE = import.meta.env.VITE_API_URL
  || (import.meta.env.DEV ? 'http://localhost:8000' : 'https://djmate.onrender.com');

// LRU-ish cache: trackId → blob URL or backend URL string.
// Blob URLs are only created once per track and reused until the page unloads.
const _cache = new Map();

// Track which entries are blob URLs so we can revoke them on eviction.
const _isBlob = new Set();

const MAX_CACHE = 200;

function _evictOldest() {
  if (_cache.size <= MAX_CACHE) return;
  const oldest = _cache.keys().next().value;
  if (_isBlob.has(oldest)) {
    URL.revokeObjectURL(_cache.get(oldest));
    _isBlob.delete(oldest);
  }
  _cache.delete(oldest);
}

/**
 * Resolve a playable audio URL for a track.
 *
 * @param {object} track — needs at least { id|trackid } and optionally { filepath }
 * @returns {Promise<string>} a URL the browser can feed to an <audio> element
 */
export async function resolveAudioSrc(track) {
  const id = track.id || track.trackid;
  if (!id) return '';

  // 1. Cache hit — return immediately
  if (_cache.has(id)) return _cache.get(id);

  // 2. Try local filesystem (fast, no network)
  const filepath = track.filepath || null;
  if (filepath) {
    try {
      const blobUrl = await getLocalAudioUrl(filepath);
      if (blobUrl) {
        _cache.set(id, blobUrl);
        _isBlob.add(id);
        _evictOldest();
        return blobUrl;
      }
    } catch {
      // Filesystem API unavailable or permission denied — fall through
    }
  }

  // 3. Backend streaming endpoint (always available)
  const backendUrl = `${API_BASE}/tracks/${id}/audio`;
  _cache.set(id, backendUrl);
  _evictOldest();
  return backendUrl;
}

/**
 * Synchronous check — returns a cached URL if we already resolved this track,
 * otherwise returns the backend fallback URL (no filesystem attempt).
 * Useful when you need a src immediately and can't await.
 */
export function resolveAudioSrcSync(track) {
  const id = track.id || track.trackid;
  if (!id) return '';
  if (_cache.has(id)) return _cache.get(id);
  return `${API_BASE}/tracks/${id}/audio`;
}

/** Revoke all cached blob URLs. Call on unmount if you want to free memory. */
export function clearAudioCache() {
  for (const id of _isBlob) {
    URL.revokeObjectURL(_cache.get(id));
  }
  _isBlob.clear();
  _cache.clear();
}
