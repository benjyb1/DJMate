// Frontend/src/stores/credentialStore.js
// Manages Supabase BYOS credentials with localStorage + central DB sync.
//
// localStorage is the fast path (avoids a network call on every page load).
// The central DB is the source of truth so credentials survive across
// devices and browser clears.

const STORAGE_KEY_URL = 'djmate_supabase_url';
const STORAGE_KEY_KEY = 'djmate_supabase_key';

export function getCredentials() {
  const url = localStorage.getItem(STORAGE_KEY_URL);
  const key = localStorage.getItem(STORAGE_KEY_KEY);
  if (url && key) return { url, key };
  return null;
}

export function saveCredentials(url, key) {
  localStorage.setItem(STORAGE_KEY_URL, url.trim());
  localStorage.setItem(STORAGE_KEY_KEY, key.trim());
}

export function clearCredentials() {
  localStorage.removeItem(STORAGE_KEY_URL);
  localStorage.removeItem(STORAGE_KEY_KEY);
}

/**
 * Sync credentials from a user profile fetched from the central DB.
 * Called on sign-in so the user's linked project carries across devices.
 */
export function syncFromProfile(profile) {
  if (profile?.supabase_url && profile?.supabase_key) {
    saveCredentials(profile.supabase_url, profile.supabase_key);
    return true;
  }
  // Profile exists but has no linked project — don't wipe localStorage
  // in case the user linked on this device but hasn't synced to the DB yet.
  return false;
}

