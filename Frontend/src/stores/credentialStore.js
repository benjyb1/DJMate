// Frontend/src/stores/credentialStore.js
// localStorage wrapper for Supabase BYOS credentials

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

export function syncFromProfile(profile) {
  if (profile?.supabase_url && profile?.supabase_key) {
    saveCredentials(profile.supabase_url, profile.supabase_key);
  } else {
    clearCredentials();
  }
}
