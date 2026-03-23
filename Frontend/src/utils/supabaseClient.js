import { createClient } from '@supabase/supabase-js';
import { getCredentials } from '../stores/credentialStore';

let _client = null;
let _currentUrl = null;

/**
 * Returns a Supabase client initialised from localStorage creds.
 * Returns null if no creds are stored.
 * Recreates the client if the URL has changed (user reconfigured).
 */
export function getSupabase() {
  const creds = getCredentials();
  if (!creds) return null;

  if (!_client || _currentUrl !== creds.url) {
    _client = createClient(creds.url, creds.key);
    _currentUrl = creds.url;
  }
  return _client;
}

/** Force client recreation on next getSupabase() call. */
export function resetSupabaseClient() {
  _client = null;
  _currentUrl = null;
}

// Backwards compat — files that import { supabase } get a proxy
// that delegates to the current dynamic client.
export const supabase = new Proxy({}, {
  get(_, prop) {
    const client = getSupabase();
    if (!client) return undefined;
    const val = client[prop];
    // Bind methods so `this` stays correct
    return typeof val === 'function' ? val.bind(client) : val;
  }
});
