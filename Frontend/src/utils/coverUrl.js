import { getCredentials } from '../stores/credentialStore';

function getCoversBase() {
  const creds = getCredentials();
  if (creds?.url) return `${creds.url}/storage/v1/object/public/album-covers/`;
  // Fallback for dev
  return 'https://cvermotfxamubejfnoje.supabase.co/storage/v1/object/public/album-covers/';
}

/**
 * Build the safe ASCII storage filename for a track's cover art.
 *
 * Must produce IDENTICAL output to create_safe_filename() in
 * scripts/upload_album_covers.py and scripts/populate_album_art_urls.py
 * and _make_cover_filename() in scripts/ingest_music.py:
 *
 *   1. NFKD-decompose  (é → e + combining acute)
 *   2. Drop ALL non-ASCII chars  (equivalent to Python's .encode('ascii','ignore'))
 *   3. Lowercase
 *   4. Replace anything not [a-z0-9-_] with _
 *   5. Collapse consecutive underscores
 *   6. Truncate to 150 chars
 */
export function makeSupabaseCoverUrl(artist, title) {
  if (!artist || !title) return null;
  let safe = `${artist}_${title}`;
  // Step 1+2: NFKD then drop every non-ASCII codepoint (matches Python's
  //           .encode("ascii", "ignore").decode("ascii"))
  safe = safe.normalize('NFKD').replace(/[^\x00-\x7F]/g, '');
  // Step 3-6: lowercase → keep only alnum/hyphen/underscore → collapse _ → cap
  safe = safe.toLowerCase();
  safe = safe.split('').map(c => /[a-z0-9\-_]/.test(c) ? c : '_').join('');
  safe = safe.split('_').filter(Boolean).join('_').slice(0, 150);
  return `${getCoversBase()}${safe}.jpg`;
}
