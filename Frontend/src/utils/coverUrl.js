import { getCredentials } from '../stores/credentialStore';

function getCoversBase() {
  const creds = getCredentials();
  if (creds?.url) return `${creds.url}/storage/v1/object/public/album-covers/`;
  return 'https://cvermotfxamubejfnoje.supabase.co/storage/v1/object/public/album-covers/';
}

/**
 * Build the cover URL for a track using its trackid.
 * Covers are stored as {trackid}.jpg — no filename generation needed.
 */
export function makeSupabaseCoverUrl(trackid) {
  if (!trackid) return null;
  return `${getCoversBase()}${trackid}.jpg`;
}
