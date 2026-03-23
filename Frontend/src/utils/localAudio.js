// Frontend/src/utils/localAudio.js
// File System Access API wrapper for playing audio from the user's local disc.
// Chromium-only (Chrome, Edge, Brave). Firefox/Safari not supported.

const DB_NAME = 'djmate_fs';
const STORE_NAME = 'handles';
const HANDLE_KEY = 'musicDir';

// ── IndexedDB helpers for persisting the directory handle ──────────────

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE_NAME);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function saveHandle(handle) {
  const db = await openDB();
  const tx = db.transaction(STORE_NAME, 'readwrite');
  tx.objectStore(STORE_NAME).put(handle, HANDLE_KEY);
  return new Promise((resolve, reject) => {
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
}

async function loadHandle() {
  const db = await openDB();
  const tx = db.transaction(STORE_NAME, 'readonly');
  const req = tx.objectStore(STORE_NAME).get(HANDLE_KEY);
  return new Promise((resolve) => {
    req.onsuccess = () => resolve(req.result || null);
    req.onerror = () => resolve(null);
  });
}

// ── Public API ─────────────────────────────────────────────────────────

/** Check if the File System Access API is available (Chromium only). */
export function isFileSystemAccessSupported() {
  return 'showDirectoryPicker' in window;
}

/** Prompt user to pick their music folder. Persists handle to IndexedDB. */
export async function pickMusicFolder() {
  const handle = await window.showDirectoryPicker({ mode: 'read' });
  await saveHandle(handle);
  return handle;
}

/** Get the stored directory handle, or null if none saved. */
export async function getMusicFolderHandle() {
  return loadHandle();
}

/** Clear the stored handle (for reconfigure). */
export async function clearMusicFolder() {
  const db = await openDB();
  const tx = db.transaction(STORE_NAME, 'readwrite');
  tx.objectStore(STORE_NAME).delete(HANDLE_KEY);
}

/**
 * Verify we still have read permission for the stored handle.
 * Returns true if granted, false if the user needs to re-grant.
 */
export async function verifyPermission(handle) {
  if (!handle) return false;
  const opts = { mode: 'read' };
  if ((await handle.queryPermission(opts)) === 'granted') return true;
  if ((await handle.requestPermission(opts)) === 'granted') return true;
  return false;
}

/**
 * Recursively search a directory for a file by name.
 */
async function findFileInDir(dirHandle, filename) {
  for await (const [name, handle] of dirHandle.entries()) {
    if (handle.kind === 'file' && name === filename) {
      return handle;
    }
    if (handle.kind === 'directory') {
      try {
        const found = await findFileInDir(handle, filename);
        if (found) return found;
      } catch {
        // Permission denied on subdirectory — skip
      }
    }
  }
  return null;
}

/**
 * Get a playable blob URL for a track's filepath.
 * The filepath comes from the database and is an absolute local path
 * like /Users/dave/Music/artist/track.mp3. We extract the filename
 * and search for it recursively in the user's granted music folder.
 *
 * Returns null if the file can't be found or permission is denied.
 */
export async function getLocalAudioUrl(filepath) {
  if (!filepath) return null;

  const handle = await getMusicFolderHandle();
  if (!handle) return null;

  const hasPermission = await verifyPermission(handle);
  if (!hasPermission) return null;

  // Extract filename from the absolute path (handles both / and \ separators)
  const filename = filepath.split('/').pop().split('\\').pop();
  if (!filename) return null;

  try {
    const fileHandle = await findFileInDir(handle, filename);
    if (!fileHandle) return null;

    const file = await fileHandle.getFile();
    return URL.createObjectURL(file);
  } catch {
    return null;
  }
}

/**
 * Check whether we have a music folder handle stored and it has permission.
 * Useful for showing UI state without triggering a permission prompt.
 */
export async function hasMusicFolder() {
  const handle = await getMusicFolderHandle();
  if (!handle) return false;
  try {
    return (await handle.queryPermission({ mode: 'read' })) === 'granted';
  } catch {
    return false;
  }
}
