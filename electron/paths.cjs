// Locates the pieces the desktop app needs at runtime: the repo root that holds
// the backend + .env, a Python interpreter, and the built frontend.
//
// The app is developed inside a git worktree
// (…/DJMate/.claude/worktrees/<name>) but the working .env, virtualenv and
// models live in the main checkout (…/DJMate). So "where the Electron code
// sits" and "where the backend must run" can differ — this module reconciles
// the two.
//
// Packaged as Needle.app, this code sits inside the app bundle, far from the
// repo. The build bakes the repo's location into repo-root.json, and
// NEEDLE_REPO can override it. The built frontend ships inside the bundle.

const fs = require('fs');
const os = require('os');
const path = require('path');

const EFFNET_MODEL = 'discogs-effnet-bs64.pb';

// Repo root that physically contains this electron/ folder.
const APP_ROOT = path.resolve(__dirname, '..');

/** If `root` is a git worktree, return the main checkout root; else null. */
function mainCheckoutOf(root) {
  const marker = `${path.sep}.claude${path.sep}worktrees${path.sep}`;
  const idx = root.indexOf(marker);
  return idx === -1 ? null : root.slice(0, idx);
}

function exists(p) {
  try {
    return fs.existsSync(p);
  } catch {
    return false;
  }
}

/** Repo roots to try, most specific first. */
function repoRoots() {
  const roots = [];
  if (process.env.NEEDLE_REPO) roots.push(process.env.NEEDLE_REPO);
  try {
    const baked = JSON.parse(fs.readFileSync(path.join(__dirname, 'repo-root.json'), 'utf8'));
    if (baked && baked.root) roots.push(baked.root);
  } catch { /* not packaged, or built without it */ }
  roots.push(APP_ROOT);
  const main = mainCheckoutOf(APP_ROOT);
  if (main) roots.push(main);
  return [...new Set(roots)];
}

/**
 * The checkout the backend should run from — the one that actually has a .env
 * (and therefore Supabase / LLM keys). Prefers APP_ROOT, then the main checkout
 * when APP_ROOT is a worktree.
 */
function resolveRuntimeRoot() {
  const candidates = repoRoots();
  for (const c of candidates) {
    if (exists(path.join(c, '.env')) && exists(path.join(c, 'main.py'))) return c;
  }
  // Fall back to the first candidate that at least has the backend code.
  for (const c of candidates) {
    if (exists(path.join(c, 'main.py'))) return c;
  }
  return APP_ROOT;
}

/** Find a Python interpreter, preferring a project virtualenv. */
function resolvePython(runtimeRoot) {
  if (process.env.DJMATE_PYTHON && exists(process.env.DJMATE_PYTHON)) {
    return process.env.DJMATE_PYTHON;
  }
  const venvNames = ['.venv1', '.venv', 'venv', 'env'];
  const roots = [runtimeRoot, ...repoRoots()].filter(Boolean);
  for (const root of roots) {
    for (const v of venvNames) {
      const p = path.join(root, v, 'bin', 'python');
      if (exists(p)) return p;
    }
  }
  return 'python3';
}

/** Locate the built frontend (Frontend/dist with an index.html). */
function resolveFrontendDist() {
  // Packaged: the build copies Frontend/dist into the bundle's Resources.
  if (process.resourcesPath) {
    const bundled = path.join(process.resourcesPath, 'frontend');
    if (exists(path.join(bundled, 'index.html'))) return bundled;
  }
  const roots = repoRoots();
  for (const root of roots) {
    const dist = path.join(root, 'Frontend', 'dist');
    if (exists(path.join(dist, 'index.html'))) return dist;
  }
  // Return the APP_ROOT path even if missing, so callers can report it.
  return path.join(APP_ROOT, 'Frontend', 'dist');
}

/**
 * Locate the Essentia EffNet embedding model that ingest_music.py needs. The
 * in-app ingest runs that script via the backend, so if the model can't be
 * found the embedding step fails. We mirror the script's own resolution order
 * and add the known local fallback under ~/Desktop/OlderFiles/Models.
 * Returns an absolute path, or null if nothing is found.
 */
function resolveEssentiaModel(runtimeRoot) {
  const envPaths = [process.env.ESSENTIA_MODEL_PATH, process.env.DJMATE_ESSENTIA_MODEL];
  for (const p of envPaths) {
    if (p && exists(p)) return p;
  }
  const roots = [runtimeRoot, ...repoRoots()].filter(Boolean);
  const candidates = roots.map((r) => path.join(r, 'models', EFFNET_MODEL));
  candidates.push(path.join(os.homedir(), 'Desktop', 'OlderFiles', 'Models', EFFNET_MODEL));
  for (const c of candidates) {
    if (exists(c)) return c;
  }
  return null;
}

module.exports = {
  APP_ROOT,
  repoRoots,
  resolveRuntimeRoot,
  resolvePython,
  resolveFrontendDist,
  resolveEssentiaModel,
  exists,
};
