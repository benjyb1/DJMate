// Needle (DJMate) desktop — Electron main process.
//
// On launch it:
//   1. Serves the built frontend on a fixed loopback port (stable origin so the
//      music-folder handle in IndexedDB survives restarts).
//   2. Reuses an already-running backend on :8000, or spawns uvicorn from the
//      project virtualenv.
//   3. Waits for the backend to report healthy, then opens the window.
//
// Quitting the app tears down the backend it started (but leaves alone a
// backend that was already running before launch).

const { app, BrowserWindow, shell, dialog, nativeImage } = require('electron');

// The app's name in the menu bar, About box and notifications.
app.setName('Needle');
const http = require('http');
const path = require('path');
const { spawn } = require('child_process');

const {
  resolveRuntimeRoot,
  resolvePython,
  resolveFrontendDist,
  resolveEssentiaModel,
} = require('./paths.cjs');
const { startStaticServer } = require('./static-server.cjs');

const FRONTEND_PORT = 5178;          // stable origin for the packaged frontend
const BACKEND_HOST = '127.0.0.1';
const BACKEND_PORT = 8000;           // must match the baked VITE_API_URL
const BACKEND_URL = `http://${BACKEND_HOST}:${BACKEND_PORT}`;
const BACKEND_STARTUP_TIMEOUT_MS = 45_000;

let mainWindow = null;
let staticServer = null;
let backendProc = null;              // set only if WE spawned it
let backendLog = [];

// Launched from Finder there's no terminal, so everything also goes to
// ~/Library/Logs/Needle/needle.log.
const LOG_FILE = require('path').join(require('os').homedir(), 'Library', 'Logs', 'Needle', 'needle.log');
try { require('fs').mkdirSync(require('path').dirname(LOG_FILE), { recursive: true }); } catch { /* best effort */ }
function toLogFile(text) {
  try { require('fs').appendFileSync(LOG_FILE, text); } catch { /* best effort */ }
}

function log(...args) {
  const line = `[needle] ${args.join(' ')}`;
  console.log(line);
  toLogFile(`${new Date().toISOString()} ${line}\n`);
}

// ── Backend health ────────────────────────────────────────────────────────────

function pingBackend(timeoutMs = 1500) {
  return new Promise((resolve) => {
    const req = http.get(`${BACKEND_URL}/`, { timeout: timeoutMs }, (res) => {
      res.resume();
      resolve(res.statusCode >= 200 && res.statusCode < 500);
    });
    req.on('error', () => resolve(false));
    req.on('timeout', () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function waitForBackend(deadlineMs) {
  const start = Date.now();
  while (Date.now() - start < deadlineMs) {
    if (await pingBackend()) return true;
    await new Promise((r) => setTimeout(r, 600));
  }
  return false;
}

// ── Backend process ────────────────────────────────────────────────────────────

function spawnBackend() {
  const runtimeRoot = resolveRuntimeRoot();
  const python = resolvePython(runtimeRoot);

  log(`Starting backend: ${python} -m uvicorn main:app (cwd=${runtimeRoot})`);

  const env = {
    ...process.env,
    PYTHONUNBUFFERED: '1',
    // Allow the packaged frontend's origin through CORS.
    CORS_ORIGINS: [
      `http://localhost:${FRONTEND_PORT}`,
      `http://127.0.0.1:${FRONTEND_PORT}`,
      'http://localhost:5173',
    ].join(','),
  };

  // In-app ingest shells out to ingest_music.py, which needs the EffNet model.
  // Point it there so the embedding step works without a manual env var.
  const model = resolveEssentiaModel(runtimeRoot);
  if (model) {
    env.ESSENTIA_MODEL_PATH = model;
    log(`EffNet model: ${model}`);
  } else {
    log('EffNet model not found — in-app ingest will fail at the embedding step until it is placed in models/.');
  }

  const child = spawn(
    python,
    ['-m', 'uvicorn', 'main:app', '--host', BACKEND_HOST, '--port', String(BACKEND_PORT)],
    { cwd: runtimeRoot, env, stdio: ['ignore', 'pipe', 'pipe'] },
  );

  const capture = (buf) => {
    const text = buf.toString();
    backendLog.push(text);
    if (backendLog.length > 400) backendLog = backendLog.slice(-400);
    process.stdout.write(`[backend] ${text}`);
    toLogFile(`[backend] ${text}`);
  };
  child.stdout.on('data', capture);
  child.stderr.on('data', capture);
  child.on('exit', (code, signal) => {
    log(`Backend process exited (code=${code}, signal=${signal})`);
    backendProc = null;
  });

  return child;
}

function stopBackend() {
  if (backendProc && !backendProc.killed) {
    log('Stopping backend we started…');
    try {
      backendProc.kill('SIGTERM');
    } catch { /* already gone */ }
    // Hard stop if it lingers.
    const proc = backendProc;
    setTimeout(() => {
      try {
        if (proc && !proc.killed) proc.kill('SIGKILL');
      } catch { /* ignore */ }
    }, 4000);
    backendProc = null;
  }
}

// ── Window ──────────────────────────────────────────────────────────────────

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1024,
    minHeight: 680,
    backgroundColor: '#0b0b0f',
    title: 'Needle',
    titleBarStyle: 'hiddenInset',
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      spellcheck: false,
    },
  });

  // Open target=_blank / external links in the system browser, not new windows.
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('http')) shell.openExternal(url);
    return { action: 'deny' };
  });

  mainWindow.loadURL(`http://localhost:${FRONTEND_PORT}`);
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// ── Lifecycle ────────────────────────────────────────────────────────────────

function applyDockIcon() {
  if (process.platform !== 'darwin' || !app.dock) return;
  const iconPath = path.join(__dirname, 'assets', 'icon.png');
  const img = nativeImage.createFromPath(iconPath);
  if (!img.isEmpty()) app.dock.setIcon(img);
}

async function boot() {
  applyDockIcon();

  const distDir = resolveFrontendDist();
  const fs = require('fs');
  if (!fs.existsSync(path.join(distDir, 'index.html'))) {
    dialog.showErrorBox(
      'Needle — frontend not built',
      `No built frontend found at:\n${distDir}\n\n` +
        `Build it first:\n  cd Frontend && npm install && npm run build\n` +
        `(or run "npm run build:frontend" from the electron/ folder)`,
    );
    app.quit();
    return;
  }

  // 1. Static server for the frontend.
  try {
    staticServer = await startStaticServer(distDir, FRONTEND_PORT);
    log(`Frontend served from ${distDir} at http://localhost:${FRONTEND_PORT}`);
  } catch (err) {
    dialog.showErrorBox(
      'Needle — cannot start',
      `Port ${FRONTEND_PORT} is in use, so the frontend can't be served.\n\n${err.message}`,
    );
    app.quit();
    return;
  }

  // 2. Backend — reuse if already up, else spawn.
  const alreadyUp = await pingBackend();
  if (alreadyUp) {
    log('Reusing backend already running on :8000');
  } else {
    backendProc = spawnBackend();
  }

  // 3. Window as soon as we have a page to show. Wait briefly for the backend so
  //    the first data calls succeed, but never hang forever on it.
  createWindow();
  const healthy = await waitForBackend(BACKEND_STARTUP_TIMEOUT_MS);
  if (!healthy) {
    log('Backend did not become healthy in time — the window is open but API calls may fail.');
    if (mainWindow) {
      mainWindow.webContents.once('did-finish-load', () => {
        dialog.showMessageBox(mainWindow, {
          type: 'warning',
          title: 'Backend not responding',
          message: 'Needle opened, but the local backend did not start.',
          detail:
            'Check that your Python virtualenv and .env are in place. ' +
            'Recent backend output:\n\n' +
            backendLog.slice(-12).join(''),
          buttons: ['OK'],
        });
      });
    }
  } else {
    log('Backend healthy.');
  }
}

app.whenReady().then(boot);

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0 && staticServer) createWindow();
});

app.on('window-all-closed', () => {
  app.quit(); // single-window utility: closing the window quits the app
});

app.on('before-quit', () => {
  stopBackend();
  if (staticServer) {
    try {
      staticServer.close();
    } catch { /* ignore */ }
  }
});
