// Tiny static file server for the built frontend (Frontend/dist).
//
// Serving over http://localhost:<port> — rather than file:// — matters for two
// reasons:
//   1. Vite emits absolute asset paths (/assets/…), which only resolve from a
//      server root.
//   2. The frontend persists the chosen music-folder handle in IndexedDB, which
//      is keyed by origin. A fixed localhost port keeps that origin stable
//      across launches, so the user picks their folder once, not every time.
//
// No dependencies — Node's built-ins only, so the Electron main process stays
// lean and doesn't need a native module compiled against Electron's ABI.

const http = require('http');
const fs = require('fs');
const path = require('path');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.ico': 'image/x-icon',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf',
  '.map': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
};

/**
 * Start a static server for `distDir` on `port` (bound to loopback only).
 * Returns a promise resolving to the http.Server once it is listening.
 */
function startStaticServer(distDir, port) {
  const server = http.createServer((req, res) => {
    // Strip query string and decode.
    let urlPath = decodeURIComponent((req.url || '/').split('?')[0]);
    if (urlPath === '/') urlPath = '/index.html';

    // Resolve within distDir and guard against path traversal.
    const resolved = path.normalize(path.join(distDir, urlPath));
    if (!resolved.startsWith(distDir)) {
      res.writeHead(403);
      res.end('Forbidden');
      return;
    }

    fs.stat(resolved, (err, stat) => {
      if (err || !stat.isFile()) {
        // SPA fallback: unknown routes and extension-less paths get index.html
        // so client-side routing works on reload.
        const hasExt = path.extname(urlPath) !== '';
        if (!hasExt) {
          serveFile(path.join(distDir, 'index.html'), res);
          return;
        }
        res.writeHead(404);
        res.end('Not found');
        return;
      }
      serveFile(resolved, res);
    });
  });

  return new Promise((resolve, reject) => {
    server.on('error', reject);
    server.listen(port, '127.0.0.1', () => resolve(server));
  });
}

function serveFile(filePath, res) {
  const ext = path.extname(filePath).toLowerCase();
  const type = MIME[ext] || 'application/octet-stream';
  res.writeHead(200, { 'Content-Type': type });
  fs.createReadStream(filePath)
    .on('error', () => {
      res.writeHead(500);
      res.end('Read error');
    })
    .pipe(res);
}

module.exports = { startStaticServer };
