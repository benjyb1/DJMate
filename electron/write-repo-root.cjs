// Build step for Needle.app: record where the repo lives (the checkout with
// main.py, .env and the virtualenv) so the packaged app can start the backend
// from there. Override at runtime with NEEDLE_REPO if the repo ever moves.
const fs = require('fs');
const path = require('path');
const { resolveRuntimeRoot } = require('./paths.cjs');

const root = resolveRuntimeRoot();
if (!fs.existsSync(path.join(root, 'main.py'))) {
  console.error(`No main.py under ${root}; can't tell Needle.app where the backend is.`);
  process.exit(1);
}
fs.writeFileSync(path.join(__dirname, 'repo-root.json'), JSON.stringify({ root }, null, 2) + '\n');
console.log(`Needle.app will run the backend from ${root}`);
