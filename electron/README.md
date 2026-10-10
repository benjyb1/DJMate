# DJMate desktop (macOS)

A thin Electron shell that turns DJMate into a native macOS app. On launch it
serves the built frontend locally and starts the Python backend for you, so
there's no browser tab, no `npm run dev`, and no manual `uvicorn` — just open
the app.

## What it does

- Serves `Frontend/dist` on a fixed port (`http://localhost:5178`). The port is
  fixed on purpose: the app stores your chosen music folder in IndexedDB, which
  is keyed by origin, so a stable port means you pick the folder **once**.
- Starts the backend with `uvicorn main:app` on `:8000` using the project
  virtualenv, unless something is already answering there (then it reuses it).
- Local audio plays straight from disk via the File System Access API, which
  works in Electron's Chromium.
- Closing the window quits the app and stops the backend it started.

## Requirements (already set up on this machine)

- The Python virtualenv at `../.venv1` with the backend deps installed.
- A `../.env` with your Supabase + LLM keys (the backend reads it).
- Node / npm (for the one-off Electron install).

## Run it

```bash
cd electron
npm install        # first time only — downloads Electron
npm start
```

Or double-click **`DJMate.command`** in Finder (right-click → Open the first
time, to clear Gatekeeper).

## Rebuild the frontend after code changes

The bundled UI is a static build. After changing anything under `Frontend/src`:

```bash
npm run build:frontend   # from the electron/ folder
```

## Data source

DJMate needs a live Supabase project. The backend reads its URL/keys from
`../.env`; the frontend's are baked in at build time from `../Frontend/.env`.
If you point at a new project, update both and re-run `npm run build:frontend`.

## Notes / limits

- Paths are resolved relative to the repo, so run the app from inside the repo
  (via `npm start` or `DJMate.command`). `npm run dist` produces an unsigned
  `.app` under `release/`, but it still expects the repo (venv, `.env`, models)
  present on this machine — it does not bundle Python.
- Override the interpreter with `DJMATE_PYTHON=/path/to/python npm start`.

## Needle.app

The app is now called Needle. To build a real macOS app:

```bash
cd electron
npm run dist            # builds the frontend, then release/mac-arm64/Needle.app
ditto release/mac-arm64/Needle.app /Applications/Needle.app
```

The bundle carries the Electron shell and the built frontend. Python, the
models and the backend code stay in the repo: the build records the repo's
location in `repo-root.json`, and `NEEDLE_REPO=/path/to/DJMate` overrides it
if the repo moves. Rebuild after frontend changes; backend changes only need
an app restart. Logs: `~/Library/Logs/Needle/needle.log`.
