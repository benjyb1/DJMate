# DJMate Setup

Get your own DJMate running with your music library. Takes about 20 minutes.

You'll need: Python 3.11, Chrome/Edge/Brave, a free [Supabase](https://supabase.com) account.

---

## 1. Supabase project

1. [supabase.com](https://supabase.com) → **New Project** → name it whatever, pick a region near you
2. Once it's provisioned, go to **SQL Editor** → paste the entire contents of [`schema.sql`](https://github.com/benjyb1/DJMate/blob/main/schema.sql) → **Run**
3. Go to **Storage** → **New Bucket** → name it `album-covers` → set to **Public** → **Create**
4. Go to **Project Settings > API** and note down:
   - **Project URL** (`https://something.supabase.co`)
   - **anon key** (the long JWT under "Project API keys") — this one goes in the browser
   - **service_role key** (the other long JWT) — this one goes in your `.env`, never share it

## 2. Clone and install

```bash
git clone https://github.com/benjyb1/DJMate.git
cd DJMate
pip install -r configuration/requirements-scripts.txt
```

### Essentia (the annoying one)

Essentia doesn't install cleanly with a normal `pip install`. What works:

```bash
pip install essentia-tensorflow
```

If that fails (it probably will on some systems), try:

```bash
# macOS with Homebrew
brew install essentia

# Or use the pre-built wheel (Python 3.11 only)
pip install https://essentia.upf.edu/python/essentia-2.1b6.dev1389-cp311-cp311-macosx_11_0_arm64.whl

# Linux
pip install essentia-tensorflow
```

If you're on Windows, honestly use WSL. Essentia on native Windows is not worth the fight.

### Download the embedding model

```bash
mkdir -p models
curl -o models/discogs-effnet-bs64.pb \
  https://essentia.upf.edu/models/music-style-classification/discogs-effnet/discogs-effnet-bs64-1.pb
```

## 3. Configure

```bash
cp .env.example .env
```

Edit `.env`:
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key-here
```

Use the **service_role** key here (not the anon key). The scripts need write access.

## 4. Ingest your music

```bash
python scripts/ingest_music.py /path/to/your/music/folder
```

This scans for audio files, extracts BPM/key/metadata, generates embeddings, and uploads everything to your Supabase. A few hundred tracks takes about 10 minutes.

Useful flags:
- `--dry-run` — see what would happen without writing anything
- `--skip-audio-upload` — don't upload audio files to Supabase Storage (saves time if you're only using local playback)
- `--skip-auto-tag` — skip the LLM auto-tagging step

## 5. Open the website

1. Go to [djmate-frontend.vercel.app](https://djmate-frontend.vercel.app) in **Chrome**
2. Paste your **Project URL** and **anon key** (not the service_role key)
3. Click **Connect**
4. Click **Select Music Folder** and pick the folder with your audio files
5. Your 3D music graph should load

The music folder step lets DJMate play audio straight from your hard drive — nothing gets uploaded. Chrome will ask to re-confirm access when you come back (one click).

---

## Troubleshooting

**No tracks?** Check the ingest script finished without errors. Check your anon key is correct.

**Can't connect?** Make sure you ran `schema.sql` first. The URL should be `https://something.supabase.co` (with the https).

**No audio?** Must be Chrome/Edge/Brave (Firefox and Safari don't support local file access). Make sure you picked the right folder. Click the gear icon in the nav to reconfigure.

**Backend slow on first load?** The shared backend runs on Render's free tier — it sleeps after inactivity. First request takes 30-50 seconds to wake up. Subsequent requests are fast.

## Two keys, two purposes

| Key | Where it goes | What it does |
|-----|--------------|--------------|
| **anon key** | Browser (setup screen) | Read-only-ish, safe to use client-side |
| **service_role key** | `.env` file on your machine | Full database access, used by ingest scripts |

Never put the service_role key in a browser. Never share it.
