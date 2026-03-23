# DJMate Setup Guide

Set up your own DJMate instance with your personal music library. You'll need a Supabase account and a copy of this repo.

## Prerequisites

- Python 3.10+
- Chrome, Edge, or Brave (for local audio playback)
- A Supabase account (free tier works)

## Step 1: Create a Supabase Project

1. Go to [supabase.com](https://supabase.com) and sign in (or create an account)
2. Click **New Project**
3. Pick a name (e.g. "djmate"), set a database password, choose a region close to you
4. Wait for it to finish provisioning (~1 minute)

## Step 2: Set Up the Database Schema

1. In your Supabase dashboard, go to **SQL Editor**
2. Open the `schema.sql` file from this repo
3. Paste the entire contents into the editor
4. Click **Run**

This creates the tables, indexes, and functions DJMate needs.

## Step 3: Create a Storage Bucket for Album Art

1. In Supabase, go to **Storage**
2. Click **New Bucket**
3. Name it `album-covers`
4. Set it to **Public**
5. Click **Create**

(Audio files stay on your local machine — no need to upload them.)

## Step 4: Get Your API Keys

1. Go to **Project Settings** > **API**
2. You need two keys:
   - **Project URL** — looks like `https://abcdefg.supabase.co`
   - **anon / public key** — the long JWT token under "Project API keys"

You'll also need the **service_role key** for the analysis scripts (keep this secret, never share it).

## Step 5: Analyse Your Music

1. Clone this repo:
   ```bash
   git clone https://github.com/benjyb1/DJMate.git
   cd DJMate
   ```

2. Copy the environment template and fill in your keys:
   ```bash
   cp .env.example .env
   ```

   Edit `.env`:
   ```
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-service-role-key-here
   ```

   **Important:** Use the **service_role** key here (not the anon key). The analysis scripts need write access.

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the analysis pipeline on your music folder:
   ```bash
   python scripts/ingest_music.py /path/to/your/music
   ```

   This will:
   - Scan for audio files (MP3, FLAC, WAV, etc.)
   - Extract metadata (BPM, key, title, artist)
   - Generate embeddings for similarity search
   - Upload everything to your Supabase database

   Depending on your library size, this can take a while. A few hundred tracks usually finishes in under 10 minutes.

## Step 6: Connect to the Website

1. Open [DJMate](https://djmate-frontend.vercel.app) in Chrome
2. You'll see a setup screen asking for your Supabase credentials
3. Paste your **Project URL** and **anon key** (NOT the service_role key)
4. Click **Connect**
5. Grant access to your music folder when prompted (this lets DJMate play audio directly from your computer)
6. You're in — your 3D music graph should load with all your tracks

## Troubleshooting

**No tracks showing up?**
- Check that the analysis script completed without errors
- Verify your Supabase URL and anon key are correct (not the service_role key)
- Check the browser console for errors (F12 > Console)

**Audio not playing?**
- Make sure you're using Chrome, Edge, or Brave (Firefox and Safari don't support the File System Access API)
- Make sure you granted access to the correct music folder (the one containing your audio files)
- Try clicking the settings gear in the nav bar to reconfigure your music folder

**"Connection failed" on the setup screen?**
- Double-check your Supabase URL format: `https://your-project.supabase.co`
- Make sure you ran `schema.sql` in the SQL editor first
- Check that your anon key is the full JWT token (it's quite long)

## Key Concepts

**anon key vs service_role key:**
- **anon key** (public): Safe to use in the browser. Limited by Row Level Security policies. Used when connecting via the website.
- **service_role key** (secret): Full database access, bypasses RLS. Used only in the analysis scripts running on your machine. Never share this or put it in a browser.

**Where does my data live?**
- Your music files stay on your computer. DJMate plays them directly from your disc using the File System Access API.
- Track metadata, embeddings, and coordinates live in your Supabase database.
- Album art (if uploaded) lives in your Supabase Storage bucket.
- The DJMate website and backend are shared infrastructure — they don't store any of your data.
