# Needle (Next Track)

A rekordbox companion. It watches what rekordbox is playing and shows a compass
around it: one track per direction (more energy, darker, faster, more vocal,
more electronic, bring it down, slower, closest blend), each one mixable from
where you are.

It's what DJMate opens to. The older Supabase-backed app (sign-in, 3D map,
playlists, tagging) is switched off for now; its code is still in the repo.

## Build plan (as built)

1. **Local library index.** Scan rekordbox's `master.db` (every track whose file
   exists) plus `~/Desktop/Mixing`. Analyse each file once with Discogs-EffNet
   and the Essentia heads. Store the result in
   `~/Library/Application Support/DJMate/next-track/`. Re-runs only touch new or
   changed files.
2. **Direction engine.** Turn the head scores into four axes (energy,
   darkness, vocals, electronic), rank every track against the library so
   "darker" is relative to your music, then pick the closest mixable track that
   moves along each axis.
3. **Now playing.** Read rekordbox's play history (`djmdSongHistory`). The
   newest row is on air and the rest of that session counts as played tonight.
4. **Tracks that aren't yours.** Files rekordbox plays that aren't indexed get
   analysed on the fly. Dropped files and 15 s mic recordings go through the
   same analyser. If a clip is clearly one of your tracks, the compass jumps to
   your copy.
5. **Screen.** A compass with the track on air in the middle, suggestions on
   eight sides joined by animated arrows, alternates on hover, and click to
   step into a pick. It follows rekordbox whenever rekordbox moves on.

## How each direction is chosen

Candidates are first filtered to what's mixable:

- within ±6% BPM (±9% if nothing else qualifies),
- not on air and not already played this session,
- not a duplicate copy of the same record.

Each direction then scores what's left:

| Direction | Axis | Rule |
|---|---|---|
| More energy / Bring it down | energy | Your rekordbox star rating where you've rated, else a ridge fit of party, aggressive and relaxed heads + BPM to those ratings |
| Darker | dark | low *happy*, plus *sad*, *aggressive* and minor key |
| More vocal | vocal | *voice* head (voice vs instrumental) |
| More electronic | electronic | *mood_electronic* head |
| Faster / Slower | BPM | +/−1.5 to 10 BPM, at most 9% |
| Closest blend | none | nearest by embedding |

For the axis directions the score is 45% fit, 30% how far the track moves
along the axis (saturating at 35 percentile points so it's a step, not a
leap), 15% Camelot key compatibility and 10% tempo closeness. Harmonically
safe picks (compatibility ≥ 0.5) are preferred when there are enough of them.

**Fit** blends two signals, each ranked within the candidate pool: whole-track
similarity (EffNet embedding of the middle four minutes, standardised,
cosine) and how this track's last minute sounds against each candidate's
first minute. Half- and double-time matches (87 under 174) count as in tempo,
with a small penalty so they never beat a straight match.

## How it's evaluated

`scripts/eval_next_track.py` scores any settings two ways:

- **Mixing rules (the main judge).** From 250 random starting tracks, what's
  on screen: share of harmonic picks, tempo jumps, empty cards, directions
  that go the wrong way.
- **Taste check (a sanity check, not a target).** Consecutive plays from
  rekordbox sessions since February 2026 that follow the mixing rules (tempo
  within 6%, compatible key). Centred on the first track, is the one you
  actually played on screen? Older sessions aren't used, and transitions that
  break the rules aren't something to copy.

October 2026 results (1,061 tracks; 50 recent transitions, 16 clean):

| Settings | On screen | Top 3 | Harmonic picks |
|---|---|---|---|
| Whole-track similarity only | 25% | 19% | 88% |
| + outro→intro fit (current) | 38% | 31% | 89% |

Crate affinity (a classifier predicting which Mixing crate a track sounds
like, out-of-fold) was tried and added nothing on top, so it's in the code
but switched off. The sample is small, so these are a direction, not a
verdict. Re-run the script as more sets are logged.

Of the 50 recent transitions, 30 clash keys on the Camelot wheel. Worth
knowing if harmonic mixing matters to you, though rekordbox's key detection
on percussive minimal isn't always reliable.

## Running it

- **Desktop app:** `cd electron && npm start`. The backend mounts `/next` itself.
- **First run:** press *Analyse my library*, or run
  `.venv1/bin/python scripts/build_next_track_index.py`. That's roughly 1,000
  tracks at 1–2 s each across three workers. Re-run it after adding music.
- **Models:** EffNet backbone at `ESSENTIA_MODEL_PATH` or the usual
  `~/Desktop/OlderFiles/Models/discogs-effnet-bs64.pb`; the heads are in
  `models/`.

## Analysis

ffmpeg seeks to just the parts that are analysed (middle four minutes, first
and last minute), with the downmix matched to Essentia's so fingerprints stay
comparable. A track on air that isn't in the library uses a quicker pass (the
middle two minutes plus the outro). Without ffmpeg it falls back to decoding
whole files.

Tests: `.venv1/bin/python -m unittest discover tests`.

## Known limits

- rekordbox writes a history row when a track is *played*, not when it's
  loaded, so the compass updates as you bring the new track in rather than
  while it's cued.
- The mic needs macOS permission the first time. Identification was
  calibrated on simulated room recordings, not a real club.
- On Render/Vercel there's no rekordbox or audio, so `/next` reports itself
  unavailable and the tab says to use the desktop app.
- rekordbox phrase analysis (intro/outro/chorus markers) isn't switched on in
  this library. With it, intro and outro could come from the real sections
  instead of the first and last minute.
