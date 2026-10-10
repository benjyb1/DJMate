# Needle (Next Track)

A rekordbox companion. It watches what rekordbox is playing and shows a compass
around it: one track per direction (more energy, darker, faster, more vocal,
deeper, bring it down, slower, closest blend), each one mixable from
where you are.

It's what DJMate opens to. The older Supabase-backed app (sign-in, 3D map,
playlists, tagging) is switched off for now; its code is still in the repo.

## Build plan (as built)

1. **Local library index.** Scan rekordbox's `master.db` (every track whose file
   exists) plus `~/Desktop/Mixing`. Analyse each file once with Discogs-EffNet
   and the Essentia heads. Store the result in
   `~/Library/Application Support/DJMate/next-track/`. Re-runs only touch new or
   changed files.
2. **Direction engine.** Turn the head scores into axes (energy, darkness,
   vocals, depth), rank every track against the library so
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
| More energy / Bring it down | energy | Your 1-10 hand rating where there is one, else a ridge fit of the seven heads + BPM to those ratings (cross-validated r ≈ 0.6) |
| Darker | dark | low *happy*, plus *sad*, *aggressive* and minor key, blended with the mood/theme model's *dark* (provisional, untested) |
| More vocal | vocal | *voice* head (voice vs instrumental) |
| Deeper / More driving | deep | classifier trained on your Discogs styles (Deep House, Deep Techno, Dub Techno) over the EffNet fingerprint |
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
| + outro→intro fit | 38% | 31% | 89% |
| "Deeper" in place of "More electronic" (current) | 31% | 25% | 90% |

"More electronic" was dropped because its head scores ~1.0 for almost every
track, so the card behaved like a second closest blend. The one transition
it caught that "Deeper" doesn't is a move with no change in depth.

Crate affinity (a classifier predicting which Mixing crate a track sounds
like, out-of-fold) was tried and added nothing on top, so it's in the code
but switched off. The sample is small, so these are a direction, not a
verdict. Re-run the script as more sets are logged.

Of the 50 recent transitions, 30 clash keys on the Camelot wheel. Worth
knowing if harmonic mixing matters to you, though rekordbox's key detection
on percussive minimal isn't always reliable.

## Tags and fingerprints (October 2026)

`scripts/bakeoff_embeddings.py` scores fingerprints against Discogs styles
(`scripts/fetch_discogs_tags.py`, 713 of 1,061 tracks matched, 559 with
confident styles). A classifier per fingerprint, 5-fold, three seeds,
macro ROC-AUC over 16 styles:

| Fingerprint | Style AUC | Top style right | "Deep" AUC |
|---|---|---|---|
| EffNet (current) | 0.765 | 55% | 0.74-0.76 |
| EffNet + intro/outro | 0.775 | 56% | 0.75 |
| EffNet + MuQ-MuLan | 0.772 | 56% | 0.77 |
| MuQ-MuLan alone | 0.728 | 44% | 0.75 |
| Jamendo mood/theme (56 labels) | 0.727 | 40% | 0.71 |
| MuQ zero-shot ("<style> music", no training) | 0.607 | 20% | 0.64 |
| Mood/theme "deep" label as is | | | 0.70 |

EffNet stays. MuQ-MuLan adds little and would cost a 4 GB PyTorch model,
2 s a track and a non-commercial licence; its zero-shot tagging is weak on
club music. CLAP (`laion/larger_clap_music`) couldn't be tested: the
checkpoint on Hugging Face is untrained. Dark and vocal have no Discogs
answer key, so they need hand spot-checks.

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
