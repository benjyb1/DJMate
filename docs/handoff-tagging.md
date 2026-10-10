# Handoff: better tagging for Needle (October 2026)

Needle (formerly DJMate's Next Track) is the local-first rekordbox compass.
Read `docs/next-track.md` first for how the app, index, engine and evaluation
work. This note covers the in-progress tagging work.

## Where things live

- Repo: `~/Desktop/PycharmProjects/KingsCodingClub/DJMate`. Work happens on
  branch `claude/next-track` in the worktree `.claude/worktrees/next-track`.
  `main` is what the installed app runs. Merge to main and push when a piece
  is done (DJMate's rule: always push to main).
- Installed app: `/Applications/Needle.app` (Electron shell in `electron/`,
  rebuild with `cd electron && npm run dist`, see `electron/README.md`). It
  runs the backend from the main checkout with `.venv1` (Essentia env).
  Backend changes need an app restart; frontend changes need a rebuild.
- Index data: `~/Library/Application Support/DJMate/next-track/`
  (`index.json`, `embeddings.npy`, `edges.npz`, `discogs.json`, `emb_*.npz`).
- Discogs token: `DISCOGS_TOKEN` in `DJMate/.env` (git-ignored). Never copy
  it into code, docs or memory.
- `.venv-ml/` (repo root, git-ignored): separate Python 3.12 env with
  PyTorch, `muq`, `transformers` 5.x for the text-and-audio bake-off. Model
  weights are in `~/.cache/huggingface/hub` (CLAP-music 0.78 GB, MuQ-MuLan
  2.65 GB, both fully downloaded).

## Done on this branch (not yet merged)

- **Energy retrained on Benjy's real hand ratings** (`scripts/calib_set.json`,
  129 tracks, 1-10) instead of rekordbox stars. The stars were written by the
  old DJMate model from the same mood scores, so fitting to them was
  circular. Cross-validated r = 0.59 vs 0.51 before. New ratings can go in
  `energy_labels.json` in the data dir. Energy is stored 0-5 for the UI bars.
- **Essentia mood/theme head** (`models/mtg_jamendo_moodtheme-*`, 56 labels)
  wired into the analyser (`mode="moods"` for existing tracks) and the index
  (`e["moods"]`, a provisional `dark` blend and a new `deep` axis). First look:
  weak on club music (a dark jungle classic scored 0.04 "dark"; labels like
  "summer"/"corporate" rank high). "Deep" looks more sensible. Don't let it
  steer dials until the bake-off says it helps.
- `scripts/fetch_discogs_tags.py`: Discogs metadata lookup per library track
  (artist+album, then artist+title, then catno with artist check). Resumable,
  records misses. About 69% matched.
- `scripts/bakeoff_embeddings.py`: scores fingerprints by predicting held-out
  tracks' Discogs styles (5-fold, macro ROC-AUC, hit@1/@3). Trial on 261
  tracks: EffNet 0.748 AUC, EffNet+intro/outro 0.762, heads only 0.641,
  majority baseline hit@1 0.39 vs EffNet 0.56.
- `scripts/embed_text_audio.py`: embeds the library with MuQ-MuLan or CLAP
  (3 x 10 s clips per track) plus a word vocabulary, into `emb_<model>.npz`.

## In progress when handed off

1. `fetch_discogs_tags.py` **finished**: 713 of 1,061 tracks matched
   (`discogs.json`). Re-run with `--retry-missing` only if matching improves.
2. The mood pass (`scripts/build_next_track_index.py`) was at 485/1076. It
   also resumes (tracks without `moods` get `mode="moods"`).
   Check whether either is still running before restarting (`pgrep -fl
   fetch_discogs_tags`, `pgrep -fl build_next_track_index`).

## Open problems

- **CLAP similarities are ~0.00 for every word.** In transformers 5,
  `get_text_features` returns an object whose `pooler_output` equals the
  forward pass's `text_embeds` (verified). The audio side probably isn't the
  projected embedding: check `get_audio_features(...).pooler_output` against
  `model(**inputs).audio_embeds` and use the projected, normalised one.
- **MuQ-MuLan fails under transformers 5** (`'EasyDict' object has no
  attribute '_attn_implementation'`). Pin `transformers<5` in `.venv-ml`
  (e.g. `pip install "transformers>=4.40,<5"`), then re-test CLAP too.
- `bakeoff_embeddings.py` doesn't yet load `emb_muq.npz` / `emb_clap.npz`.
  Add them as feature sets, plus a zero-shot score (text embedding of
  "<style> music" vs audio, no training) using the stored word embeddings.
- The `deep` axis exists in the index but has no direction in
  `backend/next_track/engine.py` (`DIRECTIONS`, `_EXTREMES`, `_reasons`) and
  no UI entry (`Frontend/src/components/next/directions.jsx`
  `DIRECTION_META`, `parts.jsx` `AxisBars`). The plan was to replace the dead
  "More electronic" default (that head says ~1.0 for 97% of tracks) with
  "Deeper" or a Discogs-style dial, once the bake-off picks the signal.

## Agreed plan with Benjy

- Discogs is the answer key and source of style words (official API, metadata
  only, no audio downloads). Bandcamp and SoundCloud: not used (terms / paid
  API). Vocal separation: dropped. allin1 is kept for future automated
  mixing; try rekordbox phrase analysis first (it's switched off in his
  library; he needs to check Preferences > Analysis).
- Pick the fingerprint by bake-off number, not opinion. Candidates: EffNet
  (current), EffNet+intro/outro, MuQ-MuLan, CLAP, optionally Essentia MAEST.
  Keep the winner's model in Needle's analyser.
- Then rebuild the dials (dark, deep/style, vocal) on the winning signal and
  ask Benjy for ~50 spot-checks to report real accuracy.

## Checks before calling anything done

- `.venv1/bin/python -m unittest discover tests`
- `.venv1/bin/python scripts/eval_next_track.py --variants` (mixing rules +
  taste check; don't let harmonic share or empty cards get worse)
- Restart `/Applications/Needle.app` and look at the compass.
