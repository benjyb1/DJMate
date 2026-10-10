"""
embed_text_audio.py — fingerprint the library with text-and-audio models.

These models put music and words in the same space, so a phrase like
"dark rolling dub techno" can be scored against any track. Two candidates:

  muq   MuQ-MuLan large (Tencent, CC BY-NC 4.0), 24 kHz audio
  clap  LAION CLAP trained on music (Apache-2.0), 48 kHz audio. Not usable:
        the laion/larger_clap_music checkpoint on Hugging Face is untrained
        (every bias is zero, weights at their 0.02 init spread), so all text
        embeddings come out identical. Checked October 2026.

Each track is summarised by three 10-second clips (25%, 50%, 75% of the way
through), embedded and averaged. Also embeds a vocabulary of words (Discogs
styles and descriptive terms) so the bake-off can test zero-shot tagging.

Runs in the separate .venv-ml environment (PyTorch, muq, transformers):
    .venv-ml/bin/python scripts/embed_text_audio.py muq
    .venv-ml/bin/python scripts/embed_text_audio.py clap
Output: ~/Library/Application Support/DJMate/next-track/emb_<model>.npz
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DATA = Path(os.getenv("NEXT_TRACK_DATA_DIR", str(
    Path.home() / "Library" / "Application Support" / "DJMate" / "next-track")))
CLIP_SECONDS = 10
CLIP_POSITIONS = (0.25, 0.5, 0.75)
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"

WORDS = [
    # Discogs styles common in this library
    "techno", "house", "deep house", "tech house", "minimal", "minimal techno",
    "dub techno", "acid", "acid house", "electro", "breakbeat", "uk garage",
    "downtempo", "ambient", "drum and bass", "jungle", "disco", "trance",
    "progressive house", "breaks", "leftfield", "idm", "experimental", "dub",
    "nu-disco", "deep techno", "bass music",
    # descriptive words for dials
    "dark", "deep", "hypnotic", "energetic", "melodic", "vocal", "instrumental",
    "dubby", "rolling", "groovy", "uplifting", "aggressive", "minimal and stripped back",
    "warm", "euphoric", "industrial", "atmospheric", "funky", "percussive",
]


def probe(path):
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "a:0",
                          "-show_entries", "stream=channels:format=duration",
                          "-of", "default=nw=1:nk=1", path],
                         capture_output=True, text=True, timeout=30)
    vals = [v for v in out.stdout.split() if v and v != "N/A"]
    return int(vals[0]), float(vals[1])


def clip(path, start, seconds, sr, channels):
    mix = [] if channels == 1 else [
        "-af", "pan=mono|c0=" + "+".join(f"{1 / channels:.6f}*c{i}" for i in range(channels))]
    p = subprocess.run([FFMPEG, "-v", "error", "-nostdin", "-ss", f"{start:.2f}",
                        "-t", f"{seconds}", "-i", path, "-vn", *mix, "-ac", "1",
                        "-ar", str(sr), "-f", "f32le", "-"], capture_output=True, timeout=120)
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


class Muq:
    sr = 24000

    def __init__(self, device):
        import torch
        from muq import MuQMuLan
        self.torch = torch
        self.device = device
        self.m = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large").to(device).eval()

    def audio(self, clips):
        with self.torch.no_grad():
            x = self.torch.tensor(np.stack(clips)).to(self.device)
            return self.m(wavs=x).float().cpu().numpy()

    def text(self, words):
        with self.torch.no_grad():
            return self.m(texts=[f"{w} music" for w in words]).float().cpu().numpy()


def _vec(out):
    """transformers 5 returns an output object (projected embedding in
    pooler_output); older versions returned the tensor itself."""
    t = getattr(out, "pooler_output", out)
    return t.float().cpu().numpy()


class Clap:
    sr = 48000

    def __init__(self, device):
        import torch
        from transformers import ClapModel, ClapProcessor
        self.torch = torch
        self.device = device
        self.m = ClapModel.from_pretrained("laion/larger_clap_music").to(device).eval()
        self.p = ClapProcessor.from_pretrained("laion/larger_clap_music")

    def audio(self, clips):
        with self.torch.no_grad():
            try:
                inp = self.p(audio=list(clips), sampling_rate=self.sr, return_tensors="pt")
            except TypeError:
                inp = self.p(audios=list(clips), sampling_rate=self.sr, return_tensors="pt")
            inp = {k: v.to(self.device) for k, v in inp.items()}
            return _vec(self.m.get_audio_features(**inp))

    def text(self, words):
        with self.torch.no_grad():
            inp = self.p(text=[f"{w} music" for w in words], return_tensors="pt", padding=True)
            inp = {k: v.to(self.device) for k, v in inp.items()}
            return _vec(self.m.get_text_features(**inp))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "muq"
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = {"muq": Muq, "clap": Clap}[which](device)
    print(f"{which} loaded on {device}", flush=True)

    meta = json.loads((DATA / "index.json").read_text())
    analysed = set(meta.get("emb_ids", []))
    tracks = [e for e in meta["entries"] if e["id"] in analysed and "/PioneerDJ/" not in e["path"]]

    out_p = DATA / f"emb_{which}.npz"
    done = {}
    if out_p.exists():
        z = np.load(out_p, allow_pickle=False)
        done = {str(i): v for i, v in zip(z["ids"], z["emb"])}
    todo = [e for e in tracks if e["id"] not in done]
    print(f"{len(tracks)} tracks, {len(todo)} to embed", flush=True)

    def save():
        ids = list(done)
        np.savez(out_p.with_suffix(".tmp.npz"), ids=np.array(ids),
                 emb=np.stack([done[i] for i in ids]).astype(np.float32),
                 words=np.array(WORDS), word_emb=model.text(WORDS).astype(np.float32))
        os.replace(out_p.with_suffix(".tmp.npz"), out_p)

    t0 = time.time()
    for n, e in enumerate(todo, 1):
        try:
            ch, dur = probe(e["path"])
            clips = []
            for pos in CLIP_POSITIONS:
                start = max(0.0, min(dur - CLIP_SECONDS, dur * pos - CLIP_SECONDS / 2))
                c = clip(e["path"], start, CLIP_SECONDS, model.sr, ch)
                need = CLIP_SECONDS * model.sr
                if len(c) >= model.sr * 3:
                    clips.append(np.pad(c, (0, max(0, need - len(c))))[:need])
            if clips:
                v = model.audio(clips).mean(axis=0)
                done[e["id"]] = v / (np.linalg.norm(v) + 1e-9)
        except Exception as exc:
            print(f"  skipped {e.get('title')}: {exc}", flush=True)
        if n % 25 == 0 or n == len(todo):
            save()
            rate = (time.time() - t0) / n
            print(f"{n}/{len(todo)}  {rate:.1f}s/track  ~{rate * (len(todo) - n) / 60:.0f} min left",
                  flush=True)
    if not todo:
        save()
    print(f"Done: {len(done)} tracks embedded with {which}.")


if __name__ == "__main__":
    main()
