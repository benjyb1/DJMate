"""
fetch_discogs_tags.py — look up each library track on Discogs and keep its
genres and styles (metadata only, no audio).

Matching, most reliable first:
  1. artist + release title (rekordbox's album field)
  2. artist + track title (free-text search)
  3. catalogue number from the filename, e.g. [RS032], with the artist checked
A result is only accepted when its artist matches ours (or it's a compilation
whose title matches the album exactly). Misses are recorded so re-runs skip
them; pass --retry-missing to try them again.

Needs DISCOGS_TOKEN in the repo's .env. Output:
  ~/Library/Application Support/DJMate/next-track/discogs.json

    .venv1/bin/python scripts/fetch_discogs_tags.py
"""
import argparse
import json
import os
import re
import ssl
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from backend.next_track import paths  # noqa: E402
from backend.next_track.index import LibraryIndex  # noqa: E402

API = "https://api.discogs.com/database/search"
USER_AGENT = "Needle/1.0 +https://github.com/benjyb1/DJMate"
MIN_INTERVAL = 1.1          # seconds between requests (Discogs allows 60/min)
# python.org's macOS Python ships without trusted CA certificates.
try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()
CATNO_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9 .\-]{1,12}\d{1,4}[A-Za-z]?)\]")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", s)        # bracketed extras
    s = re.sub(r"\b(feat|ft|featuring)\b.*", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def artist_names(s: str) -> set[str]:
    parts = re.split(r",|&| x | and |/|;| vs\.? ", (s or "").lower())
    return {norm(p) for p in parts if norm(p)}


class Discogs:
    def __init__(self, token: str):
        self.token = token
        self._last = 0.0

    def search(self, **params) -> list[dict]:
        wait = MIN_INTERVAL - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        params = {k: v for k, v in params.items() if v}
        params.update(type="release", per_page=10)
        req = urllib.request.Request(
            f"{API}?{urllib.parse.urlencode(params)}",
            headers={"Authorization": f"Discogs token={self.token}", "User-Agent": USER_AGENT})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=30, context=SSL_CONTEXT) as r:
                    self._last = time.time()
                    remaining = int(r.headers.get("X-Discogs-Ratelimit-Remaining", "60"))
                    if remaining < 3:
                        time.sleep(10)
                    return json.load(r).get("results", [])
            except urllib.error.HTTPError as e:
                self._last = time.time()
                if e.code == 429:
                    time.sleep(15 * (attempt + 1))
                    continue
                if e.code in (401, 403):
                    raise SystemExit("Discogs rejected the token (check DISCOGS_TOKEN in .env)")
                return []
            except urllib.error.URLError as e:
                if isinstance(e.reason, ssl.SSLError):
                    raise SystemExit(f"Can't make a secure connection to Discogs: {e.reason}")
                time.sleep(3 * (attempt + 1))
            except Exception:
                time.sleep(3 * (attempt + 1))
        return []


def score(result: dict, artist: str, album: str, title: str) -> float:
    """0..1: how sure are we this release is our track's release?"""
    full = result.get("title") or ""
    r_artist, _, r_title = full.partition(" - ")
    ours = artist_names(artist)
    theirs = artist_names(r_artist)
    artist_ok = bool(ours & theirs) or any(
        SequenceMatcher(None, a, b).ratio() > 0.85 for a in ours for b in theirs)
    album_sim = SequenceMatcher(None, norm(album), norm(r_title)).ratio() if album else 0.0
    if not artist_ok:
        # Compilations ("Various") only count if the album title matches.
        if norm(r_artist) == "various" and album_sim > 0.9:
            return 0.6
        return 0.0
    s = 0.7 + 0.25 * album_sim
    if norm(title) and norm(title) in norm(r_title):
        s += 0.05
    return min(s, 1.0)


def lookup(api: Discogs, e: dict) -> dict:
    artist, title, album = e.get("artist") or "", e.get("title") or "", e.get("album") or ""
    fname = Path(e.get("path", "")).name
    m = CATNO_RE.search(f"{title} {fname}")
    catno = m.group(1) if m else None
    tries = []
    if album and norm(album) != norm(title):
        tries.append(("artist+album", dict(artist=artist, release_title=album)))
    tries.append(("artist+title", dict(q=f"{artist} {norm(title)}")))
    if catno:
        tries.append(("catno", dict(catno=catno)))

    best, best_s, best_how = None, 0.0, None
    for how, params in tries:
        for r in api.search(**params):
            s = score(r, artist, album, title)
            if s > best_s:
                best, best_s, best_how = r, s, how
        if best_s >= 0.9:
            break
    if best is None or best_s < 0.7:
        return {"status": "not_found", "tried": [h for h, _ in tries]}
    return {
        "status": "ok", "how": best_how, "confidence": round(best_s, 2),
        "release_id": best.get("id"), "title": best.get("title"),
        "year": best.get("year"), "label": (best.get("label") or [None])[0],
        "catno": best.get("catno"), "country": best.get("country"),
        "genres": best.get("genre") or [], "styles": best.get("style") or [],
        "have": (best.get("community") or {}).get("have"),
        "want": (best.get("community") or {}).get("want"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retry-missing", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    load_dotenv(repo / ".env")
    if not os.getenv("DISCOGS_TOKEN"):
        # A worktree has no .env of its own; use the main checkout's.
        s = str(repo)
        i = s.find("/.claude/worktrees/")
        if i != -1:
            load_dotenv(Path(s[:i]) / ".env")
    token = os.getenv("DISCOGS_TOKEN")
    if not token:
        raise SystemExit("DISCOGS_TOKEN is not set in .env")

    out_p = paths.data_dir() / "discogs.json"
    found = json.loads(out_p.read_text()) if out_p.exists() else {}
    idx = LibraryIndex()
    todo = [idx.entries[i] for i in idx.lib_ids
            if i not in found or (args.retry_missing and found[i].get("status") != "ok")]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(idx.lib_ids)} tracks, {len(todo)} to look up "
          f"(about {len(todo) * 2 * MIN_INTERVAL / 60:.0f} min)", flush=True)

    api = Discogs(token)
    hits = 0
    for n, e in enumerate(todo, 1):
        res = lookup(api, e)
        res["artist"], res["track"] = e.get("artist"), e.get("title")
        found[e["id"]] = res
        hits += res["status"] == "ok"
        if n % 10 == 0 or n == len(todo):
            tmp = out_p.with_suffix(".tmp")
            tmp.write_text(json.dumps(found, indent=1))
            os.replace(tmp, out_p)
            print(f"{n}/{len(todo)}  matched {hits}  ({e.get('artist')} - {e.get('title')})", flush=True)
    ok = sum(1 for v in found.values() if v.get("status") == "ok")
    print(f"Done. {ok} of {len(found)} tracks matched on Discogs.")


if __name__ == "__main__":
    main()
