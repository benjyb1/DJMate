"""
write_rekordbox_stars.py — write energy-derived star ratings into rekordbox for
every track under ~/Desktop/Mixing EXCEPT anything in 'Older Selections'.

  python scripts/write_rekordbox_stars.py --dry-run   # show plan, no writes
  python scripts/write_rekordbox_stars.py             # back up + write

rekordbox MUST be quit (it locks master.db).
"""
import sys, os, json, shutil, time, collections

HERE = os.path.dirname(__file__)
CACHE = json.load(open(os.path.join(HERE, "energy_cache.json")))
MIX = os.path.expanduser("~/Desktop/Mixing")
EXCLUDE = "/Older Selections/"
W = {"party": 0.40, "aggressive": 0.35, "relaxed": 0.25}
DRY = "--dry-run" in sys.argv


def energy(s):
    return 1 + 9 * (W["party"] * s["party"] + W["aggressive"] * s["aggressive"] + W["relaxed"] * (1 - s["relaxed"]))


def stars(e):
    return 1 if e <= 3.5 else 2 if e <= 5.5 else 3 if e <= 7.0 else 4 if e <= 8.0 else 5


def want(fp):
    return fp and fp.startswith(MIX) and EXCLUDE not in fp and fp in CACHE and "error" not in CACHE[fp]


from pyrekordbox.config import get_config
from pyrekordbox import Rekordbox6Database

if not DRY:
    dbp = str(get_config("rekordbox6", "db_path"))
    bak = dbp + ".bak-energy-" + time.strftime("%Y%m%d-%H%M%S")
    shutil.copy2(dbp, bak)
    print("backup ->", os.path.basename(bak))

db = Rekordbox6Database()
dist = collections.Counter()
changed = skipped_old = skipped_uncached = unchanged = 0
for c in db.get_content():
    fp = c.FolderPath or ""
    if not (fp and fp.startswith(MIX)):
        continue
    if EXCLUDE in fp:
        skipped_old += 1; continue
    if fp not in CACHE or "error" in CACHE.get(fp, {"error": 1}):
        skipped_uncached += 1; continue
    st = stars(energy(CACHE[fp]))
    dist[st] += 1
    if c.Rating != st:
        if not DRY:
            c.Rating = st
        changed += 1
    else:
        unchanged += 1

if not DRY:
    db.commit()
print(("DRY-RUN " if DRY else "") + f"plan: {changed} to change, {unchanged} already correct")
print(f"skipped: {skipped_old} in Older Selections, {skipped_uncached} not scored")
print("star distribution of written set:", {f"{k}★": dist[k] for k in sorted(dist)})
if not DRY:
    print("committed.")
