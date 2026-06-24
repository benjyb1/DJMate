"""
library_stars.py — apply the energy blend + Benjy's star thresholds across the
whole library, and report the distribution per playlist.

energy = 1 + 9 * (0.40*party + 0.35*aggressive + 0.25*(1-relaxed))   [danceability dropped: it saturates]
stars  : <=3.5 ->1, <=5.5 ->2, <=7 ->3, <=8 ->4, else 5
"""
import os, json, collections

HERE = os.path.dirname(__file__)
CACHE = json.load(open(os.path.join(HERE, "energy_cache.json")))
TRACKS = json.load(open(os.path.join(HERE, "all_tracks.json")))
W = {"party": 0.40, "aggressive": 0.35, "relaxed": 0.25}


def energy(s):
    e01 = W["party"] * s["party"] + W["aggressive"] * s["aggressive"] + W["relaxed"] * (1 - s["relaxed"])
    return round(1 + 9 * e01, 2)


def stars(e):
    if e <= 3.5: return 1
    if e <= 5.5: return 2
    if e <= 7.0: return 3
    if e <= 8.0: return 4
    return 5


out = []
for t in TRACKS:
    fp = t.get("filepath")
    s = CACHE.get(fp)
    if not s or "error" in s:
        continue
    e = energy(s)
    pl = os.path.basename(os.path.dirname(fp))
    out.append({"filepath": fp, "title": t.get("title"), "playlist": pl,
                "energy": e, "stars": stars(e)})

json.dump(out, open(os.path.join(HERE, "library_energy.json"), "w"), indent=2)
print(f"scored {len(out)} tracks\n")

dist = collections.Counter(r["stars"] for r in out)
tot = len(out)
print("library-wide star distribution:")
for s in range(1, 6):
    n = dist[s]; bar = "█" * round(40 * n / tot)
    print(f"  {s}★ {n:4d} ({100*n/tot:4.1f}%) {bar}")

es = sorted(r["energy"] for r in out)
print(f"\nenergy: min {es[0]} / median {es[len(es)//2]} / max {es[-1]}")

print("\nper-playlist mean energy (sorted):")
byp = collections.defaultdict(list)
for r in out:
    byp[r["playlist"]].append(r["energy"])
for pl, vals in sorted(byp.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
    m = sum(vals) / len(vals)
    sd = collections.Counter(stars(v) for v in vals)
    spread = "".join(f"{sd[s]}×{s}★ " for s in range(1, 6) if sd[s])
    print(f"  {pl[:22]:22s} n={len(vals):3d} mean E={m:4.1f}   {spread}")
