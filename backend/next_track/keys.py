"""Musical key normalisation and Camelot-wheel compatibility.

rekordbox stores keys in whatever notation the library was analysed with, so a
single library mixes "F#m", "8A" and Open Key's "3m". Everything is normalised
to Camelot (number 1-12 + letter A/B) before comparing.
"""
from __future__ import annotations

import re
from typing import Optional

# Musical key -> Camelot. Minor keys are 'A', major keys 'B'.
_CAMELOT = {
    "G#m": "1A", "Abm": "1A", "D#m": "2A", "Ebm": "2A", "A#m": "3A", "Bbm": "3A",
    "Fm": "4A", "Cm": "5A", "Gm": "6A", "Dm": "7A", "Am": "8A", "Em": "9A",
    "Bm": "10A", "F#m": "11A", "Gbm": "11A", "C#m": "12A", "Dbm": "12A",
    "B": "1B", "F#": "2B", "Gb": "2B", "C#": "3B", "Db": "3B", "G#": "4B",
    "Ab": "4B", "D#": "5B", "Eb": "5B", "A#": "6B", "Bb": "6B", "F": "7B",
    "C": "8B", "G": "9B", "D": "10B", "A": "11B", "E": "12B",
}

_CAMELOT_RE = re.compile(r"^(\d{1,2})\s*([ABab])$")
_OPENKEY_RE = re.compile(r"^(\d{1,2})\s*([mdMD])$")


def to_camelot(key: Optional[str]) -> Optional[str]:
    """Return a Camelot code like '8A', or None if the key can't be read."""
    if not key:
        return None
    k = key.strip()
    m = _CAMELOT_RE.match(k)
    if m and 1 <= int(m.group(1)) <= 12:
        return f"{int(m.group(1))}{m.group(2).upper()}"
    m = _OPENKEY_RE.match(k)
    if m and 1 <= int(m.group(1)) <= 12:
        # Open Key 1m (A minor) is Camelot 8A; the wheel is offset by seven.
        num = (int(m.group(1)) + 6) % 12 + 1
        return f"{num}{'A' if m.group(2).lower() == 'm' else 'B'}"
    k = k.replace("min", "m").replace("maj", "").replace(" ", "")
    if k.endswith("minor"):
        k = k[:-5] + "m"
    if k.endswith("major"):
        k = k[:-5]
    if len(k) >= 1:
        k = k[0].upper() + k[1:]
    return _CAMELOT.get(k)


def _parse(code: Optional[str]):
    if not code:
        return None
    m = _CAMELOT_RE.match(code)
    return (int(m.group(1)), m.group(2).upper()) if m else None


def compatibility(a: Optional[str], b: Optional[str]) -> float:
    """How well two Camelot keys mix, 0-1. Unknown keys score a neutral 0.5."""
    pa, pb = _parse(a), _parse(b)
    if not pa or not pb:
        return 0.5
    (na, la), (nb, lb) = pa, pb
    diff = min(abs(na - nb), 12 - abs(na - nb))
    if diff == 0 and la == lb:
        return 1.0
    if diff == 0:
        return 0.85          # relative major/minor
    if diff == 1 and la == lb:
        return 0.9           # adjacent on the wheel
    if diff == 2 and la == lb:
        return 0.55          # energy-boost jump
    if diff == 1:
        return 0.5           # diagonal move
    return 0.15


def is_minor(code: Optional[str]) -> Optional[bool]:
    p = _parse(code)
    return None if p is None else p[1] == "A"
