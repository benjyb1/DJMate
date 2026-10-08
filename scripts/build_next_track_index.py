"""
build_next_track_index.py — analyse the library for the Next Track screen.

Scans rekordbox (every track whose file exists) plus ~/Desktop/Mixing, then runs
EffNet + the classification heads on anything new or changed. Safe to re-run:
already-analysed tracks are skipped, so after adding music this only does the
new files. The app's "Scan for new tracks" button does the same thing.

Run with the Essentia virtualenv:
    .venv1/bin/python scripts/build_next_track_index.py
"""
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.next_track.index import LibraryIndex  # noqa: E402
from backend.next_track import paths  # noqa: E402


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    idx = LibraryIndex()
    print(f"Index folder: {paths.data_dir()}")
    idx.start_build()
    last = None
    while True:
        st = idx.build_state
        line = f"{st['done']}/{st['total']}  failed={st['failed']}  {st['current'] or ''}"
        if line != last:
            print(line[:140], flush=True)
            last = line
        if not st["running"] and st["finished_at"]:
            break
        time.sleep(2)
    if st["error"]:
        print(f"Stopped with error: {st['error']}")
        sys.exit(1)
    s = idx.status()
    print(f"Done: {s['analysed']} analysed of {s['tracks']} tracks. "
          f"Energy model: {s['energy_model']}")
    idx.analyser.shutdown()


if __name__ == "__main__":
    main()
