"""Filesystem locations for Next Track: index storage, models, artwork cache."""
import os
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

EFFNET_NAME = "discogs-effnet-bs64.pb"


def data_dir() -> Path:
    """Where the library index lives. Shared by every checkout and worktree."""
    override = os.getenv("NEXT_TRACK_DATA_DIR")
    base = Path(override) if override else (
        Path.home() / "Library" / "Application Support" / "DJMate" / "next-track"
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


def art_cache_dir() -> Path:
    d = data_dir() / "art"
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir() -> Path:
    return REPO_ROOT / "models"


def _main_checkout(root: Path) -> Optional[Path]:
    marker = f"{os.sep}.claude{os.sep}worktrees{os.sep}"
    s = str(root)
    idx = s.find(marker)
    return Path(s[:idx]) if idx != -1 else None


def effnet_model() -> Optional[Path]:
    """Locate the Discogs-EffNet backbone (18 MB, not checked in)."""
    candidates = [
        Path(os.getenv("ESSENTIA_MODEL_PATH", "")),
        REPO_ROOT / "models" / EFFNET_NAME,
    ]
    main = _main_checkout(REPO_ROOT)
    if main:
        candidates.append(main / "models" / EFFNET_NAME)
    candidates.append(Path.home() / "Desktop" / "OlderFiles" / "Models" / EFFNET_NAME)
    for c in candidates:
        if str(c) and c.is_file():
            return c
    return None


def mixing_folder() -> Path:
    return Path(os.getenv("NEXT_TRACK_MIXING_DIR", str(Path.home() / "Desktop" / "Mixing")))
