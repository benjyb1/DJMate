"""Shared utilities for DJMate scripts."""
import re
import unicodedata


def make_cover_filename(artist: str, title: str, ext: str = ".jpg") -> str:
    """Create a safe, normalized filename for album cover storage.

    Strips diacritics (e->e, u->u, n->n, etc.) then removes anything that isn't
    alphanumeric, a hyphen, or underscore.

    Must match the logic in Frontend/src/utils/coverUrl.js for URL generation.
    """
    raw = f"{artist}_{title}"
    # 1. NFKD decompose (e.g. e-acute -> e + combining accent)
    # 2. Drop all non-ASCII (keeping only base letters)
    raw = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    # 3. Lowercase
    raw = raw.lower()
    # 4. Replace non-alnum (except - _) with _
    raw = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in raw)
    # 5. Collapse consecutive underscores, truncate
    raw = "_".join(filter(None, raw.split("_")))
    name = raw[:150]
    if ext:
        return name + ext
    return name


def get_supabase():
    """Create and return a Supabase client using env vars."""
    from dotenv import load_dotenv
    import os
    from supabase import create_client

    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    return create_client(url, key)
