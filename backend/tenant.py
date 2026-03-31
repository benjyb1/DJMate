"""Per-tenant Supabase client management for BYOS (Bring Your Own Supabase)."""
import os
import re
import time
import hashlib
import logging
from urllib.parse import urlparse
from fastapi import Request, HTTPException
from backend.data.db_interface import DatabaseManager

logger = logging.getLogger(__name__)

# Cache keyed on hash(url + key) so a key rotation takes effect immediately.
_tenant_cache: dict[str, tuple[DatabaseManager, float]] = {}
_CACHE_TTL = 600   # 10 minutes
_MAX_TENANTS = 100  # cap to prevent memory exhaustion

# The central DJMate Supabase host.
_CENTRAL_HOST = urlparse(
    os.getenv("CENTRAL_SUPABASE_URL", "https://cvermotfxamubejfnoje.supabase.co")
).hostname

# Only allow *.supabase.co URLs (prevents SSRF to internal networks).
_ALLOWED_HOST_RE = re.compile(r"^[a-z0-9-]+\.supabase\.co$")


def _cache_key(url: str, key: str) -> str:
    return hashlib.sha256(f"{url}|{key}".encode()).hexdigest()[:24]


def _evict_stale():
    """Remove entries unused for longer than TTL."""
    now = time.time()
    stale = [k for k, (_, ts) in _tenant_cache.items() if now - ts > _CACHE_TTL]
    for k in stale:
        del _tenant_cache[k]


def get_tenant_db(request: Request) -> DatabaseManager:
    """
    FastAPI dependency: extract Supabase creds from request headers,
    return a cached DatabaseManager for that tenant.

    Falls back to the default (central) DB when no tenant headers are
    present or when the headers point at the central project itself.
    """
    sb_url = request.headers.get("x-supabase-url")
    sb_key = request.headers.get("x-supabase-key")

    # No tenant headers or pointing at central DB — use the default connection
    if not sb_url or not sb_key:
        from backend.dependencies import get_db
        return get_db()

    try:
        parsed = urlparse(sb_url)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Supabase URL")

    host = (parsed.hostname or "").lower()

    # Central DB URL sent as tenant — just fall back to default connection
    if host == _CENTRAL_HOST:
        from backend.dependencies import get_db
        return get_db()

    # SSRF protection: only allow *.supabase.co URLs
    if not _ALLOWED_HOST_RE.match(host):
        raise HTTPException(
            status_code=400,
            detail="Supabase URL must be a *.supabase.co project URL.",
        )

    # ── Cache lookup (keyed on url + key) ───────────────────────────────
    _evict_stale()

    ck = _cache_key(sb_url, sb_key)
    if ck in _tenant_cache:
        db, _ = _tenant_cache[ck]
        _tenant_cache[ck] = (db, time.time())
        return db

    # Enforce an upper bound on cached tenants
    if len(_tenant_cache) >= _MAX_TENANTS:
        oldest = min(_tenant_cache, key=lambda k: _tenant_cache[k][1])
        del _tenant_cache[oldest]

    try:
        db = DatabaseManager(
            enable_caching=False,
            pool_size=5,
            supabase_url=sb_url,
            supabase_key=sb_key,
        )
        _tenant_cache[ck] = (db, time.time())
        logger.info(f"Created tenant DB client for {sb_url[:40]}...")
        return db
    except Exception as e:
        logger.error(f"Failed to create tenant client: {e}")
        raise HTTPException(status_code=502, detail="Could not connect to your Supabase")
