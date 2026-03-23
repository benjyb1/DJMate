"""Per-tenant Supabase client management for BYOS (Bring Your Own Supabase)."""
import time
import logging
from fastapi import Request, HTTPException
from backend.data.db_interface import DatabaseManager

logger = logging.getLogger(__name__)

# Simple cache: {supabase_url: (DatabaseManager, last_used_timestamp)}
_tenant_cache: dict[str, tuple[DatabaseManager, float]] = {}
_CACHE_TTL = 600  # 10 minutes


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
    Falls back to default env var creds if headers are absent.
    """
    sb_url = request.headers.get("x-supabase-url")
    sb_key = request.headers.get("x-supabase-key")

    if not sb_url or not sb_key:
        # Fallback to default (owner's Supabase)
        from backend.dependencies import get_db
        return get_db()

    _evict_stale()

    if sb_url in _tenant_cache:
        db, _ = _tenant_cache[sb_url]
        _tenant_cache[sb_url] = (db, time.time())
        return db

    try:
        db = DatabaseManager(
            enable_caching=False,
            pool_size=5,
            supabase_url=sb_url,
            supabase_key=sb_key,
        )
        _tenant_cache[sb_url] = (db, time.time())
        logger.info(f"Created tenant DB client for {sb_url[:40]}...")
        return db
    except Exception as e:
        logger.error(f"Failed to create tenant client: {e}")
        raise HTTPException(status_code=502, detail="Could not connect to your Supabase")
