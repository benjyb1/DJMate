"""Shared FastAPI dependencies."""
import logging
from functools import lru_cache
from backend.data.db_interface import DatabaseManager

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_db() -> DatabaseManager:
    logger.info("Initializing shared DatabaseManager")
    return DatabaseManager(enable_caching=True, pool_size=10)
