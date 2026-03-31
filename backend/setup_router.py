"""Setup router — serves the schema SQL for new-user onboarding."""
import os
from fastapi import APIRouter

router = APIRouter()

# Read the canonical schema.sql at import time so there's only one source of truth.
_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "schema.sql")
try:
    with open(_SCHEMA_PATH, "r") as f:
        _SCHEMA_SQL = f.read()
except FileNotFoundError:
    _SCHEMA_SQL = "-- schema.sql not found. Check the DJMate repo for setup instructions."


@router.get("/schema-sql")
async def get_schema_sql():
    """Return the SQL needed to set up a fresh Supabase project for DJMate."""
    return {"sql": _SCHEMA_SQL}
