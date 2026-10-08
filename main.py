'''
DJMate API: the Next Track backend.

    uvicorn main:app --reload --host 127.0.0.1 --port 8000

Everything runs locally: rekordbox's database, your audio files and the
Essentia models. The old Supabase-backed routes (chat, tags, crates,
playlists, ingest, 3D map) are switched off for now. Their modules are still
under backend/ and in git history if they're ever wanted again.
'''

from dotenv import load_dotenv
load_dotenv()

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.next_track.router import router as next_track_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="DJMate", version="3.0.0")

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:5178,https://djmate.vercel.app",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)
app.include_router(next_track_router, prefix="/next", tags=["next-track"])


@app.get("/")
async def root():
    """Health check (the desktop app waits for this before opening)."""
    return {"status": "healthy", "service": "DJMate Next Track", "version": "3.0.0"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
