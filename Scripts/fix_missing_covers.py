"""
Fix missing album covers: find tracks whose {trackid}.jpg doesn't exist
in the Supabase album-covers bucket, extract art from audio files or
generate placeholders, and upload them.
"""
import os
import sys
import io
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client
from mutagen import File as MutagenFile
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
from PIL import Image, ImageDraw, ImageFont

# ── Config ──────────────────────────────────────────────────────────
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://cvermotfxamubejfnoje.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET = "album-covers"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Helpers ─────────────────────────────────────────────────────────

def list_all_bucket_files():
    """List every file in the album-covers bucket (handles pagination)."""
    files = []
    offset = 0
    limit = 1000
    while True:
        batch = supabase.storage.from_(BUCKET).list(
            path="",
            options={"limit": limit, "offset": offset}
        )
        if not batch:
            break
        files.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
    return files


def fetch_all_tracks():
    """Fetch all tracks from the DB (paginated to avoid row limits)."""
    all_tracks = []
    page_size = 1000
    offset = 0
    while True:
        resp = supabase.table("tracks").select(
            "trackid, artist, title, filepath"
        ).range(offset, offset + page_size - 1).execute()
        batch = resp.data
        if not batch:
            break
        all_tracks.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return all_tracks


def extract_album_art(audio_path: str) -> bytes | None:
    """Try to pull embedded cover art from an audio file."""
    try:
        audio = MutagenFile(audio_path)
        if audio is None:
            return None

        if isinstance(audio, MP3) and audio.tags:
            for tag in audio.tags.values():
                if hasattr(tag, "mime") and "image" in tag.mime:
                    return tag.data

        elif isinstance(audio, FLAC):
            if audio.pictures:
                return audio.pictures[0].data

        elif isinstance(audio, MP4):
            if audio.tags and "covr" in audio.tags:
                return bytes(audio.tags["covr"][0])

        # Generic: check for APIC frames (works for ID3-tagged files)
        if hasattr(audio, "tags") and audio.tags:
            for key in audio.tags:
                if key.startswith("APIC"):
                    return audio.tags[key].data
    except Exception:
        pass
    return None


def make_placeholder(artist: str, title: str) -> bytes:
    """Generate a 400x400 dark placeholder with initials."""
    size = 400
    bg_color = (30, 30, 45)  # dark blue-grey
    text_color = (120, 120, 160)

    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)

    # Build initials (first letter of artist + first letter of title)
    a_init = (artist or "?")[0].upper()
    t_init = (title or "?")[0].upper()
    initials = f"{a_init}{t_init}"

    # Try to use a large font; fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 120)
    except Exception:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 120)
        except Exception:
            font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), initials, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - tw) / 2 - bbox[0]
    y = (size - th) / 2 - bbox[1]
    draw.text((x, y), initials, fill=text_color, font=font)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def upload_cover(trackid: str, image_data: bytes):
    """Upload image_data as {trackid}.jpg to the bucket (upsert)."""
    file_path = f"{trackid}.jpg"
    supabase.storage.from_(BUCKET).upload(
        file_path,
        image_data,
        file_options={"content-type": "image/jpeg", "upsert": "true"},
    )
    url = supabase.storage.from_(BUCKET).get_public_url(file_path)
    if isinstance(url, dict):
        return url.get("data", {}).get("publicUrl") or url.get("publicUrl")
    return url


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("1) Listing bucket files...")
    bucket_files = list_all_bucket_files()
    existing_names = {f["name"] for f in bucket_files if isinstance(f, dict) and "name" in f}
    print(f"   {len(existing_names)} files already in bucket")

    print("2) Fetching all tracks from DB...")
    tracks = fetch_all_tracks()
    print(f"   {len(tracks)} tracks in DB")

    # Find missing
    missing = []
    for t in tracks:
        tid = str(t["trackid"])
        expected = f"{tid}.jpg"
        if expected not in existing_names:
            missing.append(t)

    print(f"3) {len(missing)} tracks missing covers\n")

    if not missing:
        print("Nothing to do!")
        return

    extracted = 0
    placeholders = 0
    errors = 0

    for i, track in enumerate(missing, 1):
        tid = str(track["trackid"])
        artist = track.get("artist") or "Unknown"
        title = track.get("title") or "Unknown"
        filepath = track.get("filepath")

        # Try extracting from audio file
        image_data = None
        if filepath and Path(filepath).exists():
            image_data = extract_album_art(filepath)

        if image_data:
            source = "extracted"
        else:
            image_data = make_placeholder(artist, title)
            source = "placeholder"

        try:
            url = upload_cover(tid, image_data)
            # Also update album_art_url in DB
            supabase.table("tracks").update(
                {"album_art_url": url}
            ).eq("trackid", track["trackid"]).execute()

            if source == "extracted":
                extracted += 1
            else:
                placeholders += 1

            if i % 25 == 0 or i == len(missing):
                print(f"   [{i}/{len(missing)}] {source}: {artist} - {title}")
        except Exception as e:
            errors += 1
            print(f"   [{i}/{len(missing)}] ERROR: {artist} - {title}: {e}")

    print(f"\nDone!")
    print(f"  Extracted from audio: {extracted}")
    print(f"  Placeholder generated: {placeholders}")
    print(f"  Errors: {errors}")
    print(f"  Total processed: {extracted + placeholders + errors}")


if __name__ == "__main__":
    main()
