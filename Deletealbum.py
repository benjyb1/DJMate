import os
from pathlib import Path
from supabase import create_client, Client

# ==================== CONFIGURATION ====================
SUPABASE_URL = 'https://cvermotfxamubejfnoje.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImN2ZXJtb3RmeGFtdWJlamZub2plIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTk2NTU4MTcsImV4cCI6MjA3NTIzMTgxN30.clXSFQ4QVhL8nUK_6shyhDVxhKaHUtnrdyqCnDeCCag'

FOLDER_TO_DELETE = Path("/Users/benjyb/Desktop/Mixing/Older Selections/Wider Selection/Harry")
BUCKET_NAME = "album-covers"
AUDIO_EXTENSIONS = {'.mp3', '.flac', '.m4a', '.mp4', '.ogg', '.wma'}

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def delete_songs_from_folder():
    """Delete songs based on filename (title) matching only."""

    if not FOLDER_TO_DELETE.exists():
        print(f"❌ Folder not found: {FOLDER_TO_DELETE}")
        return

    print("=" * 60)
    print("STEP 1: Scanning Folder for Audio Files")
    print("=" * 60)
    print(f"📁 Folder: {FOLDER_TO_DELETE}\n")

    # Find all audio files in the folder
    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(FOLDER_TO_DELETE.rglob(f"*{ext}"))

    print(f"📀 Found {len(audio_files)} audio files\n")

    if len(audio_files) == 0:
        print("✅ No audio files to delete!")
        return

    # Extract titles from filenames
    print("=" * 60)
    print("STEP 2: Matching Filenames to Database Records")
    print("=" * 60)

    tracks_to_delete = []

    for audio_path in audio_files:
        # Get title from filename (remove extension)
        title = audio_path.stem

        print(f"🔍 Looking for: '{title}'")

        # Find in database by title
        try:
            response = supabase.table('tracks').select('trackid, title, artist').eq('title', title).execute()

            if response.data and len(response.data) > 0:
                for track in response.data:
                    tracks_to_delete.append(track)
                    artist = track.get('artist') or 'Unknown'
                    print(f"   ✅ Found: {track['trackid']} - {artist} - {track['title']}")
            else:
                print(f"   ⚠️  Not found in database")
        except Exception as e:
            print(f"   ❌ Error searching: {e}")

    print(f"\n📊 Summary:")
    print(f"   Tracks found in database: {len(tracks_to_delete)}")

    if len(tracks_to_delete) == 0:
        print("\n✅ No tracks found in database to delete!")
        return

    # Show what will be deleted
    print("\n" + "=" * 60)
    print("📋 Tracks that will be deleted:")
    print("=" * 60)
    for track in tracks_to_delete:
        artist = track.get('artist') or 'Unknown'
        print(f"  • {track['trackid']}: {artist} - {track['title']}")

    # Confirmation
    print("\n" + "=" * 60)
    print("⚠️  WARNING: This will permanently delete:")
    print(f"   • {len(tracks_to_delete)} tracks from 'tracks' table")
    print(f"   • Related records from 'track_labels' table")
    print(f"   • Related records from 'tag_track_feedback' table")
    print(f"   • Associated album covers from storage (if they exist)")
    print("=" * 60)

    response = input("\nType 'DELETE' to confirm: ")

    if response != 'DELETE':
        print("❌ Cancelled")
        return

    # Extract trackids
    trackids_to_delete = [track['trackid'] for track in tracks_to_delete]

    # Delete from track_labels
    print("\n" + "=" * 60)
    print("STEP 3: Deleting from track_labels")
    print("=" * 60)

    deleted_labels = 0
    for trackid in trackids_to_delete:
        try:
            result = supabase.table('track_labels').delete().eq('trackid', trackid).execute()
            if result.data:
                deleted_labels += len(result.data)
        except Exception as e:
            print(f"⚠️  Error deleting labels for {trackid}: {e}")

    print(f"✅ Deleted {deleted_labels} records from track_labels")

    # Delete from tag_track_feedback
    print("\n" + "=" * 60)
    print("STEP 4: Deleting from tag_track_feedback")
    print("=" * 60)

    deleted_feedback = 0
    for trackid in trackids_to_delete:
        try:
            result = supabase.table('tag_track_feedback').delete().eq('trackid', trackid).execute()
            if result.data:
                deleted_feedback += len(result.data)
        except Exception as e:
            print(f"⚠️  Error deleting feedback for {trackid}: {e}")

    print(f"✅ Deleted {deleted_feedback} records from tag_track_feedback")

    # Delete album covers from storage
    print("\n" + "=" * 60)
    print("STEP 5: Deleting Album Covers from Storage")
    print("=" * 60)

    deleted_covers = 0

    for track in tracks_to_delete:
        artist = track.get('artist', 'unknown')
        title = track.get('title', 'unknown')

        if artist and title:
            # Try to construct the cover filename
            import unicodedata
            combined = f"{artist}_{title}".lower()
            normalized = unicodedata.normalize('NFD', combined)
            ascii_text = normalized.encode('ascii', 'ignore').decode('ascii')
            safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in ascii_text)
            safe_name = '_'.join(filter(None, safe_name.split('_')))
            cover_file = f"{safe_name[:150]}.jpg"

            try:
                supabase.storage.from_(BUCKET_NAME).remove([cover_file])
                deleted_covers += 1
            except Exception:
                # File doesn't exist or error - that's okay
                pass

    print(f"✅ Attempted to delete {deleted_covers} album covers from storage")

    # Delete from tracks table
    print("\n" + "=" * 60)
    print("STEP 6: Deleting from tracks Table")
    print("=" * 60)

    deleted_tracks = 0
    for trackid in trackids_to_delete:
        try:
            supabase.table('tracks').delete().eq('trackid', trackid).execute()
            deleted_tracks += 1
            if deleted_tracks % 10 == 0:
                print(f"   Deleted {deleted_tracks}/{len(trackids_to_delete)} tracks...")
        except Exception as e:
            print(f"⚠️  Error deleting track {trackid}: {e}")

    print(f"✅ Deleted {deleted_tracks} tracks")

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 DELETION COMPLETE!")
    print("=" * 60)
    print(f"""
Summary:
  ✅ Deleted {deleted_tracks} tracks
  ✅ Deleted {deleted_labels} track_labels records
  ✅ Deleted {deleted_feedback} tag_track_feedback records
  ✅ Attempted to delete {deleted_covers} album covers
    """)

if __name__ == "__main__":
    print("🗑️  Bulk Track Deletion Script (Filename-Based)\n")
    delete_songs_from_folder()