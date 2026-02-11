import os
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import csv

# Load environment variables
load_dotenv()

# Connect to Supabase Postgres
conn = psycopg2.connect(
    dbname=os.getenv("dbname"),
    user=os.getenv("user"),
    password=os.getenv("password"),
    host=os.getenv("host"),
    port=os.getenv("port")
)
cur = conn.cursor()

# Read dummy metadata from Test_supabase.csv
dummy_data = []
with open("Test_supabase.csv", newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        trackid = int(row['trackid'])
        filepath = row['filepath']
        title = row['title']
        artist = row['artist']
        album = row['album']
        bpm = float(row['bpm']) if row['bpm'] else None
        key = row['key']
        embedding = None  # embedding left as NULL
        dummy_data.append((trackid, filepath, title, artist, album, bpm, key, embedding))

# Insert dummy rows into csv_tracks
execute_values(
    cur,
    """
    INSERT INTO csv_tracks (trackid, filepath, title, artist, album, bpm, key, embedding)
    VALUES %s
    ON CONFLICT (trackid) DO NOTHING
    """,
    dummy_data
)

conn.commit()
cur.close()
conn.close()

print("✅ Inserted dummy metadata rows into csv_tracks.")