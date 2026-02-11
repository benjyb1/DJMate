import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Connect to Supabase
conn = psycopg2.connect(
    dbname=os.getenv("dbname"),
    user=os.getenv("user"),
    password=os.getenv("password"),
    host=os.getenv("host"),
    port=os.getenv("port")
)
cur = conn.cursor()

# Load your embeddings
df = pd.read_csv("tracks_with_embeddings.csv", header=None, encoding="utf-8-sig")

# Convert rows to lists of floats
df['embedding'] = df.values.tolist()
df['embedding'] = df['embedding'].apply(lambda x: [float(i) for i in x])

records = [(i + 1, r) for i, r in enumerate(df['embedding'].tolist())]

# Update embeddings in csv_tracks
execute_values(
    cur,
    """
    UPDATE csv_tracks AS t
    SET embedding = v.embedding
    FROM (VALUES %s) AS v(trackid, embedding)
    WHERE t.trackid = v.trackid
    """,
    records
)

conn.commit()
cur.close()
conn.close()

print("✅ Updated embeddings in csv_tracks.")