import numpy as np
import umap
import json
from supabase import create_client

# Your Supabase setup
SUPABASE_URL = "https://cvermotfxamubejfnoje.supabase.co"
SUPABASE_KEY = "sb_secret_1U7o2RsVAD2_5eTdBQaxkw_adLbxVBe"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def generate_3d_map():
    # 1. Fetch all tracks and their embeddings
    response = supabase.table("tracks").select("trackid, embedding").execute()
    tracks = response.data

    # Extract embeddings into a list
    embeddings = [np.array(json.loads(t['embedding']), dtype=np.float32) for t in tracks]

    # 2. Use UMAP to reduce dimensions to 3 (x, y, z)
    reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine')
    coords_3d = reducer.fit_transform(embeddings)

    # 3. Update Supabase with the new coordinates
    for i, track in enumerate(tracks):
        x, y, z = coords_3d[i]
        supabase.table("tracks").update({
            "x_coord": float(x),
            "y_coord": float(y),
            "z_coord": float(z)
        }).eq("trackid", track['trackid']).execute()

    print("Map coordinates generated and synced to Supabase!")

generate_3d_map()