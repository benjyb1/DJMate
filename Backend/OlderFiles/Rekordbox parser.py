import xml.etree.ElementTree as ET

tree = ET.parse("/Users/benjyb/Desktop/RekordboxXMLV1.xml")
root = tree.getroot()

tracks = []

for track in root.findall(".//TRACK"):
    title = track.get("Name")
    artist = track.get("Artist")
    key = track.get("Tonality")
    bpm = track.get("AverageBpm")

    # Only append if at least one value is not None
    if any([title, artist, key, bpm]):
        tracks.append({
            "title": title,
            "artist": artist,
            "key": key,
            "bpm": bpm
        })




