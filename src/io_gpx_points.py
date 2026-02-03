from __future__ import annotations
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd


def read_gpx_points(gpx_path: Path) -> pd.DataFrame:
    """
    Extract track points from GPX.
    Returns DataFrame with columns: timestamp (UTC), lat, lon, src_file
    """
    tree = ET.parse(gpx_path)
    root = tree.getroot()

    # GPX namespace handling
    ns = {}
    if root.tag.startswith("{"):
        uri = root.tag.split("}")[0].strip("{")
        ns = {"g": uri}
        trkpt_xpath = ".//g:trkpt"
        time_xpath = "g:time"
    else:
        trkpt_xpath = ".//trkpt"
        time_xpath = "time"

    rows = []
    for trkpt in root.findall(trkpt_xpath, ns):
        lat = trkpt.attrib.get("lat")
        lon = trkpt.attrib.get("lon")
        t_el = trkpt.find(time_xpath, ns)
        t = t_el.text if t_el is not None else None

        rows.append({
            "timestamp": pd.to_datetime(t, utc=True, errors="coerce"),
            "lat": float(lat) if lat is not None else None,
            "lon": float(lon) if lon is not None else None,
            "src_file": gpx_path.name,
        })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["timestamp", "lat", "lon"]).sort_values("timestamp").reset_index(drop=True)
    return df
