from __future__ import annotations
from pathlib import Path
import math
import json
import xml.etree.ElementTree as ET
import pandas as pd
import argparse

def iso_z(ts) -> str:
    ts = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    lat1 = math.radians(lat1); lon1 = math.radians(lon1)
    lat2 = math.radians(lat2); lon2 = math.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def gpx_root():
    return ET.Element(
        "gpx",
        attrib={
            "version": "1.1",
            "creator": "GLH_GPX_pipeline",
            "xmlns": "http://www.topografix.com/GPX/1/1",
        },
    )


def add_wpt(gpx: ET.Element, lat: float, lon: float, name: str, desc: str, time_iso: str):
    wpt = ET.SubElement(gpx, "wpt", lat=f"{float(lat):.7f}", lon=f"{float(lon):.7f}")
    ET.SubElement(wpt, "name").text = name
    ET.SubElement(wpt, "desc").text = desc
    if time_iso:
        ET.SubElement(wpt, "time").text = time_iso


def write_gpx(path: Path, gpx: ET.Element):
    tree = ET.ElementTree(gpx)
    ET.indent(tree, space="  ", level=0)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anon_id", required=True, help="Anonymized volunteer ID, e.g., V001")
    ap.add_argument("--out_root", default="interim")
    args = ap.parse_args()

    anon_id = args.anon_id
    base_dir = Path(args.out_root) / anon_id
    in_path = base_dir / "gps_at_glh_timestamps_with_tiers.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing: {in_path}. Run run_volunteer_post_qc.py first.")

    df = pd.read_csv(in_path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

    matched = df["gps_interp_ok"].astype(str).str.lower().isin(["true","1","yes"])
    use = df.loc[matched].copy()

    # Require both coords
    use = use.dropna(subset=["timestamp_utc", "lat", "lon", "gps_lat_interp", "gps_lon_interp"])
    use = use.sort_values("timestamp_utc").reset_index(drop=True)

    if len(use) == 0:
        raise ValueError("No matched rows to export.")

    # Pair IDs
    use["pair_id"] = [f"pair_{i:06d}" for i in range(len(use))]

    # GPX files
    gpx_glh = gpx_root()
    gpx_gps = gpx_root()

    # GeoJSON lines
    features = []

    for _, r in use.iterrows():
        pid = r["pair_id"]
        t = iso_z(r["timestamp_utc"])

        glh_lat, glh_lon = float(r["lat"]), float(r["lon"])
        gps_lat, gps_lon = float(r["gps_lat_interp"]), float(r["gps_lon_interp"])

        tier = r["match_quality_tier"] if "match_quality_tier" in use.columns else ""
        seg = r["segment_id"] if "segment_id" in use.columns else ""
        jny = r["journey_id"] if "journey_id" in use.columns else ""
        bg = r["bracket_gap_s"] if "bracket_gap_s" in use.columns else ""

        err = haversine_m(glh_lat, glh_lon, gps_lat, gps_lon)

        desc_glh = f"type=GLH; pair_id={pid}; time={t}; tier={tier}; segment_id={seg}; journey_id={jny}; bracket_gap_s={bg}; error_m={err:.2f}"
        desc_gps = f"type=GPS; pair_id={pid}; time={t}; tier={tier}; segment_id={seg}; journey_id={jny}; bracket_gap_s={bg}; error_m={err:.2f}"

        add_wpt(gpx_glh, glh_lat, glh_lon, name=pid, desc=desc_glh, time_iso=t)
        add_wpt(gpx_gps, gps_lat, gps_lon, name=pid, desc=desc_gps, time_iso=t)

        features.append({
            "type": "Feature",
            "properties": {
                "pair_id": pid,
                "time_utc": t,
                "tier": None if pd.isna(tier) else str(tier),
                "segment_id": None if pd.isna(seg) else int(seg) if str(seg).isdigit() else str(seg),
                "journey_id": None if pd.isna(jny) else int(jny) if str(jny).isdigit() else str(jny),
                "bracket_gap_s": None if pd.isna(bg) else float(bg),
                "error_m": float(err),
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[glh_lon, glh_lat], [gps_lon, gps_lat]],
            }
        })

    out_glh = base_dir / "exports_glh_matched_points.gpx"
    out_gps = base_dir / "exports_gps_matched_points.gpx"
    out_lines = base_dir / "exports_matched_pairs_lines.geojson"
    out_join = base_dir / "exports_matched_pairs_join_table.csv"

    write_gpx(out_glh, gpx_glh)
    write_gpx(out_gps, gpx_gps)

    out_lines.write_text(json.dumps({"type":"FeatureCollection","features":features}, ensure_ascii=False, indent=2), encoding="utf-8")

    # Join table (safe for ArcGIS joins)
    keep = ["pair_id", "timestamp_utc", "lat", "lon", "gps_lat_interp", "gps_lon_interp", "bracket_gap_s", "match_quality_tier", "segment_id", "journey_id"]
    keep = [c for c in keep if c in use.columns]
    use[keep].to_csv(out_join, index=False)

    print("Exported:")
    print(" -", out_glh)
    print(" -", out_gps)
    print(" -", out_lines)
    print(" -", out_join)
    print("Matched pairs:", len(use))


if __name__ == "__main__":
    main()
