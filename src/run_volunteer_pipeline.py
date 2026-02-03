from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import argparse

from io_glh_unified import load_glh_points_unified
from io_gpx_points import read_gpx_points


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371000.0
    lat1 = np.deg2rad(lat1); lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2); lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def clean_and_segment_gps(gps: pd.DataFrame, max_speed_mps=50.0, gap_break_s=300.0) -> pd.DataFrame:
    g = gps.copy().sort_values("timestamp").reset_index(drop=True)
    g = g.drop_duplicates(subset=["timestamp", "lat", "lon"], keep="first").reset_index(drop=True)

    g["dt_s"] = g["timestamp"].diff().dt.total_seconds()
    g["dist_m"] = haversine_m(g["lat"].shift(1), g["lon"].shift(1), g["lat"], g["lon"])
    g.loc[g["dt_s"].isna(), "dist_m"] = np.nan
    g["speed_mps"] = g["dist_m"] / g["dt_s"]

    bad = (g["dt_s"] > 0) & (g["speed_mps"] > max_speed_mps)
    g = g.loc[~bad].reset_index(drop=True)

    g["dt_s"] = g["timestamp"].diff().dt.total_seconds()
    new_seg = g["dt_s"].isna() | (g["dt_s"] > gap_break_s)
    g["gps_seg_id"] = new_seg.cumsum().astype(int)
    return g


def resample_gps_to_glh_times(gps_df: pd.DataFrame, glh_times: pd.Series, max_interp_gap_s: float = 120.0) -> pd.DataFrame:
    """
    For each GLH timestamp, find GPS points immediately before/after and interpolate.
    Handles exact timestamp matches safely (gap=0).
    """
    gps = gps_df.copy()
    gps["timestamp"] = pd.to_datetime(gps["timestamp"], utc=True, errors="coerce")
    gps = gps.dropna(subset=["timestamp", "lat", "lon"]).sort_values("timestamp").reset_index(drop=True)

    tgt = pd.DataFrame({"timestamp_utc": pd.to_datetime(glh_times, utc=True, errors="coerce")})
    tgt = tgt.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)

    # Merge nearest GPS BEFORE each GLH time
    before = pd.merge_asof(
        tgt, gps.rename(columns={"timestamp": "t_before", "lat": "lat_before", "lon": "lon_before"}),
        left_on="timestamp_utc", right_on="t_before",
        direction="backward", tolerance=None
    )

    # Merge nearest GPS AFTER each GLH time
    after = pd.merge_asof(
        tgt, gps.rename(columns={"timestamp": "t_after", "lat": "lat_after", "lon": "lon_after"}),
        left_on="timestamp_utc", right_on="t_after",
        direction="forward", tolerance=None
    )

    out = tgt.copy()
    out["t_before"] = before["t_before"]
    out["lat_before"] = before["lat_before"]
    out["lon_before"] = before["lon_before"]
    out["t_after"] = after["t_after"]
    out["lat_after"] = after["lat_after"]
    out["lon_after"] = after["lon_after"]

    # bracket gap
    out["bracket_gap_s"] = (out["t_after"] - out["t_before"]).dt.total_seconds()

    # weight for interpolation (safe)
    dt = (out["timestamp_utc"] - out["t_before"]).dt.total_seconds()
    gap = out["bracket_gap_s"]

    # Default: NaN
    out["gps_lat_interp"] = np.nan
    out["gps_lon_interp"] = np.nan

    # Case 1: exact match or degenerate bracket (gap==0)
    exact = gap == 0
    out.loc[exact, "gps_lat_interp"] = out.loc[exact, "lat_before"]
    out.loc[exact, "gps_lon_interp"] = out.loc[exact, "lon_before"]

    # Case 2: normal interpolation (gap>0)
    normal = gap > 0
    w = (dt[normal] / gap[normal]).clip(0, 1)
    out.loc[normal, "gps_lat_interp"] = out.loc[normal, "lat_before"] + w * (out.loc[normal, "lat_after"] - out.loc[normal, "lat_before"])
    out.loc[normal, "gps_lon_interp"] = out.loc[normal, "lon_before"] + w * (out.loc[normal, "lon_after"] - out.loc[normal, "lon_before"])

    # Interp OK rule: both brackets exist, gap within threshold, and coords exist
    out["gps_interp_ok"] = (
        out["t_before"].notna()
        & out["t_after"].notna()
        & out["bracket_gap_s"].notna()
        & (out["bracket_gap_s"] <= max_interp_gap_s)
        & out["gps_lat_interp"].notna()
        & out["gps_lon_interp"].notna()
    )

    # (Optional) if you carry gps_seg_id, add it here from gps_df logic
    out["gps_seg_id"] = np.nan

    return out[["timestamp_utc", "gps_lat_interp", "gps_lon_interp", "gps_interp_ok", "bracket_gap_s", "gps_seg_id"]]


def find_glh_json(vol_dir: Path) -> Path:
    # same function you added in the batch script (keywords + size)
    candidates = list(vol_dir.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No JSON files found in volunteer folder.")

    scored = []
    for p in candidates:
        name = p.name.lower()
        size = p.stat().st_size
        if size < 5_000:
            continue
        score = 0
        if "location-history" in name:
            score += 100
        if "timeline" in name:
            score += 80
        score += min(size / 10_000, 50)
        scored.append((score, size, p))

    if not scored:
        return max(candidates, key=lambda x: x.stat().st_size)

    scored.sort(reverse=True, key=lambda x: (x[0], x[1]))
    return scored[0][2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volunteer_dir", required=True, help="Path to volunteer folder containing GLH JSON + GPX files.")
    ap.add_argument("--anon_id", required=True, help="Anonymized volunteer ID, e.g., V001.")
    ap.add_argument("--out_root", default="interim", help="Output root folder (default: interim).")
    args = ap.parse_args()

    volunteer_dir = Path(args.volunteer_dir)
    anon_id = args.anon_id

    if not volunteer_dir.exists():
        raise FileNotFoundError("Volunteer folder not found.")

    # Auto-detect GLH JSON name safely (no name assumptions)
    glh_path = find_glh_json(volunteer_dir)

    # GPX files (any naming pattern is fine)
    gpx_files = sorted(volunteer_dir.glob("*.gpx"))
    if not gpx_files:
        raise FileNotFoundError("No GPX files found in volunteer folder.")

    # Outputs go to anonymized folder only
    out_dir = Path(args.out_root) / anon_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- GLH unified points ---
    glh_points = load_glh_points_unified(glh_path, volunteer_id=anon_id)

    glh_tp = glh_points.loc[glh_points["source_type"] == "timelinePath"].copy()
    glh_tp = glh_tp.dropna(subset=["timestamp_utc", "lat", "lon"]).sort_values("timestamp_utc").reset_index(drop=True)

    # Save GLH points (already anonymized via volunteer_id)
    glh_points.to_csv(out_dir / "glh_points.csv", index=False)

    # --- GPS points ---
    gps_parts = []
    for p in gpx_files:
        dfp = read_gpx_points(p)
        if dfp is None or dfp.empty:
            continue
        gps_parts.append(dfp)

    if not gps_parts:
        raise FileNotFoundError("All GPX files had 0 usable points.")

    gps = pd.concat(gps_parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    # IMPORTANT: remove src_file to avoid leaking GPX filenames like Session_xxx
    if "src_file" in gps.columns:
        gps = gps.drop(columns=["src_file"])

    gps.to_csv(out_dir / "gps_points_raw.csv", index=False)

    # --- Clean GPS ---
    gps_clean = clean_and_segment_gps(gps, max_speed_mps=50.0, gap_break_s=300.0)
    gps_clean.to_csv(out_dir / "gps_points_clean.csv", index=False)

    # --- Clip GLH to GPS coverage (+ buffer) ---
    gps_min = gps_clean["timestamp"].min()
    gps_max = gps_clean["timestamp"].max()

    buffer_min = 10
    gps_min_b = gps_min - pd.Timedelta(minutes=buffer_min)
    gps_max_b = gps_max + pd.Timedelta(minutes=buffer_min)

    glh_tp["in_gps_window"] = glh_tp["timestamp_utc"].between(gps_min, gps_max, inclusive="both")
    glh_tp["in_gps_window_buffer"] = glh_tp["timestamp_utc"].between(gps_min_b, gps_max_b, inclusive="both")
    glh_match = glh_tp.loc[glh_tp["in_gps_window_buffer"]].copy().reset_index(drop=True)

    # --- Interpolate GPS to GLH timestamps ---
    interp = resample_gps_to_glh_times(gps_clean, glh_match["timestamp_utc"], max_interp_gap_s=120.0)
    glh_match = glh_match.sort_values("timestamp_utc").reset_index(drop=True)
    interp = interp.sort_values("timestamp_utc").reset_index(drop=True)

    out = pd.concat([glh_match, interp[["gps_lat_interp", "gps_lon_interp", "gps_interp_ok", "bracket_gap_s", "gps_seg_id"]]], axis=1)
    missing_interp = (
        matched["gps_lat_interp"].isna()
        | matched["gps_lon_interp"].isna()
    )
    matched.loc[missing_interp, "gps_interp_ok"] = False
    missing_interp = matched["gps_lat_interp"].isna() | matched["gps_lon_interp"].isna()
    matched.loc[missing_interp, "gps_interp_ok"] = False

    out.to_csv(out_dir / "gps_at_glh_timestamps.csv", index=False)

    # Minimal log (no volunteer_dir printed)
    print(f"Done: {anon_id}")
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()