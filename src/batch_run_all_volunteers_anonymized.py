from __future__ import annotations

from pathlib import Path
import shutil
import pandas as pd

# Import your existing functions/modules
from io_glh_unified import load_glh_points_unified
from io_gpx_points import read_gpx_points
from run_volunteer_pipeline import clean_and_segment_gps, resample_gps_to_glh_times
from run_volunteer_post_qc import build_segments_from_points, assign_journey_id, make_quality_tier
from export_pairs_to_two_gpx_and_lines import main as export_pairs_main  # will run per volunteer if we set env/arg; see below


# ---------- privacy helpers ----------

DROP_COLS_POINTS = [
    "place_id",          # Google place identifier
    # if you ever add more later, put them here
]

DROP_COLS_GPS = [
    "src_file",          # GPX filename can contain identifying pattern
]

def sanitize_df(df: pd.DataFrame, drop_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in drop_cols:
        if c in out.columns:
            out = out.drop(columns=[c])
    return out

def find_glh_json(vol_dir: Path) -> Path:
    """
    Find the most likely GLH JSON export inside a volunteer folder.
    Supports names like:
      - location-history.json
      - location-history_hrz.json
      - Timeline.json
      - Google_Maps_Timeline_*.json
    Strategy:
      1) Prefer filenames containing timeline / location-history
      2) Prefer larger files (more data) if multiple candidates
      3) Ignore tiny json files that are likely config/metadata
    """
    candidates = list(vol_dir.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No JSON files found in: {vol_dir}")

    scored = []
    for p in candidates:
        name = p.name.lower()
        size = p.stat().st_size

        # ignore extremely small jsons (often not GLH)
        if size < 5_000:  # 5 KB
            continue

        score = 0
        if "location-history" in name:
            score += 100
        if "timeline" in name:
            score += 80
        if "google_maps_timeline" in name or "google maps timeline" in name:
            score += 80

        # bigger usually means more complete export
        score += min(size / 10_000, 50)  # up to +50

        scored.append((score, size, p))

    if not scored:
        # fall back to any json if all were tiny
        p = max(candidates, key=lambda x: x.stat().st_size)
        return p

    scored.sort(reverse=True, key=lambda x: (x[0], x[1]))
    return scored[0][2]

def run_one_volunteer(vol_dir: Path, anon_id: str):
    """
    vol_dir structure (your current layout):
      raw_data/VolunteerX/
        location-history.json
        *.gpx   (any naming is fine)
    """

    glh_path = find_glh_json(vol_dir)


    gpx_files = sorted(vol_dir.glob("*.gpx"))
    if not gpx_files:
        raise FileNotFoundError(f"No GPX files in: {vol_dir}")

    out_dir = Path("interim") / anon_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1) GLH unified points ----------
    glh_points = load_glh_points_unified(glh_path, volunteer_id=anon_id)

    # Use timelinePath for matching (consistent with your pipeline)
    glh_tp = glh_points.loc[glh_points["source_type"] == "timelinePath"].copy()
    glh_tp = glh_tp.dropna(subset=["timestamp_utc", "lat", "lon"]).sort_values("timestamp_utc").reset_index(drop=True)

    # Sanitize and save
    glh_points_s = sanitize_df(glh_points, DROP_COLS_POINTS)
    glh_points_s.to_csv(out_dir / "glh_points.csv", index=False)

    # ---------- 2) GPS points from GPX ----------
    gps_parts = [read_gpx_points(p) for p in gpx_files]
    gps = pd.concat(gps_parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    # sanitize GPS immediately (drop src_file)
    gps_s = sanitize_df(gps, DROP_COLS_GPS)
    gps_s.to_csv(out_dir / "gps_points_raw.csv", index=False)

    # ---------- 3) Clean GPS ----------
    gps_clean = clean_and_segment_gps(gps_s, max_speed_mps=50.0, gap_break_s=300.0)
    gps_clean.to_csv(out_dir / "gps_points_clean.csv", index=False)

    # ---------- 4) Clip GLH to GPS coverage (buffer) ----------
    gps_min = gps_clean["timestamp"].min()
    gps_max = gps_clean["timestamp"].max()

    buffer_min = 10
    gps_min_b = gps_min - pd.Timedelta(minutes=buffer_min)
    gps_max_b = gps_max + pd.Timedelta(minutes=buffer_min)

    glh_tp["in_gps_window"] = glh_tp["timestamp_utc"].between(gps_min, gps_max, inclusive="both")
    glh_tp["in_gps_window_buffer"] = glh_tp["timestamp_utc"].between(gps_min_b, gps_max_b, inclusive="both")

    glh_match = glh_tp.loc[glh_tp["in_gps_window_buffer"]].copy().reset_index(drop=True)

    # ---------- 5) Interpolate GPS to GLH timestamps ----------
    interp = resample_gps_to_glh_times(gps_clean, glh_match["timestamp_utc"], max_interp_gap_s=120.0)
    glh_match = glh_match.sort_values("timestamp_utc").reset_index(drop=True)
    interp = interp.sort_values("timestamp_utc").reset_index(drop=True)

    matched = pd.concat(
        [glh_match, interp[["gps_lat_interp", "gps_lon_interp", "gps_interp_ok", "bracket_gap_s", "gps_seg_id"]]],
        axis=1
    )

    # ---------- 6) Add journeys + tiers (Step 1–3 from earlier) ----------
    # Build segment table from points (timelinePath only)
    segs = build_segments_from_points(glh_points_s)
    segs = assign_journey_id(segs, gap_threshold_minutes=20.0)
    matched = matched.merge(segs[["segment_id", "journey_id"]], on="segment_id", how="left")

    # Tier for matched points
    is_ok = matched["gps_interp_ok"].astype(str).str.lower().isin(["true", "1", "yes"])
    matched["match_quality_tier"] = pd.NA
    matched.loc[is_ok, "match_quality_tier"] = make_quality_tier(matched.loc[is_ok, "bracket_gap_s"])

    out_match = out_dir / "gps_at_glh_timestamps_with_tiers.csv"
    matched.to_csv(out_match, index=False)

    segs.to_csv(out_dir / "glh_timeline_segments_with_journeys.csv", index=False)

    # ---------- 7) QC report ----------
    qc_path = out_dir / "qc_match_detailed.txt"
    total_pts = len(matched)
    ok_pts = int(is_ok.sum())
    total_seg = matched["segment_id"].nunique(dropna=True)
    ok_seg = matched.loc[is_ok, "segment_id"].nunique(dropna=True)
    total_j = matched["journey_id"].nunique(dropna=True)
    ok_j = matched.loc[is_ok, "journey_id"].nunique(dropna=True)

    lines = []
    lines.append(f"Volunteer (anonymized): {anon_id}")
    lines.append("")
    lines.append("POINTS")
    lines.append(f"  Total GLH timelinePath points (buffered window): {total_pts}")
    lines.append(f"  Matched points: {ok_pts}")
    lines.append(f"  Match rate: {ok_pts/total_pts*100 if total_pts else 0:.2f}%")
    lines.append("")
    lines.append("SEGMENTS")
    lines.append(f"  Total segments: {int(total_seg)}")
    lines.append(f"  Segments with ≥1 matched point: {int(ok_seg)} ({(ok_seg/total_seg*100 if total_seg else 0):.2f}%)")
    lines.append("")
    lines.append("JOURNEYS")
    lines.append(f"  Total journeys: {int(total_j)}")
    lines.append(f"  Journeys with ≥1 matched point: {int(ok_j)} ({(ok_j/total_j*100 if total_j else 0):.2f}%)")
    lines.append("")
    if ok_pts:
        bg = pd.to_numeric(matched.loc[is_ok, "bracket_gap_s"], errors="coerce").dropna()
        lines.append("INTERPOLATION QUALITY (matched)")
        lines.append(f"  bracket_gap_s median: {float(bg.median()):.1f}")
        lines.append(f"  bracket_gap_s 90th pct: {float(bg.quantile(0.90)):.1f}")
        lines.append(f"  bracket_gap_s max: {float(bg.max()):.1f}")

    qc_path.write_text("\n".join(lines), encoding="utf-8")

    # ---------- 8) Export ArcGIS layers (two GPX + GeoJSON) ----------
    # Re-use your existing exporter by calling it as a module is awkward because it's hard-coded.
    # So here we simply copy the exporter outputs approach inline by importing your exporter would require refactor.
    # Easiest: call your exporter script separately after this batch run (see commands below).

    return out_dir


def main():
    raw_root = Path("raw_data")
    vol_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir() and p.name.lower().startswith("volunteer")])

    if not vol_dirs:
        raise FileNotFoundError("No volunteer folders found under raw_data/ (expected Volunteer1, Volunteer2, ...)")

    # Create mapping (stored locally; do not share)
    mapping_rows = []
    for i, d in enumerate(vol_dirs, start=1):
        anon_id = f"V{i:03d}"
        mapping_rows.append({"anon_id": anon_id, "source_folder": d.name})

    map_df = pd.DataFrame(mapping_rows)
    map_df.to_csv(Path("interim") / "anon_mapping_private.csv", index=False)

    print("Found volunteers:", [d.name for d in vol_dirs])
    print("Anonymized IDs:", map_df["anon_id"].tolist())
    print("Private mapping saved to interim/anon_mapping_private.csv")

    for i, d in enumerate(vol_dirs, start=1):
        anon_id = f"V{i:03d}"
        print("\n--- Running:", d.name, "->", anon_id, "---")
        out_dir = run_one_volunteer(d, anon_id)
        print("Saved to:", out_dir)

    print("\nDone. Next: export ArcGIS GPX/GeoJSON for each V### (see instructions).")


if __name__ == "__main__":
    main()
