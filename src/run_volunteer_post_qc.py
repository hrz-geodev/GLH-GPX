from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import argparse


def make_quality_tier(bracket_gap_s: pd.Series) -> pd.Series:
    """
    Tiering for matched points based on interpolation bracket gap.
    """
    bg = pd.to_numeric(bracket_gap_s, errors="coerce")
    tier = pd.Series(pd.NA, index=bg.index, dtype="object")

    tier[(bg <= 10)] = "A_<=10s"
    tier[(bg > 10) & (bg <= 30)] = "B_10-30s"
    tier[(bg > 30) & (bg <= 60)] = "C_30-60s"
    tier[(bg > 60) & (bg <= 120)] = "D_60-120s"
    tier[(bg > 120)] = "E_>120s"   # should not happen if you capped at 120, but keep for safety
    return tier


def build_segments_from_points(points: pd.DataFrame) -> pd.DataFrame:
    """
    Build a segment table from GLH points (timelinePath rows),
    using segment_id + segment_start_utc/segment_end_utc when available,
    otherwise fallback to min/max timestamp_utc inside segment.
    """
    p = points.copy()

    # Ensure types
    p["timestamp_utc"] = pd.to_datetime(p["timestamp_utc"], utc=True, errors="coerce")
    if "segment_start_utc" in p.columns:
        p["segment_start_utc"] = pd.to_datetime(p["segment_start_utc"], utc=True, errors="coerce")
    else:
        p["segment_start_utc"] = pd.NaT
    if "segment_end_utc" in p.columns:
        p["segment_end_utc"] = pd.to_datetime(p["segment_end_utc"], utc=True, errors="coerce")
    else:
        p["segment_end_utc"] = pd.NaT

    # Only timelinePath segments are relevant here
    p = p[p["source_type"] == "timelinePath"].copy()
    p = p.dropna(subset=["segment_id"])

    # Aggregate per segment
    agg = p.groupby("segment_id").agg(
        seg_start_raw=("segment_start_utc", "min"),
        seg_end_raw=("segment_end_utc", "max"),
        seg_start_pts=("timestamp_utc", "min"),
        seg_end_pts=("timestamp_utc", "max"),
        n_points=("timestamp_utc", "size"),
    ).reset_index()

    # Prefer raw segment start/end; fallback to point range if missing
    agg["segment_start_utc_final"] = agg["seg_start_raw"].where(agg["seg_start_raw"].notna(), agg["seg_start_pts"])
    agg["segment_end_utc_final"] = agg["seg_end_raw"].where(agg["seg_end_raw"].notna(), agg["seg_end_pts"])

    return agg[[
        "segment_id",
        "segment_start_utc_final",
        "segment_end_utc_final",
        "n_points"
    ]].rename(columns={
        "segment_start_utc_final": "segment_start_utc",
        "segment_end_utc_final": "segment_end_utc",
    })


def assign_journey_id(segments: pd.DataFrame, gap_threshold_minutes: float = 20.0) -> pd.DataFrame:
    """
    Sort segments by start time, create journey_id based on gaps between previous end and current start.
    """
    seg = segments.copy()
    seg["segment_start_utc"] = pd.to_datetime(seg["segment_start_utc"], utc=True, errors="coerce")
    seg["segment_end_utc"] = pd.to_datetime(seg["segment_end_utc"], utc=True, errors="coerce")
    seg = seg.dropna(subset=["segment_start_utc", "segment_end_utc"]).sort_values("segment_start_utc").reset_index(drop=True)

    prev_end = seg["segment_end_utc"].shift(1)
    gap_min = (seg["segment_start_utc"] - prev_end).dt.total_seconds() / 60.0
    seg["gap_from_prev_min"] = gap_min

    new_journey = gap_min.isna() | (gap_min > gap_threshold_minutes)
    seg["journey_id"] = new_journey.cumsum().astype(int) - 1  # start at 0

    return seg


def write_qc_report(df: pd.DataFrame, out_txt: Path):
    """
    df is gps_at_glh_timestamps_with_tiers.csv (contains gps_interp_ok, match_quality_tier, segment_id, journey_id)
    """
    matched = (
    df["gps_interp_ok"].astype(str).str.lower().isin(["true", "1", "yes"])
    & df["gps_lat_interp"].notna()
    & df["gps_lon_interp"].notna()
    )

    lines = []
    lines.append("POINTS (timelinePath rows in buffer window)")
    lines.append(f"  Total points: {len(df)}")
    lines.append(f"  Matched points: {int(matched.sum())} ({(matched.mean()*100 if len(df) else 0):.2f}%)")
    lines.append("")

    # By tier
    if "match_quality_tier" in df.columns:
        lines.append("MATCHED POINTS by tier")
        tier_counts = df.loc[matched, "match_quality_tier"].value_counts(dropna=False)
        for k, v in tier_counts.items():
            lines.append(f"  {k}: {int(v)}")
        lines.append("")

    # Segments
    if "segment_id" in df.columns:
        total_seg = df["segment_id"].nunique(dropna=True)
        matched_seg = df.loc[matched, "segment_id"].nunique(dropna=True)
        lines.append("SEGMENTS (timelinePath segments represented in this file)")
        lines.append(f"  Total segments: {int(total_seg)}")
        lines.append(f"  Segments with ≥1 matched point: {int(matched_seg)} ({(matched_seg/total_seg*100 if total_seg else 0):.2f}%)")
        lines.append("")

        if "match_quality_tier" in df.columns:
            lines.append("MATCHED SEGMENTS by best tier achieved")
            # For each segment, get best tier (A best, then B, C, D, E)
            tier_order = {"A_<=10s": 0, "B_10-30s": 1, "C_30-60s": 2, "D_60-120s": 3, "E_>120s": 4}
            tmp = df.loc[matched, ["segment_id", "match_quality_tier"]].dropna()
            if len(tmp):
                tmp["tier_rank"] = tmp["match_quality_tier"].map(tier_order).fillna(99).astype(int)
                best = tmp.sort_values(["segment_id", "tier_rank"]).groupby("segment_id").first()
                best_counts = best["match_quality_tier"].value_counts()
                for k, v in best_counts.items():
                    lines.append(f"  {k}: {int(v)}")
            else:
                lines.append("  (no matched segments)")
            lines.append("")

    # Journeys
    if "journey_id" in df.columns:
        total_j = df["journey_id"].nunique(dropna=True)
        matched_j = df.loc[matched, "journey_id"].nunique(dropna=True)
        lines.append("JOURNEYS (derived from segment gaps)")
        lines.append(f"  Total journeys: {int(total_j)}")
        lines.append(f"  Journeys with ≥1 matched point: {int(matched_j)} ({(matched_j/total_j*100 if total_j else 0):.2f}%)")
        lines.append("")

        if "match_quality_tier" in df.columns:
            lines.append("MATCHED JOURNEYS by best tier achieved")
            tier_order = {"A_<=10s": 0, "B_10-30s": 1, "C_30-60s": 2, "D_60-120s": 3, "E_>120s": 4}
            tmp = df.loc[matched, ["journey_id", "match_quality_tier"]].dropna()
            if len(tmp):
                tmp["tier_rank"] = tmp["match_quality_tier"].map(tier_order).fillna(99).astype(int)
                best = tmp.sort_values(["journey_id", "tier_rank"]).groupby("journey_id").first()
                best_counts = best["match_quality_tier"].value_counts()
                for k, v in best_counts.items():
                    lines.append(f"  {k}: {int(v)}")
            else:
                lines.append("  (no matched journeys)")
            lines.append("")

    # Interp quality summary
    if "bracket_gap_s" in df.columns:
        bg = pd.to_numeric(df.loc[matched, "bracket_gap_s"], errors="coerce").dropna()
        if len(bg):
            lines.append("INTERPOLATION QUALITY (matched points)")
            lines.append(f"  bracket_gap_s median: {float(bg.median()):.1f}")
            lines.append(f"  bracket_gap_s 75th pct: {float(bg.quantile(0.75)):.1f}")
            lines.append(f"  bracket_gap_s 90th pct: {float(bg.quantile(0.90)):.1f}")
            lines.append(f"  bracket_gap_s 95th pct: {float(bg.quantile(0.95)):.1f}")
            lines.append(f"  bracket_gap_s max: {float(bg.max()):.1f}")
            lines.append("")

    out_txt.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anon_id", required=True, help="Anonymized volunteer ID, e.g., V001")
    ap.add_argument("--out_root", default="interim")
    args = ap.parse_args()

    anon_id = args.anon_id
    base_dir = Path(args.out_root) / anon_id

    in_match = base_dir / "gps_at_glh_timestamps.csv"
    in_points = base_dir / "glh_points.csv"

    if not in_match.exists():
        raise FileNotFoundError(f"Missing: {in_match}")
    if not in_points.exists():
        raise FileNotFoundError(f"Missing: {in_points}")

    df = pd.read_csv(in_match)
    pts = pd.read_csv(in_points)

    # Ensure time types
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

    # Add journey_id: build segments from pts -> assign journey -> merge onto df by segment_id
    segs = build_segments_from_points(pts)
    segs = assign_journey_id(segs, gap_threshold_minutes=20.0)

    df = df.merge(segs[["segment_id", "journey_id"]], on="segment_id", how="left")

    # Add match quality tier for matched rows
    df["match_quality_tier"] = pd.NA
    matched = (
    df["gps_interp_ok"].astype(str).str.lower().isin(["true", "1", "yes"])
    & df["gps_lat_interp"].notna()
    & df["gps_lon_interp"].notna()
    )
    df.loc[matched, "match_quality_tier"] = make_quality_tier(df.loc[matched, "bracket_gap_s"])

    # Save enriched match table and segments table
    out_match = base_dir / "gps_at_glh_timestamps_with_tiers.csv"
    out_segs = base_dir / "glh_timeline_segments_with_journeys.csv"
    df.to_csv(out_match, index=False)
    segs.to_csv(out_segs, index=False)

    # QC report
    out_qc = base_dir / "qc_match_detailed.txt"
    write_qc_report(df, out_qc)

    print("Saved:")
    print(" -", out_match)
    print(" -", out_segs)
    print(" -", out_qc)


if __name__ == "__main__":
    main()
