from __future__ import annotations

from pathlib import Path
import math
import pandas as pd
import numpy as np


def haversine_m(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine distance in meters.
    """
    R = 6371000.0

    lat1 = np.radians(pd.to_numeric(lat1, errors="coerce"))
    lon1 = np.radians(pd.to_numeric(lon1, errors="coerce"))
    lat2 = np.radians(pd.to_numeric(lat2, errors="coerce"))
    lon2 = np.radians(pd.to_numeric(lon2, errors="coerce"))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def matched_mask(df: pd.DataFrame) -> pd.Series:
    """
    A point is usable for accuracy verification only if:
    - gps_interp_ok is true
    - interpolated GPS coordinates exist
    - GLH coordinates exist
    """
    interp_ok = df["gps_interp_ok"].astype(str).str.lower().isin(["true", "1", "yes"])
    return (
        interp_ok
        & df["gps_lat_interp"].notna()
        & df["gps_lon_interp"].notna()
        & df["lat"].notna()
        & df["lon"].notna()
    )


def describe_error(s: pd.Series) -> dict:
    """
    Basic accuracy stats.
    """
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return {
            "n": 0,
            "mean_error_m": np.nan,
            "median_error_m": np.nan,
            "p75_error_m": np.nan,
            "p90_error_m": np.nan,
            "p95_error_m": np.nan,
            "max_error_m": np.nan,
        }

    return {
        "n": int(len(s)),
        "mean_error_m": float(s.mean()),
        "median_error_m": float(s.median()),
        "p75_error_m": float(s.quantile(0.75)),
        "p90_error_m": float(s.quantile(0.90)),
        "p95_error_m": float(s.quantile(0.95)),
        "max_error_m": float(s.max()),
    }


def process_one_volunteer(vol_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Returns:
      - detailed matched dataframe with error_m
      - summary rows dataframe
      - text lines for report
    """
    in_file = vol_dir / "gps_at_glh_timestamps_with_tiers.csv"
    if not in_file.exists():
        raise FileNotFoundError(f"Missing file: {in_file}")

    df = pd.read_csv(in_file)

    # ensure expected columns
    required = ["lat", "lon", "gps_lat_interp", "gps_lon_interp", "gps_interp_ok"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{vol_dir.name}: missing required columns: {missing}")

    # compute valid mask
    ok = matched_mask(df)

    # compute spatial error on valid rows
    df["error_m"] = np.nan
    df.loc[ok, "error_m"] = haversine_m(
        df.loc[ok, "lat"],
        df.loc[ok, "lon"],
        df.loc[ok, "gps_lat_interp"],
        df.loc[ok, "gps_lon_interp"],
    )

    # save enriched per-volunteer dataset
    out_detailed = vol_dir / "gps_at_glh_with_error.csv"
    df.to_csv(out_detailed, index=False)

    # overall stats
    total_points = int(len(df))
    matched_points = int(ok.sum())
    point_match_rate = (matched_points / total_points * 100.0) if total_points else 0.0

    total_segments = int(df["segment_id"].nunique(dropna=True)) if "segment_id" in df.columns else 0
    matched_segments = int(df.loc[ok, "segment_id"].nunique(dropna=True)) if "segment_id" in df.columns else 0
    seg_match_rate = (matched_segments / total_segments * 100.0) if total_segments else 0.0

    total_journeys = int(df["journey_id"].nunique(dropna=True)) if "journey_id" in df.columns else 0
    matched_journeys = int(df.loc[ok, "journey_id"].nunique(dropna=True)) if "journey_id" in df.columns else 0
    journey_match_rate = (matched_journeys / total_journeys * 100.0) if total_journeys else 0.0

    err_stats = describe_error(df.loc[ok, "error_m"])

    # summary row
    summary_row = {
        "anon_id": vol_dir.name,
        "total_points": total_points,
        "matched_points": matched_points,
        "point_match_rate_pct": point_match_rate,
        "total_segments": total_segments,
        "matched_segments": matched_segments,
        "segment_match_rate_pct": seg_match_rate,
        "total_journeys": total_journeys,
        "matched_journeys": matched_journeys,
        "journey_match_rate_pct": journey_match_rate,
        **err_stats,
    }

    # tier stats
    tier_rows = []
    if "match_quality_tier" in df.columns:
        for tier, sub in df.loc[ok].groupby("match_quality_tier", dropna=False):
            d = describe_error(sub["error_m"])
            tier_rows.append({
                "anon_id": vol_dir.name,
                "tier": tier,
                **d,
            })

    summary_df = pd.DataFrame([summary_row])
    tier_df = pd.DataFrame(tier_rows)

    # write per-volunteer tier summary
    if len(tier_df):
        tier_df.to_csv(vol_dir / "accuracy_by_tier.csv", index=False)

    # write per-volunteer one-line summary
    pd.DataFrame([summary_row]).to_csv(vol_dir / "accuracy_summary.csv", index=False)

    # text report
    lines = []
    lines.append(f"Volunteer: {vol_dir.name}")
    lines.append("")
    lines.append("COVERAGE")
    lines.append(f"  Total points: {total_points}")
    lines.append(f"  Matched points: {matched_points} ({point_match_rate:.2f}%)")
    lines.append(f"  Total segments: {total_segments}")
    lines.append(f"  Matched segments: {matched_segments} ({seg_match_rate:.2f}%)")
    lines.append(f"  Total journeys: {total_journeys}")
    lines.append(f"  Matched journeys: {matched_journeys} ({journey_match_rate:.2f}%)")
    lines.append("")
    lines.append("SPATIAL ERROR (meters)")
    lines.append(f"  Mean: {summary_row['mean_error_m']:.3f}" if pd.notna(summary_row["mean_error_m"]) else "  Mean: nan")
    lines.append(f"  Median: {summary_row['median_error_m']:.3f}" if pd.notna(summary_row["median_error_m"]) else "  Median: nan")
    lines.append(f"  75th pct: {summary_row['p75_error_m']:.3f}" if pd.notna(summary_row["p75_error_m"]) else "  75th pct: nan")
    lines.append(f"  90th pct: {summary_row['p90_error_m']:.3f}" if pd.notna(summary_row["p90_error_m"]) else "  90th pct: nan")
    lines.append(f"  95th pct: {summary_row['p95_error_m']:.3f}" if pd.notna(summary_row["p95_error_m"]) else "  95th pct: nan")
    lines.append(f"  Max: {summary_row['max_error_m']:.3f}" if pd.notna(summary_row["max_error_m"]) else "  Max: nan")

    if len(tier_df):
        lines.append("")
        lines.append("BY TIER")
        for _, r in tier_df.iterrows():
            tier_name = r["tier"]
            lines.append(
                f"  {tier_name}: n={int(r['n'])}, median={r['median_error_m']:.3f}, p90={r['p90_error_m']:.3f}"
            )

    (vol_dir / "accuracy_report.txt").write_text("\n".join(lines), encoding="utf-8")

    return df, pd.concat([summary_df, tier_df], ignore_index=True, sort=False), lines


def main():
    interim_root = Path("interim")
    if not interim_root.exists():
        raise FileNotFoundError("interim/ folder not found.")

    volunteer_dirs = sorted([
        p for p in interim_root.iterdir()
        if p.is_dir() and p.name.upper().startswith("V")
    ])

    if not volunteer_dirs:
        raise FileNotFoundError("No anonymized volunteer folders found under interim/ (expected V001, V002, ...).")

    overall_summary_rows = []
    overall_tier_rows = []
    overall_report_lines = []

    print(f"Found {len(volunteer_dirs)} volunteer folders.")

    for vol_dir in volunteer_dirs:
        target = vol_dir / "gps_at_glh_timestamps_with_tiers.csv"
        if not target.exists():
            print(f"Skipping {vol_dir.name}: missing gps_at_glh_timestamps_with_tiers.csv")
            continue

        print(f"Processing {vol_dir.name} ...")
        df, summaries, lines = process_one_volunteer(vol_dir)

        # first row is overall, later rows may be tiers
        if len(summaries):
            overall_rows = summaries[summaries["anon_id"].eq(vol_dir.name) & summaries["tier"].isna()] if "tier" in summaries.columns else summaries.iloc[:1]
            if len(overall_rows):
                overall_summary_rows.append(overall_rows.iloc[0].to_dict())

            if "tier" in summaries.columns:
                tiers = summaries[summaries["tier"].notna()]
                if len(tiers):
                    overall_tier_rows.append(tiers)

        overall_report_lines.extend(lines)
        overall_report_lines.append("")
        overall_report_lines.append("=" * 60)
        overall_report_lines.append("")

    if not overall_summary_rows:
        raise RuntimeError("No volunteer summaries were produced.")

    overall_summary_df = pd.DataFrame(overall_summary_rows).sort_values("anon_id").reset_index(drop=True)
    overall_summary_df.to_csv(interim_root / "accuracy_summary_all_volunteers.csv", index=False)

    if overall_tier_rows:
        overall_tier_df = pd.concat(overall_tier_rows, ignore_index=True)
        overall_tier_df.to_csv(interim_root / "accuracy_by_tier_all_volunteers.csv", index=False)
    else:
        overall_tier_df = pd.DataFrame()

    # overall pooled metrics across volunteers
    pooled = {
        "volunteers_processed": int(len(overall_summary_df)),
        "total_points": int(overall_summary_df["total_points"].sum()),
        "matched_points": int(overall_summary_df["matched_points"].sum()),
        "total_segments": int(overall_summary_df["total_segments"].sum()),
        "matched_segments": int(overall_summary_df["matched_segments"].sum()),
        "total_journeys": int(overall_summary_df["total_journeys"].sum()),
        "matched_journeys": int(overall_summary_df["matched_journeys"].sum()),
    }
    pooled["point_match_rate_pct"] = pooled["matched_points"] / pooled["total_points"] * 100 if pooled["total_points"] else 0.0
    pooled["segment_match_rate_pct"] = pooled["matched_segments"] / pooled["total_segments"] * 100 if pooled["total_segments"] else 0.0
    pooled["journey_match_rate_pct"] = pooled["matched_journeys"] / pooled["total_journeys"] * 100 if pooled["total_journeys"] else 0.0

    pooled_df = pd.DataFrame([pooled])
    pooled_df.to_csv(interim_root / "accuracy_overall_pooled_counts.csv", index=False)

    # overall text report
    header = []
    header.append("ALL VOLUNTEERS ACCURACY SUMMARY")
    header.append("")
    header.append(f"Volunteers processed: {pooled['volunteers_processed']}")
    header.append(f"Total points: {pooled['total_points']}")
    header.append(f"Matched points: {pooled['matched_points']} ({pooled['point_match_rate_pct']:.2f}%)")
    header.append(f"Total segments: {pooled['total_segments']}")
    header.append(f"Matched segments: {pooled['matched_segments']} ({pooled['segment_match_rate_pct']:.2f}%)")
    header.append(f"Total journeys: {pooled['total_journeys']}")
    header.append(f"Matched journeys: {pooled['matched_journeys']} ({pooled['journey_match_rate_pct']:.2f}%)")
    header.append("")
    header.append("=" * 60)
    header.append("")

    all_text = "\n".join(header + overall_report_lines)
    (interim_root / "accuracy_report_all_volunteers.txt").write_text(all_text, encoding="utf-8")

    print("\nSaved:")
    print(" -", interim_root / "accuracy_summary_all_volunteers.csv")
    print(" -", interim_root / "accuracy_by_tier_all_volunteers.csv")
    print(" -", interim_root / "accuracy_overall_pooled_counts.csv")
    print(" -", interim_root / "accuracy_report_all_volunteers.txt")
    print("\nDone.")


if __name__ == "__main__":
    main()