from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def matched_mask(df: pd.DataFrame) -> pd.Series:
    interp_ok = df["gps_interp_ok"].astype(str).str.lower().isin(["true", "1", "yes"])
    return (
        interp_ok
        & df["gps_lat_interp"].notna()
        & df["gps_lon_interp"].notna()
        & df["lat"].notna()
        & df["lon"].notna()
        & df["error_m"].notna()
    )


def ensure_error_column(df: pd.DataFrame) -> pd.DataFrame:
    # This script expects files produced by debug_volunteer_error_profile.py
    # which already contain error_m.
    if "error_m" not in df.columns:
        raise ValueError(
            "Missing error_m column. Run debug_volunteer_error_profile.py first "
            "for the relevant volunteers."
        )
    return df


def segment_profile(df_valid: pd.DataFrame, high_error_threshold: float) -> pd.DataFrame:
    prof = (
        df_valid.groupby("segment_id")
        .agg(
            n=("error_m", "size"),
            mean_error_m=("error_m", "mean"),
            median_error_m=("error_m", "median"),
            p90_error_m=("error_m", lambda x: x.quantile(0.90)),
            max_error_m=("error_m", "max"),
            high_error_n=("error_m", lambda x: (x > high_error_threshold).sum()),
        )
        .reset_index()
    )
    prof["high_error_frac"] = prof["high_error_n"] / prof["n"]
    return prof


def describe_error(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return {
            "count": 0,
            "mean_error_m": np.nan,
            "median_error_m": np.nan,
            "p75_error_m": np.nan,
            "p90_error_m": np.nan,
            "p95_error_m": np.nan,
            "max_error_m": np.nan,
        }

    return {
        "count": int(len(s)),
        "mean_error_m": float(s.mean()),
        "median_error_m": float(s.median()),
        "p75_error_m": float(s.quantile(0.75)),
        "p90_error_m": float(s.quantile(0.90)),
        "p95_error_m": float(s.quantile(0.95)),
        "max_error_m": float(s.max()),
    }


def process_one(
    vol_dir: Path,
    median_threshold: float,
    high_error_threshold: float,
    high_error_frac_threshold: float,
) -> dict | None:
    in_file = vol_dir / f"debug_{vol_dir.name}_with_error.csv"
    if not in_file.exists():
        print(f"Skipping {vol_dir.name}: missing {in_file.name}")
        return None

    df = pd.read_csv(in_file)
    df = ensure_error_column(df)

    valid = df.loc[matched_mask(df)].copy()
    if len(valid) == 0:
        print(f"Skipping {vol_dir.name}: no valid matched rows.")
        return None

    seg_prof = segment_profile(valid, high_error_threshold)

    bad_seg_mask = (
        (seg_prof["median_error_m"] > median_threshold)
        | (seg_prof["high_error_frac"] > high_error_frac_threshold)
    )
    bad_segments = seg_prof.loc[bad_seg_mask, "segment_id"].tolist()

    # Save segment profile and removal list
    seg_prof = seg_prof.sort_values(
        ["median_error_m", "high_error_frac", "max_error_m"],
        ascending=False
    ).reset_index(drop=True)
    seg_prof["remove_flag"] = seg_prof["segment_id"].isin(bad_segments)

    out_profile = vol_dir / "segment_filter_profile.csv"
    seg_prof.to_csv(out_profile, index=False)

    out_bad = vol_dir / "bad_segments_to_remove.csv"
    pd.DataFrame({"segment_id": bad_segments}).to_csv(out_bad, index=False)

    # Filter full matched table, not just valid rows
    original_full = pd.read_csv(vol_dir / "gps_at_glh_timestamps_with_tiers.csv")
    filtered_full = original_full.loc[~original_full["segment_id"].isin(bad_segments)].copy()
    out_filtered_full = vol_dir / "gps_at_glh_timestamps_with_tiers_segment_filtered.csv"
    filtered_full.to_csv(out_filtered_full, index=False)

    # Filter valid-with-error table
    filtered_valid = valid.loc[~valid["segment_id"].isin(bad_segments)].copy()
    out_filtered_valid = vol_dir / "debug_with_error_segment_filtered.csv"
    filtered_valid.to_csv(out_filtered_valid, index=False)

    before = describe_error(valid["error_m"])
    after = describe_error(filtered_valid["error_m"])

    # Journey coverage after filtering
    before_journeys = int(valid["journey_id"].nunique(dropna=True)) if "journey_id" in valid.columns else 0
    after_journeys = int(filtered_valid["journey_id"].nunique(dropna=True)) if "journey_id" in filtered_valid.columns else 0

    report_lines = [
        f"Volunteer: {vol_dir.name}",
        "",
        "FILTER RULE",
        f"  Remove segment if median_error_m > {median_threshold}",
        f"  OR high_error_frac > {high_error_frac_threshold}",
        f"  where high_error means error_m > {high_error_threshold}",
        "",
        "REMOVAL SUMMARY",
        f"  Bad segments removed: {len(bad_segments)}",
        f"  Journey count before: {before_journeys}",
        f"  Journey count after: {after_journeys}",
        "",
        "BEFORE FILTER",
        f"  Rows: {before['count']}",
        f"  Mean error (m): {before['mean_error_m']:.3f}" if pd.notna(before["mean_error_m"]) else "  Mean error (m): nan",
        f"  Median error (m): {before['median_error_m']:.3f}" if pd.notna(before["median_error_m"]) else "  Median error (m): nan",
        f"  75th pct (m): {before['p75_error_m']:.3f}" if pd.notna(before["p75_error_m"]) else "  75th pct (m): nan",
        f"  90th pct (m): {before['p90_error_m']:.3f}" if pd.notna(before["p90_error_m"]) else "  90th pct (m): nan",
        f"  95th pct (m): {before['p95_error_m']:.3f}" if pd.notna(before["p95_error_m"]) else "  95th pct (m): nan",
        f"  Max error (m): {before['max_error_m']:.3f}" if pd.notna(before["max_error_m"]) else "  Max error (m): nan",
        "",
        "AFTER FILTER",
        f"  Rows: {after['count']}",
        f"  Mean error (m): {after['mean_error_m']:.3f}" if pd.notna(after["mean_error_m"]) else "  Mean error (m): nan",
        f"  Median error (m): {after['median_error_m']:.3f}" if pd.notna(after["median_error_m"]) else "  Median error (m): nan",
        f"  75th pct (m): {after['p75_error_m']:.3f}" if pd.notna(after["p75_error_m"]) else "  75th pct (m): nan",
        f"  90th pct (m): {after['p90_error_m']:.3f}" if pd.notna(after["p90_error_m"]) else "  90th pct (m): nan",
        f"  95th pct (m): {after['p95_error_m']:.3f}" if pd.notna(after["p95_error_m"]) else "  95th pct (m): nan",
        f"  Max error (m): {after['max_error_m']:.3f}" if pd.notna(after["max_error_m"]) else "  Max error (m): nan",
    ]

    out_report = vol_dir / "segment_filter_report.txt"
    out_report.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "anon_id": vol_dir.name,
        "bad_segments_removed": len(bad_segments),
        "before_rows": before["count"],
        "after_rows": after["count"],
        "before_median_error_m": before["median_error_m"],
        "after_median_error_m": after["median_error_m"],
        "before_p90_error_m": before["p90_error_m"],
        "after_p90_error_m": after["p90_error_m"],
        "before_max_error_m": before["max_error_m"],
        "after_max_error_m": after["max_error_m"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="interim")
    ap.add_argument(
        "--median_threshold",
        type=float,
        default=100.0,
        help="Remove segment if median_error_m is larger than this."
    )
    ap.add_argument(
        "--high_error_threshold",
        type=float,
        default=200.0,
        help="Threshold used to define a high-error point."
    )
    ap.add_argument(
        "--high_error_frac_threshold",
        type=float,
        default=0.30,
        help="Remove segment if fraction of high-error points is larger than this."
    )
    ap.add_argument(
        "--anon_id",
        default=None,
        help="Optional single volunteer ID, e.g. V004. If omitted, process all Vxxx folders."
    )
    args = ap.parse_args()

    root = Path(args.out_root)
    if not root.exists():
        raise FileNotFoundError(f"Missing folder: {root}")

    if args.anon_id:
        vol_dirs = [root / args.anon_id]
    else:
        vol_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.upper().startswith("V")])

    rows = []
    for vol_dir in vol_dirs:
        if not vol_dir.exists():
            print(f"Skipping {vol_dir}: folder not found.")
            continue

        print(f"Processing {vol_dir.name} ...")
        result = process_one(
            vol_dir=vol_dir,
            median_threshold=args.median_threshold,
            high_error_threshold=args.high_error_threshold,
            high_error_frac_threshold=args.high_error_frac_threshold,
        )
        if result is not None:
            rows.append(result)

    if rows:
        summary = pd.DataFrame(rows).sort_values("anon_id").reset_index(drop=True)
        out_summary = root / "segment_filter_summary_all.csv"
        summary.to_csv(out_summary, index=False)
        print("\nSaved:")
        print(" -", out_summary)
    else:
        print("No volunteers processed.")


if __name__ == "__main__":
    main()