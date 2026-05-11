from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2


def haversine_m(lat1, lon1, lat2, lon2):
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


def section(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def matched_mask(df: pd.DataFrame) -> pd.Series:
    interp_ok = df["gps_interp_ok"].astype(str).str.lower().isin(["true", "1", "yes"])
    return (
        interp_ok
        & df["gps_lat_interp"].notna()
        & df["gps_lon_interp"].notna()
        & df["lat"].notna()
        & df["lon"].notna()
    )


def error_stats(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anon_id", required=True, help="Anonymized volunteer ID, e.g. V001")
    ap.add_argument("--out_root", default="interim")
    ap.add_argument("--high_error_threshold", type=float, default=200.0)
    args = ap.parse_args()

    base_dir = Path(args.out_root) / args.anon_id
    in_file = base_dir / "gps_at_glh_timestamps_with_tiers.csv"
    if not in_file.exists():
        raise FileNotFoundError(f"Missing: {in_file}")

    df = pd.read_csv(in_file)
    ok = matched_mask(df)

    # recompute error robustly
    df["error_m"] = np.nan
    df.loc[ok, "error_m"] = haversine_m(
        df.loc[ok, "lat"],
        df.loc[ok, "lon"],
        df.loc[ok, "gps_lat_interp"],
        df.loc[ok, "gps_lon_interp"],
    )

    valid = df.loc[ok].copy()

    # save enriched copy
    out_enriched = base_dir / f"debug_{args.anon_id}_with_error.csv"
    valid.to_csv(out_enriched, index=False)

    # overall
    section("OVERALL")
    stats = error_stats(valid["error_m"])
    print(f"Volunteer: {args.anon_id}")
    print(f"Valid matched rows: {stats['count']}")
    print(f"Mean error (m):   {stats['mean']:.3f}")
    print(f"Median error (m): {stats['median']:.3f}")
    print(f"75th pct (m):     {stats['p75']:.3f}")
    print(f"90th pct (m):     {stats['p90']:.3f}")
    print(f"95th pct (m):     {stats['p95']:.3f}")
    print(f"Max error (m):    {stats['max']:.3f}")

    # tier
    if "match_quality_tier" in valid.columns:
        section("BY TIER")
        tier_stats = (
            valid.groupby("match_quality_tier", dropna=False)["error_m"]
            .agg(["count", "mean", "median", "max"])
            .sort_index()
        )
        print(tier_stats.to_string())

    # exact matches
    if "bracket_gap_s" in valid.columns:
        section("EXACT TIMESTAMP MATCHES (bracket_gap_s == 0)")
        exact = valid[pd.to_numeric(valid["bracket_gap_s"], errors="coerce") == 0].copy()
        estats = error_stats(exact["error_m"])
        print(f"Rows: {estats['count']}")
        if estats["count"] > 0:
            print(f"Median error (m): {estats['median']:.3f}")
            print(f"90th pct (m):     {estats['p90']:.3f}")
            print(f"Max error (m):    {estats['max']:.3f}")
            print("\nSample:")
            cols = [
                "timestamp_utc", "lat", "lon",
                "gps_lat_interp", "gps_lon_interp",
                "bracket_gap_s", "error_m"
            ]
            cols = [c for c in cols if c in exact.columns]
            print(exact[cols].head(20).to_string(index=False))

    # thresholds
    section("HIGH-ERROR COUNTS")
    for th in [50, 100, 200, 500, 1000]:
        n = int((valid["error_m"] > th).sum())
        pct = n / len(valid) * 100 if len(valid) else 0.0
        print(f"> {th:4.0f} m : {n:5d} ({pct:6.2f}%)")

    # top bad rows
    section(f"HIGH-ERROR ROWS (>{args.high_error_threshold:.0f}m)")
    bad = valid[valid["error_m"] > args.high_error_threshold].copy()
    print(f"Count: {len(bad)}")
    if len(bad):
        cols = [
            "timestamp_utc", "lat", "lon",
            "gps_lat_interp", "gps_lon_interp",
            "bracket_gap_s", "match_quality_tier",
            "segment_id", "journey_id", "error_m"
        ]
        cols = [c for c in cols if c in bad.columns]
        print(bad[cols].head(30).to_string(index=False))
        bad.to_csv(base_dir / f"debug_{args.anon_id}_high_error_points.csv", index=False)

    # duplicates in interpolated GPS
    section("DUPLICATE INTERPOLATED GPS COORDS")
    dup = valid.duplicated(subset=["timestamp_utc", "gps_lat_interp", "gps_lon_interp"], keep=False)
    print("Duplicate matched rows:", int(dup.sum()))

    # by segment
    if "segment_id" in valid.columns:
        section("TOP SEGMENTS BY MEDIAN ERROR")
        seg = (
            valid.groupby("segment_id")
            .agg(
                n=("error_m", "size"),
                mean_error_m=("error_m", "mean"),
                median_error_m=("error_m", "median"),
                p90_error_m=("error_m", lambda x: x.quantile(0.90)),
                max_error_m=("error_m", "max"),
                high_error_n=("error_m", lambda x: (x > args.high_error_threshold).sum()),
            )
            .sort_values(["median_error_m", "max_error_m"], ascending=False)
            .reset_index()
        )
        print(seg.head(20).to_string(index=False))
        seg.to_csv(base_dir / f"debug_{args.anon_id}_segment_error_profile.csv", index=False)

    # by journey
    if "journey_id" in valid.columns:
        section("TOP JOURNEYS BY MEDIAN ERROR")
        jny = (
            valid.groupby("journey_id")
            .agg(
                n=("error_m", "size"),
                mean_error_m=("error_m", "mean"),
                median_error_m=("error_m", "median"),
                p90_error_m=("error_m", lambda x: x.quantile(0.90)),
                max_error_m=("error_m", "max"),
                high_error_n=("error_m", lambda x: (x > args.high_error_threshold).sum()),
            )
            .sort_values(["median_error_m", "max_error_m"], ascending=False)
            .reset_index()
        )
        print(jny.head(20).to_string(index=False))
        jny.to_csv(base_dir / f"debug_{args.anon_id}_journey_error_profile.csv", index=False)

    # compact text report
    report_lines = []
    report_lines.append(f"Volunteer: {args.anon_id}")
    report_lines.append("")
    report_lines.append("OVERALL")
    report_lines.append(f"  Valid matched rows: {stats['count']}")
    report_lines.append(f"  Mean error (m): {stats['mean']:.3f}")
    report_lines.append(f"  Median error (m): {stats['median']:.3f}")
    report_lines.append(f"  75th pct (m): {stats['p75']:.3f}")
    report_lines.append(f"  90th pct (m): {stats['p90']:.3f}")
    report_lines.append(f"  95th pct (m): {stats['p95']:.3f}")
    report_lines.append(f"  Max error (m): {stats['max']:.3f}")
    report_lines.append("")
    for th in [50, 100, 200, 500, 1000]:
        n = int((valid["error_m"] > th).sum())
        pct = n / len(valid) * 100 if len(valid) else 0.0
        report_lines.append(f"  > {th} m: {n} ({pct:.2f}%)")

    (base_dir / f"debug_{args.anon_id}_error_report.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print("\nSaved:")
    print(" -", out_enriched)
    if len(bad):
        print(" -", base_dir / f"debug_{args.anon_id}_high_error_points.csv")
    if "segment_id" in valid.columns:
        print(" -", base_dir / f"debug_{args.anon_id}_segment_error_profile.csv")
    if "journey_id" in valid.columns:
        print(" -", base_dir / f"debug_{args.anon_id}_journey_error_profile.csv")
    print(" -", base_dir / f"debug_{args.anon_id}_error_report.txt")


if __name__ == "__main__":
    main()