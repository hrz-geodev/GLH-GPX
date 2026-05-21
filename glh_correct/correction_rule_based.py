"""
correction_rule_based.py
========================
Stage 2 — rule-based GLH corrector.

For each GLH point we already have, from `feature_engineering`:
    - `nearest_car_east/north/distance_m`  (snap to carriageway network)
    - `nearest_ped_east/north/distance_m`  (snap to pedestrian network)

The rule is:
    corrected_glh_position = whichever of the two snaps is closer
                             (provided either one returned a snap at all)

That is what the user picked: "use the nearest rule".

If neither network produced a snap within `max_snap_m` (default 100 m),
the point is left uncorrected — `corrected_glh_*` columns are NaN and
`corrected_deviation_m` is NaN. This is correct behaviour: the corrector
should not invent a position when no plausible edge exists.

Columns added
-------------
    corrected_glh_east, corrected_glh_north
    corrected_glh_lat,  corrected_glh_lon
    corrected_glh_network_kind   ('carriageway' | 'pedestrian' | None)
    corrected_snap_distance_m
    corrected_deviation_m        distance from corrected GLH to GPX truth
    improvement_m                = raw deviation_m − corrected_deviation_m
                                    positive → corrector helped
                                    negative → corrector hurt
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .projection import bng_to_wgs84, bng_distance
from .feature_engineering import add_map_context_features


def apply_rule_based_correction(
    df: pd.DataFrame,
    *,
    networks: dict | None = None,
    project_root: str = ".",
    max_snap_m: float = 100.0,
) -> pd.DataFrame:
    """
    Apply the nearest-network rule-based corrector.

    Parameters
    ----------
    df : DataFrame
        Either a Stage-1 matched DataFrame (with glh_east/glh_north and
        gpx_east/gpx_north), or any DataFrame that already has the
        `nearest_car_*` / `nearest_ped_*` columns added by
        `feature_engineering.add_map_context_features`.
    networks : dict, optional
        From `networks.load_all_networks(...)`. Loaded on demand if the
        feature columns aren't present yet.
    max_snap_m : float
        Used only if feature engineering is run lazily here.

    Returns
    -------
    DataFrame
        Copy of df with the added corrected_* and improvement_m columns.
    """
    out = df.copy()

    feature_columns_present = (
        {"nearest_car_distance_m", "nearest_ped_distance_m"}.issubset(out.columns)
    )
    if not feature_columns_present:
        # Run feature engineering on the fly. Less efficient (callers
        # should usually pre-compute features) but convenient.
        if networks is None:
            from .networks import load_all_networks
            networks = load_all_networks(project_root)
        out = add_map_context_features(
            out, networks, project_root=project_root, max_snap_m=max_snap_m
        )

    # ── Nearest-wins selection ───────────────────────────────────────────────
    car_d = out["nearest_car_distance_m"].astype(float)
    ped_d = out["nearest_ped_distance_m"].astype(float)

    car_wins = car_d.notna() & (ped_d.isna() | (car_d <= ped_d))
    ped_wins = ped_d.notna() & (~car_wins)

    out["corrected_glh_east"] = np.nan
    out["corrected_glh_north"] = np.nan
    out["corrected_glh_network_kind"] = pd.Series([pd.NA] * len(out), dtype="object")
    out["corrected_snap_distance_m"] = np.nan

    out.loc[car_wins, "corrected_glh_east"] = out.loc[car_wins, "nearest_car_east"]
    out.loc[car_wins, "corrected_glh_north"] = out.loc[car_wins, "nearest_car_north"]
    out.loc[car_wins, "corrected_glh_network_kind"] = "carriageway"
    out.loc[car_wins, "corrected_snap_distance_m"] = out.loc[car_wins, "nearest_car_distance_m"]

    out.loc[ped_wins, "corrected_glh_east"] = out.loc[ped_wins, "nearest_ped_east"]
    out.loc[ped_wins, "corrected_glh_north"] = out.loc[ped_wins, "nearest_ped_north"]
    out.loc[ped_wins, "corrected_glh_network_kind"] = "pedestrian"
    out.loc[ped_wins, "corrected_snap_distance_m"] = out.loc[ped_wins, "nearest_ped_distance_m"]

    # ── Add WGS84 versions for mapping / sanity-check ───────────────────────
    out["corrected_glh_lat"] = np.nan
    out["corrected_glh_lon"] = np.nan
    mask = out["corrected_glh_east"].notna()
    if mask.any():
        lat, lon = bng_to_wgs84(
            out.loc[mask, "corrected_glh_east"],
            out.loc[mask, "corrected_glh_north"],
        )
        out.loc[mask, "corrected_glh_lat"] = lat
        out.loc[mask, "corrected_glh_lon"] = lon

    # ── Corrected deviation and improvement (only where we have GPX truth) ──
    if {"gpx_east", "gpx_north"}.issubset(out.columns):
        out["corrected_deviation_m"] = bng_distance(
            out["corrected_glh_east"], out["corrected_glh_north"],
            out["gpx_east"], out["gpx_north"],
        )
        if "deviation_m" in out.columns:
            out["improvement_m"] = out["deviation_m"] - out["corrected_deviation_m"]
        else:
            out["improvement_m"] = np.nan
    else:
        out["corrected_deviation_m"] = np.nan
        out["improvement_m"] = np.nan

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Summary helper
# ─────────────────────────────────────────────────────────────────────────────

def correction_summary(corrected_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-(volunteer, glh_layer) before/after deviation statistics.

    Output columns:
        volunteer, glh_layer,
        n_matched, n_correctable, snap_rate,
        raw_median_m, raw_mean_m, raw_p95_m,
        cor_median_m, cor_mean_m, cor_p95_m,
        improvement_median_m, improvement_mean_m,
        pct_improved   (% of correctable points where improvement > 0)
    """
    if corrected_df.empty:
        return pd.DataFrame(columns=[
            "volunteer", "glh_layer", "n_matched", "n_correctable", "snap_rate",
            "raw_median_m", "raw_mean_m", "raw_p95_m",
            "cor_median_m", "cor_mean_m", "cor_p95_m",
            "improvement_median_m", "improvement_mean_m", "pct_improved",
        ])

    def _agg(g):
        matched_mask = g["matched"].fillna(False).astype(bool)
        n_matched = int(matched_mask.sum())
        gm = g[matched_mask]
        correctable_mask = gm["corrected_deviation_m"].notna()
        n_correctable = int(correctable_mask.sum())
        snap_rate = n_correctable / n_matched if n_matched else np.nan

        raw_dev = gm["deviation_m"].dropna()
        cor_dev = gm.loc[correctable_mask, "corrected_deviation_m"].dropna()
        improvement = gm.loc[correctable_mask, "improvement_m"].dropna()

        return pd.Series({
            "n_matched":     n_matched,
            "n_correctable": n_correctable,
            "snap_rate":     snap_rate,
            "raw_median_m":  raw_dev.median()    if len(raw_dev) else np.nan,
            "raw_mean_m":    raw_dev.mean()      if len(raw_dev) else np.nan,
            "raw_p95_m":     raw_dev.quantile(0.95) if len(raw_dev) else np.nan,
            "cor_median_m":  cor_dev.median()    if len(cor_dev) else np.nan,
            "cor_mean_m":    cor_dev.mean()      if len(cor_dev) else np.nan,
            "cor_p95_m":     cor_dev.quantile(0.95) if len(cor_dev) else np.nan,
            "improvement_median_m": improvement.median() if len(improvement) else np.nan,
            "improvement_mean_m":   improvement.mean()   if len(improvement) else np.nan,
            "pct_improved":  100.0 * (improvement > 0).mean() if len(improvement) else np.nan,
        })

    return (corrected_df.groupby(["volunteer", "glh_layer"], dropna=False)
            .apply(_agg).reset_index())
