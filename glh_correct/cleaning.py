"""
cleaning.py
===========
Quality-control filters for GLH and GPX point streams.

Design principles
-----------------
1. **Flag-then-apply.** Each filter adds a boolean `qc_<name>` column rather
   than dropping rows. A separate `apply_qc()` step prunes the DataFrame
   based on a policy. This preserves an audit trail of every dropped row
   and lets the same parsed data be re-cleaned under different policies.

2. **Tag-don't-drop for out-of-area points.** Subjects travel outside
   the study bbox occasionally; we want to study those points separately
   rather than silently losing them. `in_study_area` is therefore a tag,
   never a drop, in the default policy.

3. **Per-filter audit.** `clean_glh()` and `clean_gpx()` return both the
   cleaned DataFrame and an `audit` dict reporting how many rows each step
   touched. This dict is later folded into the per-track Stage 1
   audit report.

QC columns added (where applicable)
-----------------------------------
    qc_in_study_area  : bool  — point lies inside the Edinburgh bbox
    qc_accuracy_pass  : bool  — accuracy_m <= max_acc_m (GLH only)
    qc_speed_pass     : bool  — implied speed to previous point <= max
    qc_dedup_keep     : bool  — first occurrence of a duplicate timestamp
    qc_session_pass   : bool  — point is in a session that meets min size

A row passes overall cleaning if all `qc_*` columns are True except
`qc_in_study_area`, which is preserved as a tag.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .projection import bng_distance


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

# Edinburgh study area (a bit larger than the OS Highways tight bbox so we
# don't trim points right at the edges). Lat/Lon WGS84 corners.
EDINBURGH_BBOX_WGS84 = {
    "lat_min": 55.80, "lat_max": 56.05,
    "lon_min": -3.40, "lon_max": -3.00,
}

DEFAULT_MAX_ACCURACY_M = 50.0       # drop GLH points coarser than this
DEFAULT_MAX_SPEED_KMH = 200.0       # drop points implying impossible speed
DEFAULT_MIN_SESSION_POINTS = 5
DEFAULT_MIN_SESSION_DURATION_S = 60

# Google Location History uses a sentinel value of 100 m when the true
# accuracy is unknown. In some exports the accuracy distribution has a
# hard spike at exactly 100 m irrespective of source. Treating these as
# "unknown" (i.e. letting them pass the accuracy filter) is more honest
# than dropping them.
GLH_ACCURACY_UNKNOWN_SENTINEL = 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Individual filters — each adds a column rather than dropping rows
# ─────────────────────────────────────────────────────────────────────────────

def tag_in_study_area(
    df: pd.DataFrame,
    *,
    bbox: dict = EDINBURGH_BBOX_WGS84,
    lat_col: str = "lat",
    lon_col: str = "lon",
    out_col: str = "qc_in_study_area",
) -> pd.DataFrame:
    """Add a boolean `qc_in_study_area` column. Does not drop rows."""
    out = df.copy()
    if df.empty:
        out[out_col] = pd.Series(dtype=bool)
        return out
    out[out_col] = (
        (out[lat_col] >= bbox["lat_min"]) & (out[lat_col] <= bbox["lat_max"]) &
        (out[lon_col] >= bbox["lon_min"]) & (out[lon_col] <= bbox["lon_max"])
    )
    return out


def filter_accuracy(
    df: pd.DataFrame,
    *,
    max_accuracy_m: float = DEFAULT_MAX_ACCURACY_M,
    accuracy_col: str = "accuracy_m",
    out_col: str = "qc_accuracy_pass",
    treat_sentinel_as_unknown: bool = True,
    sentinel: float = GLH_ACCURACY_UNKNOWN_SENTINEL,
) -> pd.DataFrame:
    """
    Add `qc_accuracy_pass` boolean: True if accuracy_m <= threshold.

    Rows with NaN accuracy pass by default (we don't know they're bad). GPX
    has no accuracy column; this filter is a no-op there.

    When `treat_sentinel_as_unknown` is True (default), rows whose accuracy
    equals `sentinel` (100 m, the Google "unknown" placeholder) are treated
    as NaN and pass the filter. This is needed because some exports have
    a large fraction of points with accuracy=100 as a sentinel rather than
    a real measurement; filtering on it would remove most of their data.
    """
    out = df.copy()
    if accuracy_col not in out.columns:
        out[out_col] = True
        return out
    acc = out[accuracy_col]
    is_sentinel = (acc == sentinel) if treat_sentinel_as_unknown else pd.Series(False, index=out.index)
    out[out_col] = acc.isna() | is_sentinel | (acc <= max_accuracy_m)
    return out


def filter_speed(
    df: pd.DataFrame,
    *,
    max_speed_kmh: float = DEFAULT_MAX_SPEED_KMH,
    east_col: str = "east",
    north_col: str = "north",
    time_col: str = "timestamp",
    session_col: str = "session_id",
    out_col: str = "qc_speed_pass",
    speed_col: str = "implied_speed_kmh",
) -> pd.DataFrame:
    """
    Compute implied speed from successive points (within the same session
    when a session column is present) and flag rows whose implied speed
    exceeds `max_speed_kmh`. The implied speed is the displacement to the
    previous point divided by the time gap; the first point of each session
    has NaN speed and passes by default.

    Adds columns: `implied_speed_kmh` and `qc_speed_pass`.

    Requires BNG `east`/`north` columns. Run `projection.add_bng_columns`
    first.
    """
    out = df.copy()
    if out.empty:
        out[speed_col] = pd.Series(dtype=float)
        out[out_col] = pd.Series(dtype=bool)
        return out

    has_session = session_col in out.columns
    group_keys = [session_col] if has_session else None

    if has_session:
        prev_east = out.groupby(session_col)[east_col].shift()
        prev_north = out.groupby(session_col)[north_col].shift()
        prev_time = out.groupby(session_col)[time_col].shift()
    else:
        prev_east = out[east_col].shift()
        prev_north = out[north_col].shift()
        prev_time = out[time_col].shift()

    dist_m = bng_distance(prev_east, prev_north, out[east_col], out[north_col])
    dt_s = (out[time_col] - prev_time).dt.total_seconds()

    with np.errstate(divide="ignore", invalid="ignore"):
        speed_mps = np.where(dt_s > 0, dist_m / dt_s, np.nan)
    speed_kmh = speed_mps * 3.6

    out[speed_col] = speed_kmh
    out[out_col] = np.isnan(speed_kmh) | (speed_kmh <= max_speed_kmh)
    return out


def dedupe_timestamps(
    df: pd.DataFrame,
    *,
    time_col: str = "timestamp",
    session_col: str = "session_id",
    out_col: str = "qc_dedup_keep",
) -> pd.DataFrame:
    """
    Mark first occurrence of each (session, timestamp) as True, duplicates False.

    If no session column is present, dedup is global on timestamp alone.
    """
    out = df.copy()
    if out.empty:
        out[out_col] = pd.Series(dtype=bool)
        return out
    if session_col in out.columns:
        keys = [session_col, time_col]
    else:
        keys = [time_col]
    is_dup = out.duplicated(subset=keys, keep="first")
    out[out_col] = ~is_dup
    return out


def filter_short_sessions(
    df: pd.DataFrame,
    *,
    min_points: int = DEFAULT_MIN_SESSION_POINTS,
    min_duration_s: float = DEFAULT_MIN_SESSION_DURATION_S,
    session_col: str = "session_id",
    time_col: str = "timestamp",
    out_col: str = "qc_session_pass",
) -> pd.DataFrame:
    """
    Flag rows belonging to sessions that fail the minimum-size policy.

    A session passes if it has >= `min_points` AND spans >= `min_duration_s`
    seconds. No-op if no session column is present.
    """
    out = df.copy()
    if session_col not in out.columns or out.empty:
        out[out_col] = True
        return out

    g = out.groupby(session_col)
    n_points = g[time_col].size()
    duration = (g[time_col].max() - g[time_col].min()).dt.total_seconds()
    session_ok = (n_points >= min_points) & (duration >= min_duration_s)
    keep_mask = out[session_col].map(session_ok).fillna(False)
    out[out_col] = keep_mask.astype(bool)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Apply step — combine flags into a single boolean and prune
# ─────────────────────────────────────────────────────────────────────────────

#: QC flags whose failure causes a row to be dropped by default.
DEFAULT_DROP_FLAGS = (
    "qc_accuracy_pass",
    "qc_speed_pass",
    "qc_dedup_keep",
    "qc_session_pass",
)
# Note: qc_in_study_area is intentionally NOT in DROP_FLAGS — it is a tag.


def apply_qc(
    df: pd.DataFrame,
    *,
    drop_flags: Iterable[str] = DEFAULT_DROP_FLAGS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply the QC policy.

    Returns
    -------
    kept : DataFrame
        Rows where all named drop_flags are True. Original index preserved.
    dropped : DataFrame
        Rows that failed at least one drop_flag, with an extra column
        `drop_reason` listing the failing flags.
    """
    if df.empty:
        return df.copy(), df.copy().assign(drop_reason="")

    present_flags = [f for f in drop_flags if f in df.columns]
    if not present_flags:
        return df.copy(), df.iloc[:0].copy().assign(drop_reason="")

    keep_mask = df[present_flags].all(axis=1)

    kept = df[keep_mask].copy()

    dropped = df[~keep_mask].copy()
    if not dropped.empty:
        def _reasons(row):
            return ",".join(f for f in present_flags if not row[f])
        dropped["drop_reason"] = dropped[present_flags].apply(_reasons, axis=1)

    return kept, dropped


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrators — one call per stream
# ─────────────────────────────────────────────────────────────────────────────

def clean_glh(
    df: pd.DataFrame,
    *,
    bbox: dict = EDINBURGH_BBOX_WGS84,
    max_accuracy_m: float = DEFAULT_MAX_ACCURACY_M,
    max_speed_kmh: float = DEFAULT_MAX_SPEED_KMH,
    min_session_points: int = DEFAULT_MIN_SESSION_POINTS,
    min_session_duration_s: float = DEFAULT_MIN_SESSION_DURATION_S,
) -> tuple[pd.DataFrame, dict]:
    """
    Run the full GLH cleaning pipeline on a points DataFrame.

    Expects columns: `lat`, `lon`, `accuracy_m` (optional), `east`, `north`,
    `timestamp`. If `session_id` is present it is used for grouping; otherwise
    speed/session filters fall back to global behaviour.

    Returns
    -------
    cleaned : DataFrame
        QC-passing rows, all qc_* columns retained for inspection.
    audit : dict
        Per-step row counts.
    """
    audit: dict = {"input_rows": len(df)}
    if df.empty:
        return df.copy(), audit

    out = df.copy()
    out = tag_in_study_area(out, bbox=bbox)
    audit["in_study_area"] = int(out["qc_in_study_area"].sum())
    audit["out_of_study_area"] = int((~out["qc_in_study_area"]).sum())

    out = filter_accuracy(out, max_accuracy_m=max_accuracy_m)
    audit["accuracy_fail"] = int((~out["qc_accuracy_pass"]).sum())

    out = filter_speed(out, max_speed_kmh=max_speed_kmh)
    audit["speed_fail"] = int((~out["qc_speed_pass"]).sum())

    out = dedupe_timestamps(out)
    audit["dedup_drop"] = int((~out["qc_dedup_keep"]).sum())

    out = filter_short_sessions(
        out,
        min_points=min_session_points,
        min_duration_s=min_session_duration_s,
    )
    audit["short_session_drop"] = int((~out["qc_session_pass"]).sum())

    kept, dropped = apply_qc(out)
    audit["kept_rows"] = len(kept)
    audit["dropped_rows"] = len(dropped)

    return kept, audit


def clean_glh_timeline_paths(
    df: pd.DataFrame,
    *,
    bbox: dict = EDINBURGH_BBOX_WGS84,
    max_speed_kmh: float = DEFAULT_MAX_SPEED_KMH,
    min_session_points: int = 1,
    min_session_duration_s: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Cleaning pipeline tuned for GLH `timeline_paths`.

    Differs from `clean_glh()` in that the session-size minimum is relaxed
    (default: 1 point, 0 seconds). Google records timelinePath segments
    with as few as 1–4 points each — in some exports a majority of segments
    are below 5 points; the default `clean_glh()` filter would drop almost
    all of them.

    Timeline paths also have no `accuracy_m` column, so the accuracy filter
    is a no-op.
    """
    audit: dict = {"input_rows": len(df)}
    if df.empty:
        return df.copy(), audit

    out = df.copy()
    out = tag_in_study_area(out, bbox=bbox)
    audit["in_study_area"] = int(out["qc_in_study_area"].sum())
    audit["out_of_study_area"] = int((~out["qc_in_study_area"]).sum())

    # No accuracy column on timeline paths
    out["qc_accuracy_pass"] = True

    out = filter_speed(out, max_speed_kmh=max_speed_kmh)
    audit["speed_fail"] = int((~out["qc_speed_pass"]).sum())

    out = dedupe_timestamps(out)
    audit["dedup_drop"] = int((~out["qc_dedup_keep"]).sum())

    out = filter_short_sessions(
        out,
        min_points=min_session_points,
        min_duration_s=min_session_duration_s,
    )
    audit["short_session_drop"] = int((~out["qc_session_pass"]).sum())

    kept, dropped = apply_qc(out)
    audit["kept_rows"] = len(kept)
    audit["dropped_rows"] = len(dropped)

    return kept, audit


def clean_gpx(
    df: pd.DataFrame,
    *,
    bbox: dict = EDINBURGH_BBOX_WGS84,
    max_speed_kmh: float = DEFAULT_MAX_SPEED_KMH,
    min_session_points: int = DEFAULT_MIN_SESSION_POINTS,
    min_session_duration_s: float = DEFAULT_MIN_SESSION_DURATION_S,
) -> tuple[pd.DataFrame, dict]:
    """
    GPX cleaning pipeline. Same as clean_glh() minus the accuracy filter
    (GPX has no accuracy column in the project data).
    """
    audit: dict = {"input_rows": len(df)}
    if df.empty:
        return df.copy(), audit

    out = df.copy()
    out = tag_in_study_area(out, bbox=bbox)
    audit["in_study_area"] = int(out["qc_in_study_area"].sum())
    audit["out_of_study_area"] = int((~out["qc_in_study_area"]).sum())

    # No accuracy filter for GPX — set the column to True so apply_qc accepts.
    out["qc_accuracy_pass"] = True

    out = filter_speed(out, max_speed_kmh=max_speed_kmh)
    audit["speed_fail"] = int((~out["qc_speed_pass"]).sum())

    out = dedupe_timestamps(out)
    audit["dedup_drop"] = int((~out["qc_dedup_keep"]).sum())

    out = filter_short_sessions(
        out,
        min_points=min_session_points,
        min_duration_s=min_session_duration_s,
    )
    audit["short_session_drop"] = int((~out["qc_session_pass"]).sum())

    kept, dropped = apply_qc(out)
    audit["kept_rows"] = len(kept)
    audit["dropped_rows"] = len(dropped)

    return kept, audit
