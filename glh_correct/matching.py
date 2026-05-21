"""
matching.py
===========
Temporal matching of GLH points to GPX ground truth.

The matching produces, for each GLH observation, the position the GPX
trajectory implies at the **same timestamp**, by linear interpolation
between the two bracketing GPX points. The deviation between the two is
the per-point error.

Why interpolate GPX onto GLH timestamps (and not the other way round)?
-----------------------------------------------------------------------
- GPX is dense and regular (typically ~1 Hz); GLH is sparse and irregular.
- Interpolating dense GPX is well-defined and low-error.
- GLH is the variable under study — we want one truth value per GLH
  observation, not the other way round.

Algorithm
---------
1. Compute session summaries for both streams.
2. For a given volunteer:
   a. Build overlapping session pairs (any GLH session whose time range
      intersects a GPX session's time range).
   b. Within each pair, for each GLH timestamp t:
        - find bracketing GPX timestamps (t_lo, t_hi)
        - check the bracket gap is below `max_bracket_gap_s` (default 60 s);
          if not, the GLH point is unmatched (recorded but with NaN truth).
        - linearly interpolate east/north (and lat/lon) between bracketing
          points at t.
   c. Compute Euclidean deviation in BNG metres.
3. Concatenate all matched rows into a single output DataFrame.

Output columns
--------------
    volunteer, glh_layer, glh_session_id, gpx_session_id,
    timestamp,
    glh_lat, glh_lon, glh_east, glh_north,
    glh_accuracy_m, glh_source, glh_speed_mps,         (where available)
    gpx_lat, gpx_lon, gpx_east, gpx_north,
    bracket_dt_prev_s, bracket_dt_next_s, bracket_total_s,
    deviation_m, matched

`matched` is False when the GLH point either falls outside any GPX session
or only had a too-large bracket gap.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .projection import bng_distance
from .sessionize import summarise_sessions


DEFAULT_MAX_BRACKET_GAP_S = 60.0  # if the two bracketing GPX points are >60s
                                  # apart, treat the interpolation as unreliable


# ─────────────────────────────────────────────────────────────────────────────
# Session-overlap detection
# ─────────────────────────────────────────────────────────────────────────────

def find_overlapping_sessions(
    gpx_sessions: pd.DataFrame,
    glh_sessions: pd.DataFrame,
    *,
    gpx_session_col: str = "session_id",
    glh_session_col: str = "session_id",
) -> pd.DataFrame:
    """
    Return all (gpx_session_id, glh_session_id) pairs whose time ranges
    overlap. Result columns:

        gpx_session_id, glh_session_id,
        gpx_start, gpx_end, glh_start, glh_end,
        overlap_start, overlap_end, overlap_seconds
    """
    if gpx_sessions.empty or glh_sessions.empty:
        return pd.DataFrame(columns=[
            "gpx_session_id", "glh_session_id",
            "gpx_start", "gpx_end", "glh_start", "glh_end",
            "overlap_start", "overlap_end", "overlap_seconds",
        ])

    # Cross-join (small N) and filter on overlap.
    gpx = gpx_sessions[[gpx_session_col, "start_time", "end_time"]].rename(
        columns={gpx_session_col: "gpx_session_id",
                 "start_time": "gpx_start",
                 "end_time":   "gpx_end"})
    glh = glh_sessions[[glh_session_col, "start_time", "end_time"]].rename(
        columns={glh_session_col: "glh_session_id",
                 "start_time": "glh_start",
                 "end_time":   "glh_end"})

    pairs = gpx.merge(glh, how="cross")
    pairs["overlap_start"] = pairs[["gpx_start", "glh_start"]].max(axis=1)
    pairs["overlap_end"] = pairs[["gpx_end", "glh_end"]].min(axis=1)
    pairs["overlap_seconds"] = (
        (pairs["overlap_end"] - pairs["overlap_start"]).dt.total_seconds()
    )
    pairs = pairs[pairs["overlap_seconds"] > 0].reset_index(drop=True)
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Per-session interpolation
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_gpx_at_times(
    gpx_session: pd.DataFrame,
    target_times: pd.Series,
    *,
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Linearly interpolate a GPX session's position at each target timestamp.

    Parameters
    ----------
    gpx_session : DataFrame
        A single GPX session, sorted by `timestamp`, with columns
        timestamp, lat, lon, east, north.
    target_times : Series of datetime64[ns, UTC]
        Timestamps at which to interpolate.

    Returns
    -------
    DataFrame indexed like target_times with columns:
        gpx_lat, gpx_lon, gpx_east, gpx_north,
        bracket_dt_prev_s, bracket_dt_next_s, bracket_total_s,
        in_bracket   (bool — target lies between two real GPX points)
    """
    if gpx_session.empty or target_times.empty:
        return pd.DataFrame(
            index=target_times.index,
            columns=["gpx_lat", "gpx_lon", "gpx_east", "gpx_north",
                     "bracket_dt_prev_s", "bracket_dt_next_s", "bracket_total_s",
                     "in_bracket"],
            dtype=object,
        )

    gpx = gpx_session.sort_values(time_col).reset_index(drop=True)
    gpx_ts = gpx[time_col].values.astype("datetime64[ns]")
    target = pd.to_datetime(target_times, utc=True).values.astype("datetime64[ns]")

    # For each target, the index of the next GPX point (insertion right side).
    idx_right = np.searchsorted(gpx_ts, target, side="right")
    idx_left = idx_right - 1

    n_gpx = len(gpx_ts)
    in_bracket = (idx_left >= 0) & (idx_right < n_gpx)

    # Clamp indices into valid range to allow vectorised array indexing;
    # out-of-bracket rows get masked to NaN below.
    safe_left = np.clip(idx_left, 0, n_gpx - 1)
    safe_right = np.clip(idx_right, 0, n_gpx - 1)

    t_left = gpx_ts[safe_left]
    t_right = gpx_ts[safe_right]
    east_left = gpx["east"].values[safe_left]
    east_right = gpx["east"].values[safe_right]
    north_left = gpx["north"].values[safe_left]
    north_right = gpx["north"].values[safe_right]
    lat_left = gpx["lat"].values[safe_left]
    lat_right = gpx["lat"].values[safe_right]
    lon_left = gpx["lon"].values[safe_left]
    lon_right = gpx["lon"].values[safe_right]

    dt_prev = (target - t_left).astype("timedelta64[ns]").astype("float64") / 1e9
    dt_next = (t_right - target).astype("timedelta64[ns]").astype("float64") / 1e9
    bracket_total = dt_prev + dt_next

    # weight of the right-hand point, 0 → all-left, 1 → all-right
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(bracket_total > 0, dt_prev / bracket_total, 0.0)

    east_interp = east_left + w * (east_right - east_left)
    north_interp = north_left + w * (north_right - north_left)
    lat_interp = lat_left + w * (lat_right - lat_left)
    lon_interp = lon_left + w * (lon_right - lon_left)

    # mask rows where target falls outside the GPX session bracket
    east_interp = np.where(in_bracket, east_interp, np.nan)
    north_interp = np.where(in_bracket, north_interp, np.nan)
    lat_interp = np.where(in_bracket, lat_interp, np.nan)
    lon_interp = np.where(in_bracket, lon_interp, np.nan)
    dt_prev = np.where(in_bracket, dt_prev, np.nan)
    dt_next = np.where(in_bracket, dt_next, np.nan)
    bracket_total = np.where(in_bracket, bracket_total, np.nan)

    return pd.DataFrame({
        "gpx_lat": lat_interp,
        "gpx_lon": lon_interp,
        "gpx_east": east_interp,
        "gpx_north": north_interp,
        "bracket_dt_prev_s": dt_prev,
        "bracket_dt_next_s": dt_next,
        "bracket_total_s": bracket_total,
        "in_bracket": in_bracket,
    }, index=target_times.index)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level matching
# ─────────────────────────────────────────────────────────────────────────────

def match_streams(
    gpx_df: pd.DataFrame,
    glh_df: pd.DataFrame,
    *,
    glh_layer: str,
    max_bracket_gap_s: float = DEFAULT_MAX_BRACKET_GAP_S,
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Match one GLH layer (e.g. 'raw_signals' or 'timeline_paths') to GPX truth.

    Both inputs must already have:
        - `session_id`
        - WGS84 `lat`,`lon`
        - BNG `east`,`north`
        - tz-aware UTC `timestamp`

    Returns
    -------
    DataFrame
        One row per GLH point. Columns described in module docstring.
        Rows where the GLH point fell outside any GPX session (no matching
        session) are omitted entirely. Rows where a session matched but the
        bracket gap exceeded `max_bracket_gap_s` are kept with `matched=False`
        and NaN gpx_* columns.
    """
    out_cols_template = [
        "volunteer", "glh_layer", "glh_session_id", "gpx_session_id",
        "timestamp",
        "glh_lat", "glh_lon", "glh_east", "glh_north",
        "glh_accuracy_m", "glh_source", "glh_speed_mps",
        "gpx_lat", "gpx_lon", "gpx_east", "gpx_north",
        "bracket_dt_prev_s", "bracket_dt_next_s", "bracket_total_s",
        "deviation_m", "matched",
    ]
    if gpx_df.empty or glh_df.empty:
        return pd.DataFrame(columns=out_cols_template)

    # Per-session summaries so we can find temporal overlaps.
    gpx_sessions = summarise_sessions(gpx_df, time_col=time_col)
    glh_sessions = summarise_sessions(glh_df, time_col=time_col)

    pairs = find_overlapping_sessions(gpx_sessions, glh_sessions)
    if pairs.empty:
        return pd.DataFrame(columns=out_cols_template)

    # Index GPX and GLH by session id for fast lookup
    gpx_by_session = {
        sid: g.sort_values(time_col).reset_index(drop=True)
        for sid, g in gpx_df.groupby("session_id")
    }
    glh_by_session = {
        sid: g.sort_values(time_col).reset_index(drop=True)
        for sid, g in glh_df.groupby("session_id")
    }

    chunks: list[pd.DataFrame] = []
    for _, pair in pairs.iterrows():
        gpx_sid = pair["gpx_session_id"]
        glh_sid = pair["glh_session_id"]
        overlap_start = pair["overlap_start"]
        overlap_end = pair["overlap_end"]

        gpx_s = gpx_by_session.get(gpx_sid)
        glh_s = glh_by_session.get(glh_sid)
        if gpx_s is None or glh_s is None or gpx_s.empty or glh_s.empty:
            continue

        # Restrict GLH points to those inside the overlap (avoids extrapolation).
        glh_in = glh_s[(glh_s[time_col] >= overlap_start) &
                       (glh_s[time_col] <= overlap_end)].copy()
        if glh_in.empty:
            continue

        interp = interpolate_gpx_at_times(gpx_s, glh_in[time_col])

        joined = glh_in.copy()
        joined[interp.columns] = interp.values

        # Build the output schema with all expected columns (use NaN where missing).
        out = pd.DataFrame(index=joined.index)
        if "volunteer" in joined.columns:
            out["volunteer"] = joined["volunteer"]
        else:
            out["volunteer"] = pd.NA
        out["glh_layer"] = glh_layer
        out["glh_session_id"] = glh_sid
        out["gpx_session_id"] = gpx_sid
        out["timestamp"] = joined[time_col]
        out["glh_lat"] = joined["lat"]
        out["glh_lon"] = joined["lon"]
        out["glh_east"] = joined["east"]
        out["glh_north"] = joined["north"]
        out["glh_accuracy_m"] = joined.get("accuracy_m", pd.Series(np.nan, index=joined.index))
        out["glh_source"] = joined.get("source", pd.Series(pd.NA, index=joined.index))
        out["glh_speed_mps"] = joined.get("speed_mps", pd.Series(np.nan, index=joined.index))
        out["gpx_lat"] = joined["gpx_lat"]
        out["gpx_lon"] = joined["gpx_lon"]
        out["gpx_east"] = joined["gpx_east"]
        out["gpx_north"] = joined["gpx_north"]
        out["bracket_dt_prev_s"] = joined["bracket_dt_prev_s"]
        out["bracket_dt_next_s"] = joined["bracket_dt_next_s"]
        out["bracket_total_s"] = joined["bracket_total_s"]

        # deviation: NaN where bracket failed; metres where it succeeded.
        out["deviation_m"] = bng_distance(
            out["glh_east"], out["glh_north"], out["gpx_east"], out["gpx_north"]
        )

        # `matched` is True only when in-bracket AND bracket gap acceptable.
        in_bracket = joined["in_bracket"].astype(bool)
        gap_ok = joined["bracket_total_s"] <= max_bracket_gap_s
        out["matched"] = (in_bracket & gap_ok).fillna(False)

        # mask gpx_* and deviation where not matched
        not_matched = ~out["matched"]
        for col in ("gpx_lat", "gpx_lon", "gpx_east", "gpx_north", "deviation_m"):
            out.loc[not_matched, col] = np.nan

        chunks.append(out)

    if not chunks:
        return pd.DataFrame(columns=out_cols_template)

    matched_df = pd.concat(chunks, ignore_index=True)
    # Some GLH points may be in two overlapping pairs at session boundaries —
    # keep the earliest pairing (smallest gpx_session_id) for determinism.
    matched_df = (matched_df
                  .sort_values(["timestamp", "gpx_session_id"])
                  .drop_duplicates(subset=["timestamp", "glh_session_id"], keep="first")
                  .reset_index(drop=True))
    # Reorder columns to the documented template
    matched_df = matched_df[[c for c in out_cols_template if c in matched_df.columns]]
    return matched_df


def match_volunteer(
    gpx_df: pd.DataFrame,
    glh_layers: dict[str, pd.DataFrame],
    *,
    max_bracket_gap_s: float = DEFAULT_MAX_BRACKET_GAP_S,
) -> pd.DataFrame:
    """
    Run `match_streams` against each GLH layer present and concatenate.

    Parameters
    ----------
    gpx_df : DataFrame
        Cleaned, projected, sessionised GPX for one volunteer.
    glh_layers : dict[str, DataFrame]
        Mapping from layer name (e.g. 'raw_signals', 'timeline_paths') to a
        cleaned, projected, sessionised GLH DataFrame. Layers that are empty
        for a given volunteer should still be passed as empty DataFrames; they
        are skipped silently.
    """
    out_chunks: list[pd.DataFrame] = []
    for layer, df in glh_layers.items():
        if df is None or df.empty:
            continue
        matched = match_streams(
            gpx_df, df,
            glh_layer=layer,
            max_bracket_gap_s=max_bracket_gap_s,
        )
        if not matched.empty:
            out_chunks.append(matched)

    if not out_chunks:
        return pd.DataFrame()
    return pd.concat(out_chunks, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Summary helpers — used by audit reports
# ─────────────────────────────────────────────────────────────────────────────

def deviation_summary(matched_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-(volunteer, layer) deviation statistics.

    Columns: volunteer, glh_layer, n_total, n_matched, match_rate,
             mean_m, median_m, p90_m, p95_m, p99_m, max_m
    """
    if matched_df.empty:
        return pd.DataFrame(columns=[
            "volunteer", "glh_layer", "n_total", "n_matched", "match_rate",
            "mean_m", "median_m", "p90_m", "p95_m", "p99_m", "max_m",
        ])

    def _agg(g):
        n_total = len(g)
        n_matched = int(g["matched"].sum())
        dev = g.loc[g["matched"], "deviation_m"].dropna()
        return pd.Series({
            "n_total":   n_total,
            "n_matched": n_matched,
            "match_rate": n_matched / n_total if n_total else np.nan,
            "mean_m":   dev.mean()   if len(dev) else np.nan,
            "median_m": dev.median() if len(dev) else np.nan,
            "p90_m":    dev.quantile(0.90) if len(dev) else np.nan,
            "p95_m":    dev.quantile(0.95) if len(dev) else np.nan,
            "p99_m":    dev.quantile(0.99) if len(dev) else np.nan,
            "max_m":    dev.max()    if len(dev) else np.nan,
        })

    return (matched_df.groupby(["volunteer", "glh_layer"], dropna=False)
            .apply(_agg).reset_index())
