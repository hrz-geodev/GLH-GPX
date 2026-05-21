"""
sessionize.py
=============
Session-id assignment for GLH point streams.

Sessions are the unit of matching: a GLH session is paired with the GPX
session(s) it temporally overlaps. Each volunteer's streams must therefore
carry consistent `session_id` values.

GPX
---
GPX session ids are assigned by `gpx_parser.parse_gpx_file()` at parse time
(one session per `<trkseg>` in the default project configuration). No further
sessionising is needed.

GLH — three layers, three policies
----------------------------------
- **timeline_paths**:  each timeline-path segment in Google's
  `semanticSegments` already represents one continuous on-device timeline
  path. The parser exposes this as `segment_id`. We rename it to
  `session_id` here.

- **visits / activities**: these are inherently single-segment entities
  (one segment = one visit or one activity). We expose `session_id` =
  `segment_id` for consistency, but most analyses treat them by row
  rather than by session.

- **raw_signals**: a flat time-ordered stream of position fixes with no
  built-in segmentation. We assign sessions on time gaps (default 600 s =
  10 min). Sample rates vary with GLH, but 10 minutes is a natural
  threshold between recording bursts and stationary periods.

All functions return a NEW DataFrame and never mutate the input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_RAWSIGNALS_GAP_S = 600  # 10 minutes


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_sorted(df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
    """Return a copy sorted by timestamp ascending, index reset."""
    return df.sort_values(time_col).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Timeline paths / visits / activities — use the existing segment_id
# ─────────────────────────────────────────────────────────────────────────────

def sessionise_timeline_paths(df: pd.DataFrame) -> pd.DataFrame:
    """
    Promote `segment_id` to `session_id` and add `point_id` within session.

    No-op semantics if the DataFrame already has both `session_id` and
    `point_id`. Assumes one row per timeline path point.
    """
    if df.empty:
        out = df.copy()
        out["session_id"] = pd.Series(dtype=int)
        out["point_id"] = pd.Series(dtype=int)
        return out

    if "segment_id" not in df.columns:
        raise KeyError("timeline_paths DataFrame must have a `segment_id` column.")

    out = _ensure_sorted(df)
    out["session_id"] = out["segment_id"].astype(int)
    out["point_id"] = out.groupby("session_id").cumcount()
    return out


def sessionise_visits_or_activities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirror `segment_id` into `session_id` for visit / activity rows.

    Visits/activities are inherently 1-2 rows per segment so `point_id` is
    minimal but included for uniformity.
    """
    if df.empty:
        out = df.copy()
        out["session_id"] = pd.Series(dtype=int)
        out["point_id"] = pd.Series(dtype=int)
        return out

    if "segment_id" not in df.columns:
        raise KeyError("Input DataFrame must have a `segment_id` column.")

    # Visits use start_time, activities use timestamp; pick whichever exists.
    time_col = "timestamp" if "timestamp" in df.columns else "start_time"
    out = df.sort_values([time_col]).reset_index(drop=True)
    out["session_id"] = out["segment_id"].astype(int)
    out["point_id"] = out.groupby("session_id").cumcount()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Raw signals — time-gap based
# ─────────────────────────────────────────────────────────────────────────────

def sessionise_raw_signals(
    df: pd.DataFrame,
    *,
    gap_seconds: int = DEFAULT_RAWSIGNALS_GAP_S,
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Assign sessions to a rawSignals DataFrame using a time-gap rule.

    A new session starts whenever the gap to the previous point exceeds
    `gap_seconds`. Adds `session_id` and `point_id` columns.
    """
    if df.empty:
        out = df.copy()
        out["session_id"] = pd.Series(dtype=int)
        out["point_id"] = pd.Series(dtype=int)
        return out

    out = _ensure_sorted(df, time_col=time_col)
    gap_s = out[time_col].diff().dt.total_seconds()
    is_new_session = gap_s.isna() | (gap_s > gap_seconds)
    out["session_id"] = is_new_session.cumsum().astype(int) - 1
    out["point_id"] = out.groupby("session_id").cumcount()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Session summary — common report shape used by audit + matching
# ─────────────────────────────────────────────────────────────────────────────

def summarise_sessions(
    df: pd.DataFrame,
    *,
    session_col: str = "session_id",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """
    One row per session with start/end time, n_points, bbox in WGS84/BNG.

    Output columns:
        volunteer (if present), session_id, n_points,
        start_time, end_time, duration_s,
        lat_min, lat_max, lon_min, lon_max,
        east_min, east_max, north_min, north_max
    """
    cols = []
    has_v = "volunteer" in df.columns
    if has_v: cols.append("volunteer")
    cols += [
        session_col, "n_points", "start_time", "end_time", "duration_s",
        "lat_min", "lat_max", "lon_min", "lon_max",
    ]
    if {"east", "north"}.issubset(df.columns):
        cols += ["east_min", "east_max", "north_min", "north_max"]

    if df.empty or session_col not in df.columns:
        return pd.DataFrame(columns=cols)

    g = df.groupby(session_col, sort=True)
    out_dict = {
        "n_points":    g.size(),
        "start_time":  g[time_col].min(),
        "end_time":    g[time_col].max(),
        "lat_min":     g["lat"].min(),
        "lat_max":     g["lat"].max(),
        "lon_min":     g["lon"].min(),
        "lon_max":     g["lon"].max(),
    }
    if has_v:
        out_dict["volunteer"] = g["volunteer"].first()
    if {"east", "north"}.issubset(df.columns):
        out_dict["east_min"] = g["east"].min()
        out_dict["east_max"] = g["east"].max()
        out_dict["north_min"] = g["north"].min()
        out_dict["north_max"] = g["north"].max()

    out = pd.DataFrame(out_dict).reset_index()
    out["duration_s"] = (out["end_time"] - out["start_time"]).dt.total_seconds()
    return out[[c for c in cols if c in out.columns]]
