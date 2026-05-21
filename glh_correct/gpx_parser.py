"""
gpx_parser.py
=============
Parses GPX files into a standardised pandas DataFrame, with **session detection**.

A "session" is a contiguous run of recording with no large time gaps. The
parser produces session ids in three ways:

    1. trkseg boundaries  —  each <trkseg> in the GPX is a session boundary
                             (this is the project default).
    2. internal time gaps —  optionally, inside a single trkseg, a gap
                             larger than `max_gap_seconds` splits a new
                             session. Disabled by default (None).
    3. file boundary       —  when parsing a folder, files are concatenated
                             and a fresh session id is also assigned at file
                             changes.

The rationale: real-world GPX dumps vary widely — some subjects produce
a single file with many trksegs spanning weeks of activity, others
produce many small multi-segment files. The session abstraction
normalises both. trkseg-only is preferred because GPS recording apps
already split on user-driven session boundaries; treating internal
pauses as new sessions can over-fragment.

Output columns
--------------
    volunteer       : str   (when provided via parse_volunteer_folder)
    source_file     : str
    session_id      : int   (0-based, unique within the returned DataFrame)
    point_id        : int   (0-based within the session)
    timestamp       : datetime  (tz-aware, UTC)
    lat             : float
    lon             : float
    elevation_m     : float (NaN if absent)
    trk_idx         : int   (index of the parent <trk>)
    seg_idx         : int   (index of the parent <trkseg> within the file)

Implementation notes
--------------------
Uses gpxpy for robust parsing across GPX 1.0 / 1.1 / namespaces. Falls back
to lxml if gpxpy is unavailable (slim, used only for the read step).
"""

from __future__ import annotations

import os
from typing import Iterable

import pandas as pd

try:
    import gpxpy
    _HAS_GPXPY = True
except ImportError:  # pragma: no cover - fallback path
    _HAS_GPXPY = False
    from lxml import etree


# ─────────────────────────────────────────────────────────────────────────────
# Core parser
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_MAX_GAP_S: int | None = None  # disabled — trkseg boundaries only


def _iter_points_gpxpy(filepath: str) -> Iterable[dict]:
    """Yield point dicts from a GPX file using gpxpy."""
    with open(filepath, "r", encoding="utf-8") as fh:
        gpx = gpxpy.parse(fh)
    for trk_idx, track in enumerate(gpx.tracks):
        for seg_idx, segment in enumerate(track.segments):
            for point in segment.points:
                yield {
                    "trk_idx": trk_idx,
                    "seg_idx": seg_idx,
                    "timestamp": point.time,
                    "lat": point.latitude,
                    "lon": point.longitude,
                    "elevation_m": point.elevation,
                }


def _iter_points_lxml(filepath: str) -> Iterable[dict]:
    """Fallback iterator using lxml only (no gpxpy)."""
    tree = etree.parse(filepath)
    root = tree.getroot()
    ns_uri = root.nsmap.get(None) or ""
    pre = f"{{{ns_uri}}}" if ns_uri else ""
    for trk_idx, trk in enumerate(root.findall(f"{pre}trk")):
        for seg_idx, seg in enumerate(trk.findall(f"{pre}trkseg")):
            for pt in seg.findall(f"{pre}trkpt"):
                lat = float(pt.get("lat"))
                lon = float(pt.get("lon"))
                t_el = pt.find(f"{pre}time")
                ts = t_el.text if t_el is not None else None
                e_el = pt.find(f"{pre}ele")
                ele = float(e_el.text) if e_el is not None and e_el.text else None
                yield {
                    "trk_idx": trk_idx,
                    "seg_idx": seg_idx,
                    "timestamp": ts,
                    "lat": lat,
                    "lon": lon,
                    "elevation_m": ele,
                }


def parse_gpx_file(
    filepath: str,
    *,
    max_gap_seconds: int | None = _DEFAULT_MAX_GAP_S,
    volunteer: str | None = None,
    base_session_id: int = 0,
) -> pd.DataFrame:
    """
    Parse a single GPX file into a session-aware DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the .gpx file.
    max_gap_seconds : int or None, default None
        Within a single trkseg, a gap larger than this starts a new session.
        None disables time-gap splitting — each trkseg is one session.
    volunteer : str, optional
        Volunteer id added as a column if supplied.
    base_session_id : int, default 0
        Used by parse_gpx_folder to keep session ids unique across files.

    Returns
    -------
    pd.DataFrame
        Columns: volunteer (if given), source_file, session_id, point_id,
        timestamp, lat, lon, elevation_m, trk_idx, seg_idx.
    """
    iter_fn = _iter_points_gpxpy if _HAS_GPXPY else _iter_points_lxml
    records = list(iter_fn(filepath))

    if not records:
        return _empty_frame(volunteer)

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "lat", "lon"]).reset_index(drop=True)
    df = df.sort_values(["trk_idx", "seg_idx", "timestamp"]).reset_index(drop=True)

    # ── Assign session_id ────────────────────────────────────────────────────
    # Start a new session whenever:
    #   - (trk_idx, seg_idx) changes (trkseg boundary), or
    #   - the inter-point gap exceeds max_gap_seconds within a single trkseg.
    new_session = pd.Series(False, index=df.index)
    new_session.iloc[0] = True
    # boundary on (trk, seg) change
    seg_change = (
        (df["trk_idx"] != df["trk_idx"].shift())
        | (df["seg_idx"] != df["seg_idx"].shift())
    )
    new_session = new_session | seg_change
    # boundary on time gap (only within same trkseg), if enabled
    if max_gap_seconds is not None:
        same_seg = ~seg_change
        gap_s = df["timestamp"].diff().dt.total_seconds()
        big_gap = (gap_s > max_gap_seconds) & same_seg
        new_session = new_session | big_gap

    df["session_id"] = base_session_id + new_session.cumsum() - 1
    df["point_id"] = df.groupby("session_id").cumcount()
    df["source_file"] = os.path.basename(filepath)
    if volunteer is not None:
        df.insert(0, "volunteer", volunteer)

    cols = ["volunteer"] if volunteer is not None else []
    cols += ["source_file", "session_id", "point_id", "timestamp",
             "lat", "lon", "elevation_m", "trk_idx", "seg_idx"]
    return df[cols]


def parse_volunteer_folder(
    folder_path: str,
    *,
    max_gap_seconds: int | None = _DEFAULT_MAX_GAP_S,
    volunteer: str | None = None,
) -> pd.DataFrame:
    """
    Parse all GPX files in a volunteer's folder into one session-aware DataFrame.

    Session ids are unique across files: e.g. file A yields sessions 0..n,
    file B then starts at session id n+1, etc.
    """
    if volunteer is None:
        volunteer = os.path.basename(os.path.normpath(folder_path))

    gpx_files = sorted(
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".gpx")
    )
    if not gpx_files:
        return _empty_frame(volunteer)

    frames: list[pd.DataFrame] = []
    next_session_id = 0
    for fp in gpx_files:
        df = parse_gpx_file(
            fp,
            max_gap_seconds=max_gap_seconds,
            volunteer=volunteer,
            base_session_id=next_session_id,
        )
        if df.empty:
            continue
        next_session_id = int(df["session_id"].max()) + 1
        frames.append(df)

    if not frames:
        return _empty_frame(volunteer)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["session_id", "point_id"]).reset_index(drop=True)
    return combined


def session_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per session: start/end time, n_points, span, mean rate."""
    if df.empty:
        return pd.DataFrame(columns=[
            "volunteer", "session_id", "source_file", "n_points",
            "start_time", "end_time", "duration_s", "mean_rate_hz",
            "lat_min", "lat_max", "lon_min", "lon_max",
        ])
    g = df.groupby("session_id", sort=True)
    out = pd.DataFrame({
        "volunteer":   g["volunteer"].first() if "volunteer" in df.columns else None,
        "source_file": g["source_file"].first(),
        "n_points":    g.size(),
        "start_time":  g["timestamp"].min(),
        "end_time":    g["timestamp"].max(),
        "lat_min":     g["lat"].min(),
        "lat_max":     g["lat"].max(),
        "lon_min":     g["lon"].min(),
        "lon_max":     g["lon"].max(),
    }).reset_index()
    out["duration_s"] = (out["end_time"] - out["start_time"]).dt.total_seconds()
    out["mean_rate_hz"] = (out["n_points"] - 1) / out["duration_s"].replace(0, pd.NA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _empty_frame(volunteer: str | None) -> pd.DataFrame:
    cols = ["volunteer"] if volunteer is not None else []
    cols += ["source_file", "session_id", "point_id", "timestamp",
             "lat", "lon", "elevation_m", "trk_idx", "seg_idx"]
    return pd.DataFrame(columns=cols)


# ─────────────────────────────────────────────────────────────────────────────
# CLI quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python gpx_parser.py path/to/file_or_folder")
        raise SystemExit(1)
    target = sys.argv[1]
    if os.path.isdir(target):
        df = parse_volunteer_folder(target)
    else:
        df = parse_gpx_file(target)
    print(f"rows: {len(df)}, sessions: {df['session_id'].nunique() if len(df) else 0}")
    if len(df):
        print(df.head())
        print("\nSession summary:")
        print(session_summary(df).head(10))
