"""
glh_parser.py
=============
Parses Google Location History (GLH) exports into standardised pandas DataFrames.

Supports BOTH export families observed in this project:

Family A — dict-shape
    Top-level JSON is a dict with keys:
        - semanticSegments      : list of visit / timelinePath / activity / (other)
        - rawSignals            : list of position / activityRecord / wifiScan events
        - userLocationProfile   : metadata (ignored)
    Coordinates stored as: "55.943°, -3.207°"  (degree-symbol delimited)

Family B — list-shape
    Top-level JSON is a flat list of segments, each containing
        startTime, endTime, and exactly one of visit / activity / timelinePath.
    No rawSignals are present.
    Coordinates stored as: "geo:55.943,-3.207"  (geo-URI prefix, no symbol)

The parser is **format-agnostic** at the call site:
    >>> result = parse_glh_file("data/my_track/Timeline.json")
    >>> result["family"]            # "A" or "B"
    >>> result["raw_signals"]       # DataFrame (empty for Family B)
    >>> result["activity_records"]  # DataFrame (empty for Family B)
    >>> result["wifi_scans"]        # DataFrame (counts only, empty for Family B)
    >>> result["timeline_paths"]    # DataFrame, ONE ROW PER PATH POINT, with segment_id
    >>> result["visits"]            # DataFrame, ONE ROW PER VISIT
    >>> result["activities"]        # DataFrame, TWO ROWS PER ACTIVITY (start + end)
    >>> result["other_segments"]    # list of dicts — segments that didn't match known kinds

Trajectory preservation
-----------------------
Timeline paths are emitted with a `segment_id` column so that the original
path grouping (one segment = one continuous on-device timeline path) is
retained. Stage 3 trajectory models depend on this grouping.

Authoring notes
---------------
- All timestamps are returned as timezone-aware pandas Timestamp in UTC.
- Coordinates are returned as floats (decimal degrees, WGS84).
- All "row" DataFrames carry a `source_file` and `volunteer` column when
  loaded via `parse_volunteer_folder()`.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate parsing
# ─────────────────────────────────────────────────────────────────────────────

# Family A: "55.9430589°, -3.2071267°"  → strip ° then split
# Family B: "geo:55.943346,-3.184199"    → strip leading "geo:" then split
# Fallback: "55.9430589, -3.2071267"     → plain CSV
_RX_COORD = re.compile(
    r"""
    ^\s*
    (?:geo:)?                                  # optional 'geo:' prefix
    (?P<lat>[-+]?\d+(?:\.\d+)?)\s*[°°]?   # latitude, optional degree symbol
    \s*,\s*
    (?P<lon>[-+]?\d+(?:\.\d+)?)\s*[°°]?   # longitude, optional degree symbol
    \s*$
    """,
    re.VERBOSE,
)


def parse_latlng(value: Any) -> tuple[float | None, float | None]:
    """
    Parse a GLH coordinate value into (lat, lon) decimal degrees.

    Accepts any of:
        "55.9430589°, -3.2071267°"
        "geo:55.943346,-3.184199"
        "55.943, -3.207"
        {"latitudeE7": 559430589, "longitudeE7": -32071267}   # legacy E7 ints
        None / missing / unparseable  →  (None, None)
    """
    # legacy E7 dict (defensive — not seen in this dataset but cheap to support)
    if isinstance(value, dict):
        lat_e7 = value.get("latitudeE7")
        lon_e7 = value.get("longitudeE7")
        if isinstance(lat_e7, (int, float)) and isinstance(lon_e7, (int, float)):
            return lat_e7 / 1e7, lon_e7 / 1e7
        # some Family A nested dicts: {"latLng": "55.9°, -3.2°"}
        for key in ("LatLng", "latLng", "latlng"):
            inner = value.get(key)
            if inner:
                return parse_latlng(inner)
        return None, None

    if not isinstance(value, str) or not value.strip():
        return None, None

    m = _RX_COORD.match(value)
    if not m:
        return None, None
    try:
        return float(m.group("lat")), float(m.group("lon"))
    except ValueError:
        return None, None


def _to_utc(series_or_value):
    """Coerce timestamp(s) to tz-aware UTC pandas Timestamp(s)."""
    return pd.to_datetime(series_or_value, utc=True, errors="coerce")


# ─────────────────────────────────────────────────────────────────────────────
# Format detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_family(data: Any) -> str:
    """Return 'A' for dict-shape (with rawSignals), 'B' for list-shape."""
    if isinstance(data, dict) and ("rawSignals" in data or "semanticSegments" in data):
        return "A"
    if isinstance(data, list):
        return "B"
    raise ValueError(
        "Unrecognised GLH JSON shape: top-level is "
        f"{type(data).__name__}, expected dict (Family A) or list (Family B)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Family A — rawSignals
# ─────────────────────────────────────────────────────────────────────────────

def _parse_family_a_raw_signals(data: dict) -> dict[str, pd.DataFrame]:
    """
    Split rawSignals into three DataFrames: positions, activity_records, wifi_scans.

    Positions are the primary location pings (one row per ping).
    """
    pos_records, act_records, wifi_records = [], [], []

    for entry in data.get("rawSignals", []):
        # position fix
        if "position" in entry:
            p = entry["position"]
            lat, lon = parse_latlng(p.get("LatLng") or p.get("latLng"))
            if lat is None:
                continue
            pos_records.append({
                "timestamp": p.get("timestamp"),
                "lat": lat,
                "lon": lon,
                "accuracy_m": p.get("accuracyMeters"),
                "altitude_m": p.get("altitudeMeters"),
                "speed_mps": p.get("speedMetersPerSecond"),
                "source": p.get("source", "UNKNOWN"),
            })
        # activity detection event (no position)
        elif "activityRecord" in entry:
            ar = entry["activityRecord"]
            probable_activities = ar.get("probableActivities", []) or []
            top = max(probable_activities, key=lambda x: x.get("confidence", 0), default={})
            act_records.append({
                "timestamp": ar.get("timestamp"),
                "top_activity": top.get("type"),
                "confidence": top.get("confidence"),
                "n_candidates": len(probable_activities),
            })
        # wifi scan (no position by itself)
        elif "wifiScan" in entry:
            ws = entry["wifiScan"]
            scans = ws.get("deliveredAccessPoints", []) or ws.get("accessPoints", []) or []
            wifi_records.append({
                "timestamp": ws.get("timestamp"),
                "n_access_points": len(scans),
            })

    def _finalise(records, default_cols):
        if not records:
            return pd.DataFrame(columns=default_cols)
        df = pd.DataFrame(records)
        if "timestamp" in df.columns:
            df["timestamp"] = _to_utc(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    return {
        "raw_signals": _finalise(
            pos_records,
            ["timestamp", "lat", "lon", "accuracy_m", "altitude_m", "speed_mps", "source"],
        ),
        "activity_records": _finalise(
            act_records, ["timestamp", "top_activity", "confidence", "n_candidates"]
        ),
        "wifi_scans": _finalise(wifi_records, ["timestamp", "n_access_points"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# semanticSegments / list-shape segments — unified handling
# ─────────────────────────────────────────────────────────────────────────────

def _extract_segments(data: Any) -> list[dict]:
    """Return the list of segment dicts regardless of family."""
    if isinstance(data, dict):
        return data.get("semanticSegments", []) or []
    if isinstance(data, list):
        return data
    return []


def _parse_segments(segments: list[dict]) -> dict[str, Any]:
    """
    Parse a list of GLH segments into timeline_paths, visits, activities, other.

    `segment_id` is the 0-based index of the segment in the input list and is
    preserved on every emitted row so trajectory grouping can be reconstructed.
    """
    tl_records, visit_records, activity_records = [], [], []
    other_segments = []

    # Reusable converter for Family B's `durationMinutesOffsetFromStartTime`
    # field: builds an absolute timestamp from the segment startTime plus
    # the minute offset (which Google stores as a string).
    def _abs_timestamp_from_offset(start_iso: str | None, offset_str):
        if start_iso is None or offset_str is None:
            return None
        try:
            offset_min = float(offset_str)
        except (TypeError, ValueError):
            return None
        return pd.to_datetime(start_iso, utc=True, errors="coerce") + \
               pd.to_timedelta(offset_min, unit="m")

    for seg_id, seg in enumerate(segments):
        start_time = seg.get("startTime")
        end_time = seg.get("endTime")

        if "timelinePath" in seg:
            for pt in seg["timelinePath"]:
                lat, lon = parse_latlng(pt.get("point") or pt.get("latLng"))
                if lat is None:
                    continue
                # Resolve the per-point timestamp from whichever field is
                # present:
                #   Family A: pt["time"] is an absolute ISO string
                #   Family B: pt["durationMinutesOffsetFromStartTime"] is
                #            a string minute-offset relative to segment startTime
                # We compute the absolute timestamp in either case so that
                # downstream dedup, sessionise and matching all work.
                if "time" in pt and pt["time"]:
                    ts = pt["time"]
                elif "durationMinutesOffsetFromStartTime" in pt:
                    ts = _abs_timestamp_from_offset(
                        start_time, pt["durationMinutesOffsetFromStartTime"]
                    )
                else:
                    ts = start_time

                tl_records.append({
                    "segment_id": seg_id,
                    "timestamp": ts,
                    "lat": lat,
                    "lon": lon,
                    "segment_start": start_time,
                    "segment_end": end_time,
                })

        elif "visit" in seg:
            visit = seg["visit"]
            top = visit.get("topCandidate", {}) or {}
            place_loc = top.get("placeLocation")
            lat, lon = parse_latlng(place_loc)
            if lat is None:
                # Family A nests differently: placeLocation may itself be a dict
                if isinstance(place_loc, dict):
                    lat, lon = parse_latlng(place_loc.get("latLng"))
            if lat is None:
                # silently skip a visit with no usable coordinate
                continue
            visit_records.append({
                "segment_id": seg_id,
                "start_time": start_time,
                "end_time": end_time,
                "lat": lat,
                "lon": lon,
                "place_id": top.get("placeID") or top.get("placeId"),
                "semantic_type": top.get("semanticType"),
                "probability": top.get("probability"),
                "hierarchy_level": visit.get("hierarchyLevel"),
            })

        elif "activity" in seg:
            act = seg["activity"]
            top = act.get("topCandidate", {}) or {}
            mode = top.get("type") or act.get("activityType")
            for key in ("start", "end"):
                value = act.get(key)
                lat, lon = parse_latlng(value)
                if lat is None and isinstance(value, dict):
                    lat, lon = parse_latlng(value.get("latLng"))
                if lat is None:
                    continue
                activity_records.append({
                    "segment_id": seg_id,
                    "endpoint": key,
                    "timestamp": start_time if key == "start" else end_time,
                    "lat": lat,
                    "lon": lon,
                    "mode": mode,
                    "mode_probability": top.get("probability"),
                    "distance_m": _safe_float(act.get("distanceMeters")),
                    "activity_probability": _safe_float(act.get("probability")),
                })

        else:
            # Unknown kind — capture for the audit
            other_segments.append({
                "segment_id": seg_id,
                "start_time": start_time,
                "end_time": end_time,
                "keys": sorted(seg.keys()),
            })

    def _finalise(records, default_cols, sort_col="timestamp"):
        if not records:
            return pd.DataFrame(columns=default_cols)
        df = pd.DataFrame(records)
        for col in df.columns:
            if "time" in col.lower():
                df[col] = _to_utc(df[col])
        if sort_col in df.columns:
            df = df.sort_values(sort_col).reset_index(drop=True)
        return df

    return {
        "timeline_paths": _finalise(
            tl_records,
            ["segment_id", "timestamp", "lat", "lon", "segment_start", "segment_end"],
        ),
        "visits": _finalise(
            visit_records,
            ["segment_id", "start_time", "end_time", "lat", "lon",
             "place_id", "semantic_type", "probability", "hierarchy_level"],
            sort_col="start_time",
        ),
        "activities": _finalise(
            activity_records,
            ["segment_id", "endpoint", "timestamp", "lat", "lon",
             "mode", "mode_probability", "distance_m", "activity_probability"],
        ),
        "other_segments": other_segments,
    }


def _safe_float(x):
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def parse_glh_file(filepath: str, *, volunteer: str | None = None) -> dict[str, Any]:
    """
    Parse a single GLH JSON file and return a unified result dict.

    Parameters
    ----------
    filepath : str
        Path to the GLH .json file.
    volunteer : str, optional
        Volunteer identifier. If supplied, added as a `volunteer` column to
        every emitted DataFrame.

    Returns
    -------
    dict
        Keys:
            family            : 'A' or 'B'
            source_file       : os.path.basename(filepath)
            raw_signals       : DataFrame (Family A only; empty for B)
            activity_records  : DataFrame (Family A only; empty for B)
            wifi_scans        : DataFrame (Family A only; empty for B)
            timeline_paths    : DataFrame  (both families)
            visits            : DataFrame  (both families)
            activities        : DataFrame  (both families)
            other_segments    : list[dict] (both families)
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    family = detect_family(data)
    source_file = os.path.basename(filepath)

    if family == "A":
        raw = _parse_family_a_raw_signals(data)
    else:
        # Family B has no rawSignals; emit empty frames with the right columns.
        raw = {
            "raw_signals": pd.DataFrame(
                columns=["timestamp", "lat", "lon", "accuracy_m", "altitude_m", "speed_mps", "source"]
            ),
            "activity_records": pd.DataFrame(
                columns=["timestamp", "top_activity", "confidence", "n_candidates"]
            ),
            "wifi_scans": pd.DataFrame(columns=["timestamp", "n_access_points"]),
        }

    segments = _extract_segments(data)
    seg_out = _parse_segments(segments)

    result = {
        "family": family,
        "source_file": source_file,
        **raw,
        **seg_out,
    }

    # Stamp metadata onto every emitted DataFrame
    for key, val in list(result.items()):
        if isinstance(val, pd.DataFrame):
            if not val.empty:
                val.insert(0, "source_file", source_file)
                if volunteer is not None:
                    val.insert(0, "volunteer", volunteer)

    return result


def parse_volunteer_folder(folder_path: str, *, volunteer: str | None = None) -> dict[str, Any]:
    """
    Find the GLH JSON in a volunteer's folder and parse it.

    Behaviour: picks the *largest* .json in the folder (the timeline) and
    parses it. Returns the same shape as parse_glh_file().

    Returns
    -------
    dict
        Same structure as parse_glh_file().  Returns empty DataFrames if no
        .json is found.
    """
    if volunteer is None:
        volunteer = os.path.basename(os.path.normpath(folder_path))

    json_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".json")
    ]
    if not json_files:
        empty_cols = {
            "raw_signals": ["timestamp", "lat", "lon", "accuracy_m", "altitude_m", "speed_mps", "source"],
            "activity_records": ["timestamp", "top_activity", "confidence", "n_candidates"],
            "wifi_scans": ["timestamp", "n_access_points"],
            "timeline_paths": ["segment_id", "timestamp", "lat", "lon", "segment_start", "segment_end"],
            "visits": ["segment_id", "start_time", "end_time", "lat", "lon", "place_id", "semantic_type", "probability", "hierarchy_level"],
            "activities": ["segment_id", "endpoint", "timestamp", "lat", "lon", "mode", "mode_probability", "distance_m", "activity_probability"],
        }
        out = {"family": None, "source_file": None, "other_segments": []}
        out.update({k: pd.DataFrame(columns=v) for k, v in empty_cols.items()})
        return out

    # Largest JSON wins (the timeline; some folders may also contain small auxiliaries)
    main_json = max(json_files, key=os.path.getsize)
    return parse_glh_file(main_json, volunteer=volunteer)


# ─────────────────────────────────────────────────────────────────────────────
# CLI quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python glh_parser.py path/to/timeline.json")
        raise SystemExit(1)

    result = parse_glh_file(sys.argv[1])
    print(f"Family: {result['family']}  file: {result['source_file']}")
    for key in ("raw_signals", "activity_records", "wifi_scans",
                "timeline_paths", "visits", "activities"):
        df = result[key]
        print(f"  {key}: {len(df)} rows")
    print(f"  other_segments: {len(result['other_segments'])}")
