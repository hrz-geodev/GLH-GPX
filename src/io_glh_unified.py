from __future__ import annotations
from pathlib import Path
import json
import re
import pandas as pd


# --------- coordinate parsers ---------

_geo_re = re.compile(r"geo:(-?\d+\.?\d*),(-?\d+\.?\d*)")
_deg_re = re.compile(r"(-?\d+\.?\d*)\s*°?\s*,\s*(-?\d+\.?\d*)\s*°?")


def parse_latlon(s: str):
    """
    Supports:
      - "geo:55.943312,-3.196167"
      - "55.9430589°, -3.2071267°"
      - "55.9430589, -3.2071267"
    """
    if not isinstance(s, str):
        return None, None

    m = _geo_re.search(s)
    if m:
        return float(m.group(1)), float(m.group(2))

    m = _deg_re.search(s)
    if m:
        return float(m.group(1)), float(m.group(2))

    return None, None


def to_utc(ts) -> pd.Timestamp:
    return pd.to_datetime(ts, utc=True, errors="coerce")




def flatten_legacy_list(data: list, volunteer_id: str) -> pd.DataFrame:
    rows = []
    for seg_idx, seg in enumerate(data):
        if not isinstance(seg, dict):
            continue
        start = to_utc(seg.get("startTime"))
        end = to_utc(seg.get("endTime"))

        # activity
        if "activity" in seg and isinstance(seg["activity"], dict):
            act = seg["activity"]
            a_prob = act.get("probability")
            dist_m = act.get("distanceMeters")
            cand = act.get("topCandidate") or {}
            a_type = cand.get("type")
            a_type_prob = cand.get("probability")

            lat_s, lon_s = parse_latlon(act.get("start", ""))
            lat_e, lon_e = parse_latlon(act.get("end", ""))

            if pd.notna(start) and lat_s is not None:
                rows.append(dict(
                    volunteer_id=volunteer_id,
                    glh_format="legacy_list",
                    segment_id=seg_idx,
                    source_type="activity_start",
                    timestamp_utc=start,
                    lat=lat_s, lon=lon_s,
                    segment_start_utc=start, segment_end_utc=end,
                    activity_type=a_type,
                    activity_prob=a_type_prob,
                    segment_prob=a_prob,
                    distance_m=dist_m,
                    accuracy_m=pd.NA,
                    location_source=pd.NA,
                ))
            if pd.notna(end) and lat_e is not None:
                rows.append(dict(
                    volunteer_id=volunteer_id,
                    glh_format="legacy_list",
                    segment_id=seg_idx,
                    source_type="activity_end",
                    timestamp_utc=end,
                    lat=lat_e, lon=lon_e,
                    segment_start_utc=start, segment_end_utc=end,
                    activity_type=a_type,
                    activity_prob=a_type_prob,
                    segment_prob=a_prob,
                    distance_m=dist_m,
                    accuracy_m=pd.NA,
                    location_source=pd.NA,
                ))

        # visit
        if "visit" in seg and isinstance(seg["visit"], dict):
            v = seg["visit"]
            v_prob = v.get("probability")
            cand = v.get("topCandidate") or {}
            place_id = cand.get("placeID")
            sem = cand.get("semanticType")
            plc = cand.get("placeLocation")
            lat_p, lon_p = parse_latlon(plc or "")

            # represent visit point at startTime (you can switch to midpoint later)
            if pd.notna(start) and lat_p is not None:
                rows.append(dict(
                    volunteer_id=volunteer_id,
                    glh_format="legacy_list",
                    segment_id=seg_idx,
                    source_type="visit",
                    timestamp_utc=start,
                    lat=lat_p, lon=lon_p,
                    segment_start_utc=start, segment_end_utc=end,
                    activity_type=pd.NA,
                    activity_prob=pd.NA,
                    segment_prob=v_prob,
                    distance_m=pd.NA,
                    place_id=place_id,
                    semantic_type=sem,
                    accuracy_m=pd.NA,
                    location_source=pd.NA,
                ))

        # timelinePath (offset minutes)
        if "timelinePath" in seg and isinstance(seg["timelinePath"], list):
            for p in seg["timelinePath"]:
                if not isinstance(p, dict):
                    continue
                lat, lon = parse_latlon(p.get("point", ""))
                off = p.get("durationMinutesOffsetFromStartTime")
                try:
                    off_min = float(off) if off is not None else None
                except Exception:
                    off_min = None
                if lat is None or pd.isna(start) or off_min is None:
                    continue
                t = start + pd.Timedelta(minutes=off_min)
                rows.append(dict(
                    volunteer_id=volunteer_id,
                    glh_format="legacy_list",
                    segment_id=seg_idx,
                    source_type="timelinePath",
                    timestamp_utc=t,
                    lat=lat, lon=lon,
                    segment_start_utc=start, segment_end_utc=end,
                    activity_type=pd.NA,
                    activity_prob=pd.NA,
                    segment_prob=pd.NA,
                    distance_m=pd.NA,
                    accuracy_m=pd.NA,
                    location_source=pd.NA,
                ))

    return pd.DataFrame(rows)


# --------- semanticSegments format (your new Timeline.json style) ---------

def flatten_semantic_segments(obj: dict, volunteer_id: str) -> pd.DataFrame:
    segs = obj.get("semanticSegments", [])
    rows = []

    for seg_idx, seg in enumerate(segs):
        if not isinstance(seg, dict):
            continue

        start = to_utc(seg.get("startTime"))
        end = to_utc(seg.get("endTime"))

        # timelinePath with per-point time
        if "timelinePath" in seg and isinstance(seg["timelinePath"], list):
            for p in seg["timelinePath"]:
                if not isinstance(p, dict):
                    continue
                lat, lon = parse_latlon(p.get("point", ""))
                t = to_utc(p.get("time"))
                if lat is None or pd.isna(t):
                    continue
                rows.append(dict(
                    volunteer_id=volunteer_id,
                    glh_format="semanticSegments",
                    segment_id=seg_idx,
                    source_type="timelinePath",
                    timestamp_utc=t,
                    lat=lat, lon=lon,
                    segment_start_utc=start, segment_end_utc=end,
                    activity_type=pd.NA,
                    activity_prob=pd.NA,
                    segment_prob=pd.NA,
                    distance_m=pd.NA,
                    accuracy_m=pd.NA,
                    location_source=pd.NA,
                ))

        # visit
        if "visit" in seg and isinstance(seg["visit"], dict):
            v = seg["visit"]
            v_prob = v.get("probability")
            cand = v.get("topCandidate") or {}
            place_id = cand.get("placeID")
            sem = cand.get("semanticType")
            plc = (cand.get("placeLocation") or {}).get("latLng") if isinstance(cand.get("placeLocation"), dict) else cand.get("placeLocation")
            lat_p, lon_p = parse_latlon(plc or "")

            # represent visit at startTime for now
            if pd.notna(start) and lat_p is not None:
                rows.append(dict(
                    volunteer_id=volunteer_id,
                    glh_format="semanticSegments",
                    segment_id=seg_idx,
                    source_type="visit",
                    timestamp_utc=start,
                    lat=lat_p, lon=lon_p,
                    segment_start_utc=start, segment_end_utc=end,
                    activity_type=pd.NA,
                    activity_prob=pd.NA,
                    segment_prob=v_prob,
                    distance_m=pd.NA,
                    place_id=place_id,
                    semantic_type=sem,
                    accuracy_m=pd.NA,
                    location_source=pd.NA,
                ))

        # activity
        if "activity" in seg and isinstance(seg["activity"], dict):
            a = seg["activity"]
            a_prob = a.get("probability")
            dist_m = a.get("distanceMeters")
            cand = a.get("topCandidate") or {}
            a_type = cand.get("type")
            a_type_prob = cand.get("probability")

            lat_s, lon_s = parse_latlon(((a.get("start") or {}).get("latLng")) if isinstance(a.get("start"), dict) else a.get("start", ""))
            lat_e, lon_e = parse_latlon(((a.get("end") or {}).get("latLng")) if isinstance(a.get("end"), dict) else a.get("end", ""))

            if pd.notna(start) and lat_s is not None:
                rows.append(dict(
                    volunteer_id=volunteer_id,
                    glh_format="semanticSegments",
                    segment_id=seg_idx,
                    source_type="activity_start",
                    timestamp_utc=start,
                    lat=lat_s, lon=lon_s,
                    segment_start_utc=start, segment_end_utc=end,
                    activity_type=a_type,
                    activity_prob=a_type_prob,
                    segment_prob=a_prob,
                    distance_m=dist_m,
                    accuracy_m=pd.NA,
                    location_source=pd.NA,
                ))
            if pd.notna(end) and lat_e is not None:
                rows.append(dict(
                    volunteer_id=volunteer_id,
                    glh_format="semanticSegments",
                    segment_id=seg_idx,
                    source_type="activity_end",
                    timestamp_utc=end,
                    lat=lat_e, lon=lon_e,
                    segment_start_utc=start, segment_end_utc=end,
                    activity_type=a_type,
                    activity_prob=a_type_prob,
                    segment_prob=a_prob,
                    distance_m=dist_m,
                    accuracy_m=pd.NA,
                    location_source=pd.NA,
                ))

    # Optional: if this file has a separate "positions" stream with accuracyMeters/source,
    # we can add it as source_type="position" rows. (Keep separate; don't mix for training by default.)
    if "positions" in obj and isinstance(obj["positions"], list):
        for i, rec in enumerate(obj["positions"]):
            if not isinstance(rec, dict):
                continue
            pos = rec.get("position") if isinstance(rec.get("position"), dict) else rec
            lat, lon = parse_latlon(pos.get("LatLng", "") or pos.get("latLng", ""))
            t = to_utc(pos.get("timestamp"))
            if lat is None or pd.isna(t):
                continue
            rows.append(dict(
                volunteer_id=volunteer_id,
                glh_format="semanticSegments",
                segment_id=pd.NA,
                source_type="position",
                timestamp_utc=t,
                lat=lat, lon=lon,
                segment_start_utc=pd.NA,
                segment_end_utc=pd.NA,
                activity_type=pd.NA,
                activity_prob=pd.NA,
                segment_prob=pd.NA,
                distance_m=pd.NA,
                accuracy_m=pos.get("accuracyMeters", pd.NA),
                location_source=pos.get("source", pd.NA),
            ))

    return pd.DataFrame(rows)


def load_glh_points_unified(glh_json_path: Path, volunteer_id: str) -> pd.DataFrame:
    data = json.loads(glh_json_path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        df = flatten_legacy_list(data, volunteer_id)
    elif isinstance(data, dict) and "semanticSegments" in data:
        df = flatten_semantic_segments(data, volunteer_id)
    else:
        raise ValueError("Unknown GLH JSON format: expected list or dict with semanticSegments.")

    # standardize types
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df["segment_start_utc"] = pd.to_datetime(df.get("segment_start_utc"), utc=True, errors="coerce")
    df["segment_end_utc"] = pd.to_datetime(df.get("segment_end_utc"), utc=True, errors="coerce")

    # numeric columns
    for c in ["lat", "lon", "distance_m", "accuracy_m"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df
