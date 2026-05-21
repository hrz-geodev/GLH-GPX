"""
projection.py
=============
Coordinate-system conversions between WGS84 (EPSG:4326) and British National
Grid (EPSG:27700) used by OS MasterMap data.

All GLH and GPX data arrives in WGS84 (decimal degrees). OS MasterMap Highways
shapefiles are in BNG. Spatial joins, nearest-edge snapping, and accurate
short-distance arithmetic are easier in metric BNG.

Convention adopted across this project
--------------------------------------
- `lat`, `lon`  : WGS84 decimal degrees (float64)
- `east`,`north`: BNG metres (float64), EPSG:27700

The `add_bng_columns` helper adds `east` and `north` to any DataFrame that
already has `lat` and `lon`. The reverse helper `add_wgs84_columns` does the
inverse. Both are vectorised via pyproj.Transformer for speed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer


# EPSG:4326 (WGS84 lat/lon) → EPSG:27700 (British National Grid, easting/northing)
# always_xy=True keeps the conventional (lon, lat) → (east, north) call signature.
_WGS84_TO_BNG = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
_BNG_TO_WGS84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised forward / inverse transforms
# ─────────────────────────────────────────────────────────────────────────────

def wgs84_to_bng(lat, lon) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised WGS84 → BNG.

    Accepts pandas Series, numpy arrays, or scalars. Returns (east, north).
    """
    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)
    east, north = _WGS84_TO_BNG.transform(lon_arr, lat_arr)
    return east, north


def bng_to_wgs84(east, north) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised BNG → WGS84.

    Returns (lat, lon) — note the ordering matches the rest of the project.
    """
    east_arr = np.asarray(east, dtype=float)
    north_arr = np.asarray(north, dtype=float)
    lon, lat = _BNG_TO_WGS84.transform(east_arr, north_arr)
    return lat, lon


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame helpers
# ─────────────────────────────────────────────────────────────────────────────

def add_bng_columns(df: pd.DataFrame, *,
                    lat_col: str = "lat",
                    lon_col: str = "lon",
                    east_col: str = "east",
                    north_col: str = "north") -> pd.DataFrame:
    """
    Return a copy of `df` with BNG `east`/`north` columns added.

    Rows where lat/lon are NaN are preserved with NaN BNG values. This avoids
    pyproj raising on missing coords mid-stream.
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        raise KeyError(
            f"DataFrame must contain '{lat_col}' and '{lon_col}' columns "
            f"(got {list(df.columns)})."
        )

    out = df.copy()
    mask = out[lat_col].notna() & out[lon_col].notna()
    out[east_col] = np.nan
    out[north_col] = np.nan
    if mask.any():
        east, north = wgs84_to_bng(out.loc[mask, lat_col], out.loc[mask, lon_col])
        out.loc[mask, east_col] = east
        out.loc[mask, north_col] = north
    return out


def add_wgs84_columns(df: pd.DataFrame, *,
                      east_col: str = "east",
                      north_col: str = "north",
                      lat_col: str = "lat",
                      lon_col: str = "lon") -> pd.DataFrame:
    """Return a copy of `df` with WGS84 `lat`/`lon` columns added."""
    if east_col not in df.columns or north_col not in df.columns:
        raise KeyError(
            f"DataFrame must contain '{east_col}' and '{north_col}' columns "
            f"(got {list(df.columns)})."
        )
    out = df.copy()
    mask = out[east_col].notna() & out[north_col].notna()
    out[lat_col] = np.nan
    out[lon_col] = np.nan
    if mask.any():
        lat, lon = bng_to_wgs84(out.loc[mask, east_col], out.loc[mask, north_col])
        out.loc[mask, lat_col] = lat
        out.loc[mask, lon_col] = lon
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Distance helpers (metric, BNG-based)
# ─────────────────────────────────────────────────────────────────────────────

def bng_distance(e1, n1, e2, n2) -> np.ndarray:
    """
    Vectorised Euclidean distance between two BNG points (metres).

    Valid for short-to-mid distances within BNG's defined area (UK), where the
    projection's local distortion is small (sub-metre over a few km).
    """
    e1 = np.asarray(e1, dtype=float)
    n1 = np.asarray(n1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    n2 = np.asarray(n2, dtype=float)
    return np.hypot(e2 - e1, n2 - n1)
