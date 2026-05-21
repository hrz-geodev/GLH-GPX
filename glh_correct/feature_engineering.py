"""
feature_engineering.py
======================
Map-context feature builders for Stage 2 / Stage 3.

For each GLH point in a matched DataFrame, we attach a block of map-derived
attributes that downstream code uses in two ways:

1. **As Stage 3 model features** (XGBoost / sequence models). Distance to
   nearest carriageway, building density, etc. let the model condition
   prediction on map context — e.g. "GLH error tends to grow with building
   density and shrink near major roads".

2. **As inputs to the Stage 2 rule-based corrector**. The corrector picks
   the closer of carriageway- vs pedestrian-network snaps, which is exactly
   what we compute here.

Columns added (prefix conventions match `snapping.snap_points_to_network`)
------------------------------------------------------------------------
Nearest-carriageway block:
    nearest_car_east, nearest_car_north,
    nearest_car_distance_m,
    nearest_car_toid, nearest_car_network_kind,
    nearest_car_road_name, nearest_car_road_class, nearest_car_form_of_way

Nearest-pedestrian block:
    nearest_ped_east, nearest_ped_north,
    nearest_ped_distance_m,
    nearest_ped_toid, nearest_ped_network_kind,
    nearest_ped_road_name, nearest_ped_road_class, nearest_ped_form_of_way

Bearing block (cyclic-encoded direction from raw GLH to the snapped point):
    bearing_to_nearest_car_sin, bearing_to_nearest_car_cos
    bearing_to_nearest_ped_sin, bearing_to_nearest_ped_cos

    Bearing convention: 0 rad = north (+y, +northing), π/2 = east
    (+x, +easting). sin/cos pair keeps the encoding cyclic (a value near
    0° doesn't look numerically distant from 360°). Values are NaN where
    no snap was produced (no edge within max_snap_m).

Building block (from src/buildings.py):
    inside_building (bool),
    inside_building_osm_id,
    nearest_building_m,
    n_buildings_50m, building_area_50m_m2,
    n_buildings_100m, building_area_100m_m2

Convenience derived columns:
    dist_to_nearest_network_m  (min of nearest_car_distance_m, nearest_ped_distance_m)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .snapping import snap_points_to_network
from .buildings import add_building_features, load_buildings_bng


def add_map_context_features(
    df: pd.DataFrame,
    networks: dict,
    *,
    buildings=None,
    project_root: str = ".",
    east_col: str | None = None,
    north_col: str | None = None,
    max_snap_m: float = 100.0,
    building_radii_m: tuple[float, ...] = (50.0, 100.0),
) -> pd.DataFrame:
    """
    Attach the full map-context feature block to a points DataFrame.

    Parameters
    ----------
    df : DataFrame
        Should contain BNG coordinate columns. If called on a Stage-1
        matched parquet the defaults `glh_east` / `glh_north` apply; if
        called on a raw cleaned points DataFrame, set east_col/north_col
        to `'east'` / `'north'`.
    networks : dict
        From `networks.load_all_networks(...)`. Must contain
        'carriageway' and 'pedestrian' keys.
    buildings : GeoDataFrame, optional
        From `buildings.load_buildings_bng(...)`. Loaded on demand if None.
    project_root : str
        Used only when `buildings` is None.
    max_snap_m : float
        Snap radius cap for the carriageway / pedestrian snaps (default 100 m).
    building_radii_m : tuple
        Radii (metres) at which to count buildings around each point.

    Returns
    -------
    DataFrame
        A copy of `df` with all map-context columns added.
    """
    if east_col is None or north_col is None:
        # Detect the coordinate columns. Matched DataFrames use glh_east/glh_north.
        if "glh_east" in df.columns and "glh_north" in df.columns:
            east_col, north_col = "glh_east", "glh_north"
        elif "east" in df.columns and "north" in df.columns:
            east_col, north_col = "east", "north"
        else:
            raise KeyError(
                "feature_engineering: could not find BNG coordinate columns; "
                "pass east_col / north_col explicitly."
            )

    out = df.copy()
    if out.empty:
        return out

    # ── Carriageway snap (returns position + attributes) ────────────────────
    out = snap_points_to_network(
        out, networks["carriageway"],
        east_col=east_col, north_col=north_col,
        max_snap_m=max_snap_m,
        prefix="nearest_car_",
        add_wgs84=False,
    )

    # ── Pedestrian snap ─────────────────────────────────────────────────────
    out = snap_points_to_network(
        out, networks["pedestrian"],
        east_col=east_col, north_col=north_col,
        max_snap_m=max_snap_m,
        prefix="nearest_ped_",
        add_wgs84=False,
    )

    # Convenience: distance to whichever network is closest
    out["dist_to_nearest_network_m"] = np.fmin(
        out["nearest_car_distance_m"].astype(float),
        out["nearest_ped_distance_m"].astype(float),
    )

    # ── Bearing-to-nearest-road (cyclic-encoded) ────────────────────────────
    # For each nearest-edge snap we know the projected on-edge position.
    # Bearing from raw GLH to that snapped point gives the direction in
    # which the nearest road lies. Encoded as (sin, cos) of the angle so
    # the model sees a continuous, cyclic representation (0° ~ 360°).
    # Where no snap was produced (e.g. point > max_snap_m from any edge),
    # the corresponding nearest_*_east/north columns are NaN and the
    # bearing inherits NaN automatically.
    for prefix in ("nearest_car_", "nearest_ped_"):
        e_col = f"{prefix}east"
        n_col = f"{prefix}north"
        if e_col in out.columns and n_col in out.columns:
            dx = out[e_col].astype(float) - out[east_col].astype(float)
            dy = out[n_col].astype(float) - out[north_col].astype(float)
            # atan2(east, north) → 0 = north, +π/2 = east, ±π = south
            bearing = np.arctan2(dx, dy)
            out[f"bearing_to_{prefix.rstrip('_')}_sin"] = np.sin(bearing)
            out[f"bearing_to_{prefix.rstrip('_')}_cos"] = np.cos(bearing)

    # ── Buildings block ─────────────────────────────────────────────────────
    if buildings is None:
        buildings = load_buildings_bng(project_root=project_root)

    out = add_building_features(
        out, buildings,
        project_root=project_root,
        radii_m=building_radii_m,
        east_col=east_col, north_col=north_col,
    )

    return out
