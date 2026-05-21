"""
buildings.py
============
Building footprint utilities for the urban-canyon analysis.

Data source
-----------
`map/buildings.shp` — OpenStreetMap building footprints clipped to the
Edinburgh study area. 161,035 polygons, CRS WGS84 (EPSG:4326), attributes:
`osm_id`, `code`, `fclass`, `name`, `type`.

**No height attribute is present.** OSM coverage of `building:levels` /
`height` tags in Edinburgh is partial and was not exported. The features
this module provides are therefore footprint-based only:

    is_inside_building(point)         - boolean per point
    nearest_building_distance(point)  - metres to nearest building edge
    building_density(point, radius)   - count or area of buildings within r

All distances are computed in BNG (EPSG:27700) after a one-off reproject of
the buildings layer. The reprojected GeoDataFrame is cached as a pickle in
`outputs/cache/buildings_bng.pkl` so we only pay the projection cost once.

Typical usage
-------------
    from buildings import load_buildings_bng, add_building_features

    buildings = load_buildings_bng(project_root='.')
    matched = add_building_features(matched, buildings)
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Load + cache
# ─────────────────────────────────────────────────────────────────────────────

#: Building shapefiles to load. Each entry is (filename relative to map/, city).
#: Files that don't exist are silently skipped, so the loader works whether
#: or not the additional-city extracts have been added to the project.
_BUILDINGS_FILES: list[tuple[str, str]] = [
    ("buildings.shp",                                   "Edinburgh"),
    ("Glasgow_buildings.shp",                           "Glasgow"),
    ("manchester_gis_osm_buildings_a_free_1.shp",       "Manchester"),
    ("London_gis_osm_buildings_a_free_1.shp",           "London"),
    ("lancashire_gis_osm_buildings_a_free_1.shp",       "Lancashire"),
    ("Cumbria_gis_osm_buildings_a_free_1.shp",          "Cumbria"),
]


def _load_one_buildings_file(project_root: str, fname: str, city: str):
    """Load one buildings shapefile, project to BNG, keep useful attrs + city tag."""
    import geopandas as gpd

    full_path = os.path.join(project_root, "map", fname)
    if not os.path.exists(full_path):
        return None

    gdf = gpd.read_file(full_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if str(gdf.crs).upper() not in ("EPSG:27700",):
        gdf = gdf.to_crs("EPSG:27700")

    keep = [c for c in ("osm_id", "code", "fclass", "name", "type", "geometry")
            if c in gdf.columns]
    out = gdf[keep].copy()
    out["city"] = city
    return out


def load_buildings_bng(
    project_root: str = ".",
    *,
    cache_path: str = "outputs/cache/buildings_bng.pkl",
    force_rebuild: bool = False,
):
    """
    Load buildings from every available city shapefile, reprojected to BNG
    (EPSG:27700) and concatenated into a single GeoDataFrame.

    Looks for the following files under `map/` (silently skips any missing):

        buildings.shp                                 → city='Edinburgh'
        Glasgow_buildings.shp                         → city='Glasgow'
        manchester_gis_osm_buildings_a_free_1.shp     → city='Manchester'
        London_gis_osm_buildings_a_free_1.shp         → city='London'

    Output columns: osm_id, code, fclass, name, type, city, geometry
    (geometry in EPSG:27700).

    The combined layer is pickled to `outputs/cache/buildings_bng.pkl`.
    **Delete that pickle after adding new shapefiles** so the next call
    rebuilds with the new sources.
    """
    import geopandas as gpd

    full_cache = os.path.join(project_root, cache_path)
    if (not force_rebuild) and os.path.exists(full_cache):
        return pd.read_pickle(full_cache)

    parts = []
    for fname, city in _BUILDINGS_FILES:
        sub = _load_one_buildings_file(project_root, fname, city)
        if sub is None:
            continue
        parts.append(sub)

    if not parts:
        raise FileNotFoundError(
            f"No buildings shapefiles found under {os.path.join(project_root, 'map')} "
            f"matching any of: {[f for f, _ in _BUILDINGS_FILES]}"
        )

    combined = pd.concat(parts, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:27700")

    os.makedirs(os.path.dirname(full_cache), exist_ok=True)
    combined.to_pickle(full_cache)
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Spatial index helper
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_sindex(buildings):
    """Force sindex build (geopandas computes lazily)."""
    _ = buildings.sindex
    return buildings


# ─────────────────────────────────────────────────────────────────────────────
# Point-level features
# ─────────────────────────────────────────────────────────────────────────────

def is_inside_building(
    points_df: pd.DataFrame,
    buildings,
    *,
    east_col: str = "east",
    north_col: str = "north",
    out_col: str = "inside_building",
    osm_id_col: str = "inside_building_osm_id",
) -> pd.DataFrame:
    """
    Add a boolean `inside_building` column and (optionally) the matched
    `osm_id` of the containing polygon.

    Implementation: spatial join via geopandas STRtree. O(n + m log m).
    """
    import geopandas as gpd
    from shapely.geometry import Point

    out = points_df.copy()
    if out.empty:
        out[out_col] = pd.Series(dtype=bool)
        out[osm_id_col] = pd.Series(dtype=object)
        return out

    buildings = _ensure_sindex(buildings)

    pts = gpd.GeoDataFrame(
        out.reset_index(drop=False),  # keep original index in 'index' col
        geometry=[Point(x, y) for x, y in zip(out[east_col], out[north_col])],
        crs="EPSG:27700",
    )

    joined = gpd.sjoin(
        pts, buildings[["osm_id", "geometry"]], how="left", predicate="within"
    )

    # If a point sat exactly on multiple polygon boundaries it could match
    # multiple buildings; take the first by index.
    joined = joined.drop_duplicates(subset="index", keep="first")
    joined.set_index("index", inplace=True)

    # IMPORTANT: `joined.geometry` is the LEFT-side geometry (the input point)
    # and is therefore always non-null. To tell whether the sjoin actually
    # matched a building polygon, check `index_right` — sjoin sets this to NaN
    # when no polygon was matched.
    matched_mask = joined["index_right"].notna() if "index_right" in joined.columns \
                   else joined["osm_id"].notna()
    out[out_col] = matched_mask.reindex(out.index).fillna(False).astype(bool)

    # The matched building's osm_id stays in `osm_id` (no name collision with
    # the points' columns).
    out[osm_id_col] = joined["osm_id"].reindex(out.index) \
        if "osm_id" in joined.columns else pd.Series(pd.NA, index=out.index)
    return out


def nearest_building_distance(
    points_df: pd.DataFrame,
    buildings,
    *,
    east_col: str = "east",
    north_col: str = "north",
    out_col: str = "nearest_building_m",
    max_search_m: float = 200.0,
) -> pd.DataFrame:
    """
    Add `nearest_building_m`: distance in metres from each point to the
    nearest building polygon edge. Points inside a polygon get 0.

    `max_search_m` caps the query window for speed. Points further than
    this from any building get NaN (the assumption is that >200 m from
    *any* building is open ground for our purposes).
    """
    import geopandas as gpd
    from shapely.geometry import Point

    out = points_df.copy()
    if out.empty:
        out[out_col] = pd.Series(dtype=float)
        return out

    buildings = _ensure_sindex(buildings)
    geom = buildings.geometry.values
    sindex = buildings.sindex

    dists = np.full(len(out), np.nan, dtype=float)
    easts = out[east_col].values
    norths = out[north_col].values

    for i, (e, n) in enumerate(zip(easts, norths)):
        if np.isnan(e) or np.isnan(n):
            continue
        # query envelope: a search box around the point
        env = (e - max_search_m, n - max_search_m,
               e + max_search_m, n + max_search_m)
        candidate_ix = list(sindex.intersection(env))
        if not candidate_ix:
            continue
        p = Point(e, n)
        # Min distance to any candidate (polygon edge for outside points;
        # 0.0 for points strictly inside since shapely treats inside as 0).
        d_min = min(geom[ix].distance(p) for ix in candidate_ix)
        dists[i] = d_min

    out[out_col] = dists
    return out


def building_density(
    points_df: pd.DataFrame,
    buildings,
    *,
    radius_m: float = 50.0,
    east_col: str = "east",
    north_col: str = "north",
    out_count_col: str | None = None,
    out_area_col: str | None = None,
) -> pd.DataFrame:
    """
    For each point, count buildings (and optionally summed area, m²) whose
    geometry intersects a circle of `radius_m` around the point.

    Column naming defaults to `n_buildings_{radius}m` and `building_area_{radius}m_m2`.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    out = points_df.copy()

    rad_str = f"{int(radius_m)}m"
    count_col = out_count_col or f"n_buildings_{rad_str}"
    area_col  = out_area_col  or f"building_area_{rad_str}_m2"

    if out.empty:
        out[count_col] = pd.Series(dtype=int)
        out[area_col] = pd.Series(dtype=float)
        return out

    buildings = _ensure_sindex(buildings)
    geom = buildings.geometry.values
    areas = buildings.geometry.area.values
    sindex = buildings.sindex

    counts = np.zeros(len(out), dtype=int)
    sum_area = np.zeros(len(out), dtype=float)
    easts = out[east_col].values
    norths = out[north_col].values

    for i, (e, n) in enumerate(zip(easts, norths)):
        if np.isnan(e) or np.isnan(n):
            counts[i] = 0
            sum_area[i] = 0.0
            continue
        circle = Point(e, n).buffer(radius_m)
        env = (e - radius_m, n - radius_m, e + radius_m, n + radius_m)
        candidate_ix = list(sindex.intersection(env))
        if not candidate_ix:
            continue
        intersected_mask = [geom[ix].intersects(circle) for ix in candidate_ix]
        intersected_ix = [ix for ix, m in zip(candidate_ix, intersected_mask) if m]
        counts[i] = len(intersected_ix)
        sum_area[i] = float(sum(areas[ix] for ix in intersected_ix))

    out[count_col] = counts
    out[area_col] = sum_area
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: add all three feature blocks
# ─────────────────────────────────────────────────────────────────────────────

def add_building_features(
    df: pd.DataFrame,
    buildings: Optional["object"] = None,
    *,
    project_root: str = ".",
    radii_m: tuple[float, ...] = (50.0, 100.0),
    east_col: str = "east",
    north_col: str = "north",
) -> pd.DataFrame:
    """
    Add the full set of building-derived features to a points DataFrame.

    Adds columns:
        inside_building            (bool)
        nearest_building_m         (float, NaN if >200 m from any building)
        n_buildings_50m            (int)
        building_area_50m_m2       (float)
        n_buildings_100m           (int)
        building_area_100m_m2      (float)

    Pass `buildings` directly to avoid re-loading; otherwise loads from
    the standard cache via `load_buildings_bng`.
    """
    if buildings is None:
        buildings = load_buildings_bng(project_root=project_root)

    out = is_inside_building(df, buildings, east_col=east_col, north_col=north_col)
    out = nearest_building_distance(out, buildings, east_col=east_col, north_col=north_col)
    for r in radii_m:
        out = building_density(out, buildings, radius_m=r,
                               east_col=east_col, north_col=north_col)
    return out
