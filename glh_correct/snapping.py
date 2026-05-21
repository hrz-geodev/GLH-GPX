"""
snapping.py
===========
Nearest-edge snapping of points to a road / path network.

For each input point this module returns:
  - The closest point on the closest network edge (in BNG metres).
  - The snap distance from the input.
  - The attributes of the snapped-to edge (TOID, road class, road name, …).

The implementation is vectorised over input points using geopandas'
`sjoin_nearest` (which uses STRtree under the hood). For a typical Stage-1
matched parquet (a few thousand to a few hundred thousand points) it runs
in seconds; for the full multi-volunteer dataset it remains tractable.

Output convention
-----------------
When you snap a points DataFrame to a network, the result is a new
DataFrame with the snapped position and edge metadata added under a
configurable prefix (default `snap_`). Original columns are preserved.

    snap_east, snap_north   - the position on the network edge (EPSG:27700)
    snap_lat,  snap_lon     - the same in WGS84 (added if input had lat/lon)
    snap_distance_m         - distance from input point to snapped position
    snap_toid               - matched edge TOID (or NaN)
    snap_network_kind       - 'road_link' / 'connecting_link' / 'path_link'
    snap_road_name          - road / path name where present
    snap_road_class         - road classification where present
    snap_form_of_way        - form of way where present

A `max_snap_m` parameter (default 100 m) caps the search radius. Points
further from any edge than that are returned with NaN snap fields.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .projection import bng_to_wgs84


DEFAULT_MAX_SNAP_M = 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _attribute_aliases() -> dict[str, list[str]]:
    """
    Map a logical attribute name to the candidate column names the
    network GeoDataFrames may use. networks.py unifies OS + OSM rows to
    `road_class` / `form_of_way` / `road_name`, but we also accept the raw
    OS-truncated names as a fallback so this works on legacy caches.
    """
    return {
        "snap_toid":         ["toid", "TOID"],
        "snap_network_kind": ["network_kind"],
        "snap_road_name":    ["road_name", "road_name_alt", "street_name", "pathname", "name"],
        "snap_road_class":   ["road_class",   "road_classification", "roadclassi", "roadclas_1", "fclass"],
        "snap_form_of_way":  ["form_of_way",  "formofway", "formofway_"],
        "snap_city":         ["city"],
    }


def _pick_first_present(row: pd.Series, candidates: list[str]):
    for c in candidates:
        if c in row.index and pd.notna(row[c]):
            return row[c]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public: snap a points DataFrame to a network
# ─────────────────────────────────────────────────────────────────────────────

def snap_points_to_network(
    points_df: pd.DataFrame,
    network,
    *,
    east_col: str = "east",
    north_col: str = "north",
    max_snap_m: float = DEFAULT_MAX_SNAP_M,
    prefix: str = "snap_",
    add_wgs84: bool = True,
) -> pd.DataFrame:
    """
    Snap each row of `points_df` to its nearest edge in `network`.

    Parameters
    ----------
    points_df : DataFrame
        Must contain BNG `east`/`north` columns. WGS84 `lat`/`lon` is
        optional; if present, snapped lat/lon are also added.
    network : geopandas.GeoDataFrame
        A line network in EPSG:27700, e.g. from `networks.load_carriageway_network`.
        Must contain a `geometry` column of LineString / MultiLineString.
    max_snap_m : float
        Snap distance cap. Points further than this from any edge get NaN
        snap fields (a strong signal in itself for downstream features).
    prefix : str
        Column prefix for snap outputs (default `snap_`).

    Returns
    -------
    DataFrame
        `points_df.copy()` with added snap columns.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    out = points_df.copy()
    if out.empty:
        for c in (f"{prefix}east", f"{prefix}north", f"{prefix}distance_m"):
            out[c] = pd.Series(dtype=float)
        for c in (f"{prefix}toid", f"{prefix}network_kind",
                  f"{prefix}road_name", f"{prefix}road_class", f"{prefix}form_of_way"):
            out[c] = pd.Series(dtype=object)
        if add_wgs84:
            out[f"{prefix}lat"] = pd.Series(dtype=float)
            out[f"{prefix}lon"] = pd.Series(dtype=float)
        return out

    # Build a GeoDataFrame of valid input points (skip rows with NaN east/north)
    valid_mask = out[east_col].notna() & out[north_col].notna()
    valid_idx = out.index[valid_mask]

    pts = gpd.GeoDataFrame(
        out.loc[valid_idx, [east_col, north_col]].rename(
            columns={east_col: "_e", north_col: "_n"}
        ),
        geometry=[Point(e, n) for e, n in zip(out.loc[valid_idx, east_col],
                                              out.loc[valid_idx, north_col])],
        crs="EPSG:27700",
    )
    pts["_orig_index"] = valid_idx

    # Sjoin-nearest with distance, capped at max_snap_m
    network = network.copy()
    # Make sure a spatial index exists
    _ = network.sindex

    joined = gpd.sjoin_nearest(
        pts, network,
        how="left",
        max_distance=max_snap_m,
        distance_col="_snap_distance_m",
    )
    # Multiple equidistant matches → keep the first
    joined = joined.drop_duplicates(subset="_orig_index", keep="first").set_index("_orig_index")

    # For each matched row, project the input point onto the matched edge to
    # get the actual snapped position. (sjoin_nearest gives the edge, not the
    # snapped coordinate on the edge.)
    snapped_e = pd.Series(np.nan, index=out.index, dtype=float)
    snapped_n = pd.Series(np.nan, index=out.index, dtype=float)
    dist_col  = pd.Series(np.nan, index=out.index, dtype=float)

    if "index_right" in joined.columns:
        right_idx = joined["index_right"]
    else:
        right_idx = pd.Series(index=joined.index, dtype="Int64")

    # IMPORTANT: `index_right` from sjoin_nearest is the LABEL of the matched
    # right-side row (network), not its positional index. Use .loc[] so we
    # always look up by label. Using .iloc[] silently grabs the wrong
    # geometry when the network has a sparse / non-contiguous index — which
    # happens whenever drop_duplicates removed any rows during the
    # combined-network build (e.g. when the same OSM way appears in both the
    # Manchester and London Geofabrik extracts).
    network_geom = network.geometry
    for orig_idx, edge_label in right_idx.dropna().items():
        try:
            # edge_label comes through as float from pandas because of the NaN
            # column type; cast to whatever the network index uses.
            edge_label_native = type(network_geom.index[0])(edge_label) \
                if len(network_geom.index) else edge_label
        except (TypeError, ValueError):
            edge_label_native = edge_label
        try:
            edge_geom = network_geom.loc[edge_label_native]
        except KeyError:
            continue
        p = Point(out.at[orig_idx, east_col], out.at[orig_idx, north_col])
        # `project` returns linear distance along the edge; `interpolate` gives
        # the corresponding point on the edge.
        s = edge_geom.project(p)
        snapped_pt = edge_geom.interpolate(s)
        # Sanity-check: actual point-to-snapped distance should agree with
        # sjoin_nearest's reported distance to within ~1 m. If they disagree
        # wildly the snap is suspect — drop it rather than emit a wrong
        # corrected position.
        sjoin_dist = float(joined.at[orig_idx, "_snap_distance_m"])
        actual_dist = float(p.distance(snapped_pt))
        if not np.isfinite(sjoin_dist) or abs(actual_dist - sjoin_dist) > max(1.0, 0.05 * max(actual_dist, sjoin_dist)):
            continue
        snapped_e[orig_idx] = snapped_pt.x
        snapped_n[orig_idx] = snapped_pt.y
        dist_col[orig_idx]  = sjoin_dist

    out[f"{prefix}east"] = snapped_e
    out[f"{prefix}north"] = snapped_n
    out[f"{prefix}distance_m"] = dist_col

    # Optionally add WGS84 (we already have a vectorised inverse transform)
    if add_wgs84:
        m = out[f"{prefix}east"].notna() & out[f"{prefix}north"].notna()
        out[f"{prefix}lat"] = np.nan
        out[f"{prefix}lon"] = np.nan
        if m.any():
            lat, lon = bng_to_wgs84(out.loc[m, f"{prefix}east"], out.loc[m, f"{prefix}north"])
            out.loc[m, f"{prefix}lat"] = lat
            out.loc[m, f"{prefix}lon"] = lon

    # Attribute columns — pick the first present alias per row
    aliases = _attribute_aliases()
    for out_col, candidates in aliases.items():
        col_present = [c for c in candidates if c in joined.columns]
        if not col_present:
            out[out_col] = pd.NA
            continue
        # Pick first non-null candidate per row
        out[out_col] = pd.NA
        for c in col_present:
            ser = joined[c].reindex(out.index)
            out[out_col] = out[out_col].where(out[out_col].notna(), ser)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: distance-only (no edge attributes), for feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def distance_to_network(
    points_df: pd.DataFrame,
    network,
    *,
    east_col: str = "east",
    north_col: str = "north",
    max_search_m: float = 500.0,
    out_col: str = "dist_to_network_m",
) -> pd.DataFrame:
    """
    Cheaper variant: return only the snap *distance* to the network, no
    edge attributes or snapped coordinates. Use this when building feature
    columns like `dist_to_nearest_carriageway_m`.

    Points further than `max_search_m` get NaN.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    out = points_df.copy()
    if out.empty:
        out[out_col] = pd.Series(dtype=float)
        return out

    valid_mask = out[east_col].notna() & out[north_col].notna()
    valid_idx = out.index[valid_mask]

    pts = gpd.GeoDataFrame(
        out.loc[valid_idx, [east_col, north_col]],
        geometry=[Point(e, n) for e, n in zip(out.loc[valid_idx, east_col],
                                              out.loc[valid_idx, north_col])],
        crs="EPSG:27700",
    )
    pts["_orig_index"] = valid_idx

    _ = network.sindex
    joined = gpd.sjoin_nearest(
        pts, network,
        how="left",
        max_distance=max_search_m,
        distance_col="_dist_m",
    )
    joined = joined.drop_duplicates(subset="_orig_index", keep="first").set_index("_orig_index")

    dist_series = pd.Series(np.nan, index=out.index, dtype=float)
    dist_series.update(joined["_dist_m"])
    out[out_col] = dist_series
    return out
