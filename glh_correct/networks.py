"""
networks.py
===========
Loaders for the OS MasterMap Highways shapefiles **and** OSM Geofabrik
road extracts for additional cities. Builds three combined networks
(carriageway / pedestrian / full) used downstream.

Source files (in `map/`)
------------------------
Edinburgh — OS MasterMap (rich attributes, BNG):
    main_ExportFeature1.shp     - OS Road Link        (carriageway)
    main_connectinglink1.shp    - OS Connecting Link  (junction connectors)
    main_pathlink1.shp          - OS Path Link        (footpaths/cycleways)
    main_Street.shp             - OS Street (named)   (for labelling only)

Glasgow / Manchester / London — OSM Geofabrik (simpler schema, WGS84 →
reprojected to BNG on load):
    Glasgow_streets.shp,                                Glasgow_buildings.shp
    manchester_gis_osm_roads_free_1.shp,                manchester_gis_osm_buildings_a_free_1.shp
    London_gis_osm_roads_free_1.shp,                    London_gis_osm_buildings_a_free_1.shp

OSM `fclass` is used to split each city's roads into carriageway-class
(motorway, trunk, primary, secondary, tertiary, residential, service, …)
versus pedestrian-class (footway, path, cycleway, pedestrian, steps,
bridleway). Each row is tagged with `network_kind` and `city`.

Combined networks built here
----------------------------
    carriageway_network = Edinburgh Road Link ∪ Connecting Link
                          ∪ OSM carriageway from Glasgow + Manchester + London
    pedestrian_network  = Edinburgh Path Link
                          ∪ OSM pedestrian from Glasgow + Manchester + London
    full_network        = carriageway ∪ pedestrian.

CRS: every output is in EPSG:27700 (British National Grid) so that snap
distances are in metres and consistent with `buildings.py`.

Caching: reprojected/combined networks are pickled under
`outputs/cache/networks/`. **Delete the cache pickles after adding new
shapefiles** so the next call rebuilds with the new sources.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Source filenames + the attribute columns we want to keep on each
# ─────────────────────────────────────────────────────────────────────────────

_ROAD_LINK_FILE        = "main_ExportFeature1.shp"
_CONNECTING_LINK_FILE  = "main_connectinglink1.shp"
_PATH_LINK_FILE        = "main_pathlink1.shp"
_STREET_FILE           = "main_Street.shp"

# DBF column names are truncated to 10 chars by shapefile spec. The
# below lists what we extract per layer and rename to friendlier names.

_ROAD_LINK_KEEP = {
    "TOID":        "toid",
    "roadname":    "road_name",
    "roadname_l":  "road_name_alt",
    "roadclassi":  "road_class",          # e.g. "A Road" / "B Road" / "Motorway"
    "formofway":   "form_of_way",         # e.g. "Single Carriageway" / "Roundabout"
    "averagewid":  "average_width_m",
    "minimumwid":  "minimum_width_m",
    "cyclefacil":  "cycle_facility",
    "wholelink":   "whole_link",
    "roadstruct":  "road_structure",
}

_CONNECTING_LINK_KEEP = {
    "TOID":       "toid",
    "pathnode":   "path_node",
    "connecting": "connecting",
}

_PATH_LINK_KEEP = {
    "identifier": "toid",
    "pathname":   "road_name",     # treat path names alongside road names
    "length":     "length_m",
    "cyclefacil": "cycle_facility",
    "wholelink":  "whole_link",
}


# ─────────────────────────────────────────────────────────────────────────────
# OSM Geofabrik integration (Glasgow / Manchester / London)
# ─────────────────────────────────────────────────────────────────────────────

# OSM `fclass` taxonomy. We split into carriageway-class (vehicle-friendly)
# vs pedestrian-class (foot/cycle). `_OSM_OTHER_FCLASSES` are skipped
# entirely (under construction, proposed, etc.).
_OSM_CARRIAGEWAY_FCLASSES = {
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "unclassified", "residential", "living_street",
    "service", "track", "track_grade1", "track_grade2",
    "track_grade3", "track_grade4", "track_grade5",
    "road",
}
_OSM_PEDESTRIAN_FCLASSES = {
    "footway", "pedestrian", "path", "bridleway",
    "cycleway", "steps", "corridor",
}


def _classify_osm_fclass(fclass: str) -> str:
    """Map OSM highway fclass to one of {'osm_carriageway', 'osm_pedestrian', 'osm_other'}."""
    if not isinstance(fclass, str):
        return "osm_other"
    fclass = fclass.lower().strip()
    if fclass in _OSM_CARRIAGEWAY_FCLASSES:
        return "osm_carriageway"
    if fclass in _OSM_PEDESTRIAN_FCLASSES:
        return "osm_pedestrian"
    return "osm_other"


# OSM road files we expect to find in `map/`. Each entry is (filename, city).
# Entries whose file is missing are silently skipped, so the loader still
# works on a project that doesn't have the extra cities.
_OSM_ROAD_FILES: list[tuple[str, str]] = [
    ("Glasgow_streets.shp",                       "Glasgow"),
    ("manchester_gis_osm_roads_free_1.shp",       "Manchester"),
    ("London_gis_osm_roads_free_1.shp",           "London"),
    ("lancashire_gis_osm_roads_free_1.shp",       "Lancashire"),
    ("Cumbria_gis_osm_roads_free_1.shp",          "Cumbria"),
]

# Keep these OSM road columns under unified names (others get dropped).
_OSM_ROAD_KEEP = {
    "osm_id":  "_osm_id",        # raw id; the unified `toid` is built from this
    "name":    "road_name",
    "fclass":  "road_class",
    "ref":     "road_ref",
    "oneway":  "osm_oneway",
    "maxspeed":"osm_maxspeed",
    "bridge":  "osm_bridge",
    "tunnel":  "osm_tunnel",
}


def load_osm_roads_for_city(project_root: str, fname: str, city: str):
    """
    Load one OSM Geofabrik road shapefile, project to BNG, classify rows
    into carriageway/pedestrian via fclass, drop 'osm_other' rows.

    Returns a GeoDataFrame with columns:
        toid, road_name, road_class, road_ref,
        osm_oneway, osm_maxspeed, osm_bridge, osm_tunnel,
        network_kind, city, geometry
    """
    import geopandas as gpd

    full_path = os.path.join(project_root, "map", fname)
    if not os.path.exists(full_path):
        raise FileNotFoundError(full_path)

    gdf = gpd.read_file(full_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if str(gdf.crs).upper() not in ("EPSG:27700",):
        gdf = gdf.to_crs("EPSG:27700")

    # Classify
    gdf["network_kind"] = gdf.get("fclass", "").astype(str).map(_classify_osm_fclass)
    gdf = gdf[gdf["network_kind"] != "osm_other"].copy()

    # Rename to unified schema
    present = {src: dst for src, dst in _OSM_ROAD_KEEP.items() if src in gdf.columns}
    out = gdf[list(present) + ["geometry", "network_kind"]].rename(columns=present)

    # Build a unified `toid` (string) from osm_id, prefixed to avoid collision
    # with OS TOIDs in the combined deduplication step.
    if "_osm_id" in out.columns:
        out["toid"] = "osm_" + out["_osm_id"].astype(str)
        out = out.drop(columns=["_osm_id"])
    out["city"] = city

    return out

_STREET_KEEP = {
    "USRN":       "usrn",
    "name":       "street_name",
    "streettype": "street_type",
    "town":       "town",
    "administra": "administrative_area",
    "locality":   "locality",
    "descriptor": "descriptor",
    "roadclassi": "road_classification",
}


# ─────────────────────────────────────────────────────────────────────────────
# Low-level loader for a single shapefile → BNG GeoDataFrame
# ─────────────────────────────────────────────────────────────────────────────

def _read_layer(
    project_root: str,
    fname: str,
    keep_columns: dict[str, str],
    network_kind: str,
):
    """
    Read one shapefile, reproject to EPSG:27700 if needed, keep only the
    columns in `keep_columns` (renamed), and add a `network_kind` column
    so the rows are distinguishable after a union.
    """
    import geopandas as gpd

    full_path = os.path.join(project_root, "map", fname)
    gdf = gpd.read_file(full_path)

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:27700")  # OS Highways shapefiles default
    if str(gdf.crs).upper() not in ("EPSG:27700",):
        gdf = gdf.to_crs("EPSG:27700")

    # Only keep configured attributes that actually exist on this layer
    present = {src: dst for src, dst in keep_columns.items() if src in gdf.columns}
    out = gdf[list(present) + ["geometry"]].copy()
    out = out.rename(columns=present)
    out["network_kind"] = network_kind
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public: load one layer at a time
# ─────────────────────────────────────────────────────────────────────────────

def load_road_link(project_root: str = "."):
    return _read_layer(project_root, _ROAD_LINK_FILE, _ROAD_LINK_KEEP, "road_link")


def load_connecting_link(project_root: str = "."):
    return _read_layer(project_root, _CONNECTING_LINK_FILE, _CONNECTING_LINK_KEEP, "connecting_link")


def load_path_link(project_root: str = "."):
    return _read_layer(project_root, _PATH_LINK_FILE, _PATH_LINK_KEEP, "path_link")


def load_street(project_root: str = "."):
    return _read_layer(project_root, _STREET_FILE, _STREET_KEEP, "street")


# ─────────────────────────────────────────────────────────────────────────────
# Combined networks with caching
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(project_root: str, name: str) -> str:
    return os.path.join(project_root, "outputs", "cache", "networks", f"{name}.pkl")


def _osm_subnet(project_root: str, kind: str):
    """
    Yield OSM road sub-frames matching `kind` ('osm_carriageway' or
    'osm_pedestrian') from every available OSM city extract.
    """
    out = []
    for fname, city in _OSM_ROAD_FILES:
        try:
            gdf = load_osm_roads_for_city(project_root, fname, city)
        except FileNotFoundError:
            continue
        sub = gdf[gdf["network_kind"] == kind].copy()
        if not sub.empty:
            out.append(sub)
    return out


def load_carriageway_network(
    project_root: str = ".",
    *,
    force_rebuild: bool = False,
):
    """
    Combined carriageway network.

    Edinburgh: OS Road Link ∪ Connecting Link (rich OS attributes).
    Plus:       OSM carriageway-class roads from Glasgow + Manchester + London.

    Deduplicated by `toid`. The OS rows have a TOID like '4000000027845321';
    OSM rows have 'osm_<osm_id>'. Cross-source TOIDs are guaranteed unique.
    """
    cache = _cache_path(project_root, "carriageway_network")
    if (not force_rebuild) and os.path.exists(cache):
        return pd.read_pickle(cache)

    import geopandas as gpd

    parts = [
        load_road_link(project_root),
        load_connecting_link(project_root),
    ]
    parts.extend(_osm_subnet(project_root, "osm_carriageway"))

    combined = pd.concat(parts, ignore_index=True)
    if "toid" in combined.columns:
        combined = combined.drop_duplicates(subset=["toid"], keep="first")
    # CRITICAL: reset_index after drop_duplicates so the GeoDataFrame has a
    # contiguous 0..N-1 index. Snapping looks up matched edges by index
    # label; a sparse index would mean some labels don't exist and the snap
    # would silently fall back to the wrong geometry.
    combined = combined.reset_index(drop=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:27700")

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    combined.to_pickle(cache)
    return combined


def load_pedestrian_network(
    project_root: str = ".",
    *,
    force_rebuild: bool = False,
):
    """
    Combined pedestrian network.

    Edinburgh: OS Path Link.
    Plus:       OSM pedestrian-class ways from Glasgow + Manchester + London
                (footway, path, cycleway, pedestrian, steps, bridleway).
    """
    cache = _cache_path(project_root, "pedestrian_network")
    if (not force_rebuild) and os.path.exists(cache):
        return pd.read_pickle(cache)

    import geopandas as gpd

    parts = [load_path_link(project_root)]
    parts.extend(_osm_subnet(project_root, "osm_pedestrian"))

    combined = pd.concat(parts, ignore_index=True)
    if "toid" in combined.columns:
        combined = combined.drop_duplicates(subset=["toid"], keep="first")
    combined = combined.reset_index(drop=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:27700")

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    combined.to_pickle(cache)
    return combined


def load_full_network(
    project_root: str = ".",
    *,
    force_rebuild: bool = False,
):
    """Carriageway ∪ Pedestrian. Used for mode-agnostic snapping."""
    cache = _cache_path(project_root, "full_network")
    if (not force_rebuild) and os.path.exists(cache):
        return pd.read_pickle(cache)

    import geopandas as gpd

    carr = load_carriageway_network(project_root, force_rebuild=force_rebuild)
    ped  = load_pedestrian_network(project_root, force_rebuild=force_rebuild)
    combined = pd.concat([carr, ped], ignore_index=True).reset_index(drop=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:27700")

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    combined.to_pickle(cache)
    return combined


def load_streets(project_root: str = ".", *, force_rebuild: bool = False):
    """Named-street entities. Labelling layer; not used for snapping."""
    cache = _cache_path(project_root, "streets")
    if (not force_rebuild) and os.path.exists(cache):
        return pd.read_pickle(cache)
    s = load_street(project_root)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    s.to_pickle(cache)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Convenience bundle
# ─────────────────────────────────────────────────────────────────────────────

def load_all_networks(project_root: str = ".", *, force_rebuild: bool = False) -> dict:
    """
    Load every combined network into a single dict.

    Returns
    -------
    {
        "carriageway": GeoDataFrame,
        "pedestrian":  GeoDataFrame,
        "full":        GeoDataFrame,
        "streets":     GeoDataFrame,
    }
    """
    return {
        "carriageway": load_carriageway_network(project_root, force_rebuild=force_rebuild),
        "pedestrian":  load_pedestrian_network(project_root, force_rebuild=force_rebuild),
        "full":        load_full_network(project_root, force_rebuild=force_rebuild),
        "streets":     load_streets(project_root, force_rebuild=force_rebuild),
    }
