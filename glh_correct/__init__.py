"""
glh_correct
===========

Tools for characterising and correcting Google Location History (GLH) error
against high-precision GPS reference data, and for applying the trained
indicator and HMM map-matching models to new GLH data.

Submodules
----------

Parsing & geometry
    glh_parser           parse Google Location History exports (rawSignals + timelinePaths)
    gpx_parser           parse high-precision GPX reference tracks
    projection           WGS84 ↔ British National Grid (EPSG:27700)
    cleaning             QC filters for GLH and GPX points
    sessionize           sessionise GLH points and GPX tracks
    matching             interpolate GPX truth at GLH timestamps

Map context
    networks             load OS MasterMap Highways and OSM road networks
    buildings            load OSM building footprints
    snapping             vectorised nearest-edge snap utility
    feature_engineering  produce per-point map-context features

Corrector models
    correction_rule_based   Stage 2 — rule-based snap-to-road corrector
    model_xgboost           Stage 3.1 indicator + Stage 3.2 classifier corrector
    model_hmm_mapmatch      Stage 5 — Newson-Krumm-style HMM map-matching

See `docs/USAGE.md` for end-to-end recipes and the `examples/` folder for
runnable scripts.
"""

__version__ = "0.1.0"
