"""
model_xgboost.py
================
Stage 3 — XGBoost deviation-magnitude regressor + hybrid (ML × rule-based)
corrector.

Design history
--------------
Iteration 1 (delta vector, with position leakage). Predicted
   (delta_east, delta_north) directly, with glh_east/glh_north among
   features. Under LOVO the model memorised fold-specific geography and
   degenerated to ~zero on held-out folds.
Iteration 2 (delta vector, leakage removed). Removed glh_east/glh_north
   and the cyclic time features. Importance table became clean — but the
   model still predicted close to zero, because the map-context features
   carry useful signal for the *magnitude* of GLH error but essentially
   none for its *direction*. Signed-vector regression therefore collapses
   to the conditional mean (~0).
Iteration 3 (current — magnitude + hybrid). The model predicts the **magnitude**
   of error (`deviation_m`, log-transformed) instead of the signed delta.
   The Stage-2 rule-based corrector already provides a *direction* (from
   raw GLH toward the snapped position). The two are blended through an
   alpha-gated snap:

        α = predicted_deviation_m
            ─────────────────────────────────────────
            predicted_deviation_m + snap_distance_m

        pred_position = raw_glh + α · (snapped − raw_glh)

   When the model thinks error is large relative to the snap distance,
   α → 1 → take the Stage-2 snap. When it thinks error is small, α → 0
   → keep raw. The hybrid corrector cannot make easy points worse
   (small predicted_dev ⇒ no movement) and only acts on points where
   the predicted error exceeds the cost of moving to the network.

Why a single regressor (not two)
--------------------------------
The target is the magnitude of error (positive, log-transformed). Direction
is supplied externally by the rule-based snap. One XGBoost regressor
suffices.

Targets / features
------------------
Target  y     : log1p(deviation_m)  →  trains regressor on log scale; at
                inference time we exp/expm1 to recover metres.
Features X    : only *relative / local* attributes — no glh_east/north,
                no absolute time, no volunteer id. See `_NUMERIC_FEATURES`,
                `_BOOLEAN_FEATURES`, `_CATEGORICAL_FEATURES`.

Public API
----------
    prepare_features(df)              -> (X, y, feature_columns, cat_levels)
    train(X_train, y_train, ...)      -> XGBModelBundle
    predict_deviation(bundle, X)      -> ndarray (metres, expm1 applied)
    apply_model(bundle, df)           -> df with predicted_deviation_m,
                                                 pred_glh_east/north/lat/lon,
                                                 pred_deviation_m,
                                                 improvement_vs_raw_m,
                                                 improvement_vs_stage2_m,
                                                 hybrid_alpha
    feature_importance(bundle, ...)
    save_bundle / load_bundle
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .projection import bng_distance, bng_to_wgs84


# ─────────────────────────────────────────────────────────────────────────────
# Feature specification — *no* absolute positions or absolute time.
# ─────────────────────────────────────────────────────────────────────────────

_NUMERIC_FEATURES: list[str] = [
    "glh_accuracy_m",
    "glh_speed_mps",
    "nearest_car_distance_m",
    "nearest_ped_distance_m",
    "dist_to_nearest_network_m",
    "nearest_building_m",
    "n_buildings_50m",
    "building_area_50m_m2",
    "n_buildings_100m",
    "building_area_100m_m2",
    "corrected_snap_distance_m",
    # Bearing-to-nearest-road features (cyclic sin/cos encoding). Give the
    # model the *direction* of the nearest road as well as the distance to it.
    "bearing_to_nearest_car_sin",
    "bearing_to_nearest_car_cos",
    "bearing_to_nearest_ped_sin",
    "bearing_to_nearest_ped_cos",
]

_BOOLEAN_FEATURES: list[str] = [
    "inside_building",
]

_CATEGORICAL_FEATURES: list[str] = [
    "glh_source",
    "glh_layer",
    "corrected_glh_network_kind",
    "nearest_car_road_class",
    "nearest_car_form_of_way",
]


# ─────────────────────────────────────────────────────────────────────────────
# Feature preparation
# ─────────────────────────────────────────────────────────────────────────────

def _safe_get(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index, dtype=float if not isinstance(default, str) else object)


def prepare_features(
    df: pd.DataFrame,
    *,
    categorical_levels: Optional[dict[str, list[str]]] = None,
    matched_only: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[str], dict[str, list[str]]]:
    """
    Build (X, y) for the magnitude regressor.

    Returns
    -------
    X : DataFrame (features)
    y : Series (log1p deviation_m)
    feature_columns : list[str]
    categorical_levels : dict[str, list[str]]
    """
    if matched_only and "matched" in df.columns:
        df = df[df["matched"].fillna(False).astype(bool)].copy()
    else:
        df = df.copy()

    # ── Target: log(1 + deviation_m) ────────────────────────────────────────
    if "deviation_m" not in df.columns:
        raise KeyError("prepare_features needs `deviation_m` column.")
    dev = df["deviation_m"].astype(float)
    y_log = np.log1p(dev).rename("_log_deviation_m")

    # Drop rows with NaN target (unmatched rows)
    valid = y_log.notna()
    df = df.loc[valid].copy()
    y_log = y_log.loc[valid].copy()

    # ── Numeric ─────────────────────────────────────────────────────────────
    X_num = pd.DataFrame(
        {c: _safe_get(df, c).astype(float) for c in _NUMERIC_FEATURES},
        index=df.index,
    )

    # ── Boolean ─────────────────────────────────────────────────────────────
    X_bool = pd.DataFrame(
        {c: _safe_get(df, c, default=False).fillna(False).astype(bool).astype(int)
         for c in _BOOLEAN_FEATURES},
        index=df.index,
    )

    # ── Categorical (one-hot) ───────────────────────────────────────────────
    if categorical_levels is None:
        categorical_levels = {}
        for c in _CATEGORICAL_FEATURES:
            if c in df.columns:
                lv = sorted(df[c].dropna().astype(str).unique().tolist())
                categorical_levels[c] = lv

    cat_frames = []
    for c, levels in categorical_levels.items():
        ser = _safe_get(df, c, default=None)
        for lvl in levels:
            cat_frames.append(
                pd.Series((ser.astype(str) == lvl).astype(int),
                          name=f"{c}__{lvl}", index=df.index)
            )
    X_cat = pd.concat(cat_frames, axis=1) if cat_frames else pd.DataFrame(index=df.index)

    X = pd.concat([X_num, X_bool, X_cat], axis=1).replace([np.inf, -np.inf], np.nan)
    feature_columns = list(X.columns)
    return X, y_log, feature_columns, categorical_levels


# ─────────────────────────────────────────────────────────────────────────────
# Train / predict
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class XGBModelBundle:
    """Single-target regressor (predicts log1p deviation_m) plus metadata."""
    model: object  # xgb.XGBRegressor
    feature_columns: list[str]
    categorical_levels: dict[str, list[str]]


_DEFAULT_PARAMS = dict(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.04,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=6,
    reg_lambda=1.0,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    feature_columns: list[str],
    categorical_levels: dict[str, list[str]],
    params: dict | None = None,
    eval_set: Optional[tuple[pd.DataFrame, pd.Series]] = None,
) -> XGBModelBundle:
    """Train a single XGBRegressor on log1p(deviation_m)."""
    import xgboost as xgb

    p = dict(_DEFAULT_PARAMS)
    if params:
        p.update(params)

    m = xgb.XGBRegressor(**p)
    kw = {}
    if eval_set is not None:
        X_val, y_val = eval_set
        kw["eval_set"] = [(X_val[feature_columns], y_val)]
        kw["verbose"] = False
    m.fit(X_train[feature_columns], y_train, **kw)

    return XGBModelBundle(
        model=m,
        feature_columns=feature_columns,
        categorical_levels=categorical_levels,
    )


def predict_deviation(bundle: XGBModelBundle, X_test: pd.DataFrame) -> np.ndarray:
    """
    Predict deviation magnitude in **metres** (after expm1 inverse transform).
    """
    log_pred = bundle.model.predict(X_test[bundle.feature_columns])
    return np.expm1(log_pred).clip(min=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid corrector: ML magnitude × rule-based direction
# ─────────────────────────────────────────────────────────────────────────────

def apply_model(
    bundle: XGBModelBundle,
    df: pd.DataFrame,
    *,
    alpha_floor: float = 0.0,
    alpha_ceil: float = 1.0,
) -> pd.DataFrame:
    """
    Run the magnitude model over a matched_corrected DataFrame and apply
    the hybrid corrector.

    For each row with a Stage-2 snap available:
        snap_vec_east  = corrected_glh_east  − glh_east
        snap_vec_north = corrected_glh_north − glh_north
        snap_dist      = corrected_snap_distance_m
        predicted_dev  = predict_deviation(...)
        alpha          = predicted_dev / (predicted_dev + snap_dist)
        pred_east      = glh_east  + alpha · snap_vec_east
        pred_north     = glh_north + alpha · snap_vec_north

    For rows where Stage 2 could not snap (corrected_snap_distance_m is
    NaN — typically points outside Edinburgh / outside the loaded
    network), there is no direction to scale into; we keep raw GLH and
    record alpha=0.

    Adds columns:
        predicted_deviation_m
        hybrid_alpha                 (∈ [alpha_floor, alpha_ceil])
        pred_glh_east, pred_glh_north
        pred_glh_lat,  pred_glh_lon
        pred_deviation_m             (distance pred → gpx truth)
        improvement_vs_raw_m         = deviation_m − pred_deviation_m
        improvement_vs_stage2_m      = corrected_deviation_m − pred_deviation_m
    """
    out = df.copy()
    if out.empty:
        for c in ("predicted_deviation_m", "hybrid_alpha",
                  "pred_glh_east", "pred_glh_north",
                  "pred_glh_lat", "pred_glh_lon", "pred_deviation_m",
                  "improvement_vs_raw_m", "improvement_vs_stage2_m"):
            out[c] = pd.Series(dtype=float)
        return out

    X, _, _, _ = prepare_features(out, categorical_levels=bundle.categorical_levels,
                                  matched_only=False)
    # Align columns (any missing categorical level → zero column)
    for c in bundle.feature_columns:
        if c not in X.columns:
            X[c] = 0.0
    X = X[bundle.feature_columns]

    pred_dev = np.full(len(out), np.nan, dtype=float)
    pred_dev_for_X = predict_deviation(bundle, X)
    pred_dev[X.index] = pred_dev_for_X
    out["predicted_deviation_m"] = pred_dev

    # ── Hybrid α: blend toward Stage-2 snapped position ─────────────────────
    has_snap = (out.get("corrected_glh_east", pd.Series(dtype=float)).notna()
                & out.get("corrected_glh_north", pd.Series(dtype=float)).notna()
                & out.get("corrected_snap_distance_m", pd.Series(dtype=float)).notna())
    snap_dist = out.get("corrected_snap_distance_m", pd.Series(np.nan, index=out.index)).astype(float)
    p = out["predicted_deviation_m"].astype(float)

    alpha = np.where(
        has_snap & p.notna(),
        p / (p + snap_dist),
        0.0,  # no direction available → keep raw
    )
    alpha = np.clip(alpha, alpha_floor, alpha_ceil)
    out["hybrid_alpha"] = alpha

    out["pred_glh_east"] = out["glh_east"].astype(float)
    out["pred_glh_north"] = out["glh_north"].astype(float)
    mask_snap = has_snap.to_numpy()
    if mask_snap.any():
        snap_east  = out["corrected_glh_east"].astype(float)
        snap_north = out["corrected_glh_north"].astype(float)
        out.loc[mask_snap, "pred_glh_east"] = (
            out.loc[mask_snap, "glh_east"]
            + alpha[mask_snap] * (snap_east.loc[mask_snap] - out.loc[mask_snap, "glh_east"])
        )
        out.loc[mask_snap, "pred_glh_north"] = (
            out.loc[mask_snap, "glh_north"]
            + alpha[mask_snap] * (snap_north.loc[mask_snap] - out.loc[mask_snap, "glh_north"])
        )

    # WGS84 for plotting
    out["pred_glh_lat"] = np.nan
    out["pred_glh_lon"] = np.nan
    m = out["pred_glh_east"].notna() & out["pred_glh_north"].notna()
    if m.any():
        lat, lon = bng_to_wgs84(out.loc[m, "pred_glh_east"], out.loc[m, "pred_glh_north"])
        out.loc[m, "pred_glh_lat"] = lat
        out.loc[m, "pred_glh_lon"] = lon

    # Final deviations
    if {"gpx_east", "gpx_north"}.issubset(out.columns):
        out["pred_deviation_m"] = bng_distance(
            out["pred_glh_east"], out["pred_glh_north"],
            out["gpx_east"], out["gpx_north"],
        )
        if "deviation_m" in out.columns:
            out["improvement_vs_raw_m"] = out["deviation_m"] - out["pred_deviation_m"]
        else:
            out["improvement_vs_raw_m"] = np.nan
        if "corrected_deviation_m" in out.columns:
            out["improvement_vs_stage2_m"] = out["corrected_deviation_m"] - out["pred_deviation_m"]
        else:
            out["improvement_vs_stage2_m"] = np.nan
    else:
        out["pred_deviation_m"] = np.nan
        out["improvement_vs_raw_m"] = np.nan
        out["improvement_vs_stage2_m"] = np.nan

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics & persistence
# ─────────────────────────────────────────────────────────────────────────────

def feature_importance(bundle: XGBModelBundle, *, top_n: int = 20) -> pd.DataFrame:
    """Return gain importance for the single regressor."""
    gain = bundle.model.get_booster().get_score(importance_type="gain")
    rows = [(f, gain.get(f, 0.0)) for f in bundle.feature_columns]
    fi = pd.DataFrame(rows, columns=["feature", "gain"]).sort_values(
        "gain", ascending=False).reset_index(drop=True)
    fi["rank"] = np.arange(1, len(fi) + 1)
    return fi.head(top_n) if top_n else fi


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIER ARM — predict P(snap_helps | features)
# ─────────────────────────────────────────────────────────────────────────────
#
# Alternative to the magnitude+hybrid approach. The classifier makes a hard
# decision per point: take the Stage-2 snapped position, or keep raw. This
# sidesteps the direction-averaging issue: if the snap is wrong half the time,
# a blended corrector averages out, but a classifier that learns *when* the
# snap helps can pick the right action per point.
#
# Target: y_clf = (corrected_deviation_m < deviation_m).astype(int)
#   = 1 if Stage 2's snap actually moved the point closer to truth
#   = 0 if it pushed the point further from truth
# Trained only on snap-applicable rows (corrected_deviation_m not NaN).
# At inference, applied only to snap-applicable rows; non-snap rows keep raw.


@dataclass
class XGBClassifierBundle:
    """Container for the trained classifier + metadata."""
    model: object  # xgb.XGBClassifier
    feature_columns: list[str]
    categorical_levels: dict[str, list[str]]
    pos_rate: float  # training-set rate of positive (snap-helped) examples


_DEFAULT_CLF_PARAMS = dict(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.04,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=6,
    reg_lambda=1.0,
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)


def _build_feature_table(
    df: pd.DataFrame,
    *,
    categorical_levels: Optional[dict[str, list[str]]] = None,
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """Shared X-table builder used by both the regressor and classifier."""
    X_num = pd.DataFrame(
        {c: _safe_get(df, c).astype(float) for c in _NUMERIC_FEATURES},
        index=df.index,
    )
    X_bool = pd.DataFrame(
        {c: _safe_get(df, c, default=False).fillna(False).astype(bool).astype(int)
         for c in _BOOLEAN_FEATURES},
        index=df.index,
    )

    if categorical_levels is None:
        categorical_levels = {}
        for c in _CATEGORICAL_FEATURES:
            if c in df.columns:
                lv = sorted(df[c].dropna().astype(str).unique().tolist())
                categorical_levels[c] = lv

    cat_frames = []
    for c, levels in categorical_levels.items():
        ser = _safe_get(df, c, default=None)
        for lvl in levels:
            cat_frames.append(
                pd.Series((ser.astype(str) == lvl).astype(int),
                          name=f"{c}__{lvl}", index=df.index)
            )
    X_cat = pd.concat(cat_frames, axis=1) if cat_frames else pd.DataFrame(index=df.index)

    X = pd.concat([X_num, X_bool, X_cat], axis=1).replace([np.inf, -np.inf], np.nan)
    return X, list(X.columns), categorical_levels


def prepare_classifier_features(
    df: pd.DataFrame,
    *,
    categorical_levels: Optional[dict[str, list[str]]] = None,
    snap_required: bool = True,
    matched_only: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[str], dict[str, list[str]]]:
    """
    Build (X, y_snap_helped) for the classifier.

    With `snap_required=True` (training-time default) keeps only rows where
    Stage 2 produced a snap. With `snap_required=False` (inference-time)
    keeps every row so we can score them — the apply step decides whether
    the snap is actually available.
    """
    if matched_only and "matched" in df.columns:
        df = df[df["matched"].fillna(False).astype(bool)].copy()
    else:
        df = df.copy()
    if snap_required:
        df = df[df["corrected_deviation_m"].notna()
                & df["deviation_m"].notna()].copy()

    if df.empty:
        X, fc, cl = _build_feature_table(df, categorical_levels=categorical_levels)
        return X, pd.Series(dtype=int), fc, cl

    if snap_required:
        y = (df["corrected_deviation_m"] < df["deviation_m"]).astype(int)
    else:
        y = pd.Series(np.nan, index=df.index)  # not used at inference
    y.name = "_snap_helped"

    X, feature_columns, cat_levels = _build_feature_table(
        df, categorical_levels=categorical_levels,
    )
    return X, y, feature_columns, cat_levels


def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    feature_columns: list[str],
    categorical_levels: dict[str, list[str]],
    params: dict | None = None,
    eval_set: Optional[tuple[pd.DataFrame, pd.Series]] = None,
) -> XGBClassifierBundle:
    """Train an XGBClassifier predicting whether the Stage-2 snap helps."""
    import xgboost as xgb

    p = dict(_DEFAULT_CLF_PARAMS)
    if params:
        p.update(params)

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_rate = n_pos / max(len(y_train), 1)
    if n_pos > 0 and n_neg > 0:
        p["scale_pos_weight"] = n_neg / n_pos

    m = xgb.XGBClassifier(**p)
    kw = {}
    if eval_set is not None:
        X_val, y_val = eval_set
        kw["eval_set"] = [(X_val[feature_columns], y_val)]
        kw["verbose"] = False
    m.fit(X_train[feature_columns], y_train, **kw)

    return XGBClassifierBundle(
        model=m,
        feature_columns=feature_columns,
        categorical_levels=categorical_levels,
        pos_rate=pos_rate,
    )


def predict_snap_probability(
    bundle: XGBClassifierBundle,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """Return P(snap_helps | features) for each row of X_test."""
    return bundle.model.predict_proba(X_test[bundle.feature_columns])[:, 1]


def apply_classifier_corrector(
    bundle: XGBClassifierBundle,
    df: pd.DataFrame,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Apply the classifier-based corrector.

    For each row that has a Stage-2 snap available AND for which the
    classifier's predicted probability of "snap helps" is at or above the
    threshold, replace the raw position with the snapped position. All
    other rows keep raw.

    Columns added:
        snap_helps_prob              float, [0,1]; NaN where classifier
                                      could not be scored (shouldn't happen
                                      in practice)
        snap_helps_decision          bool, True iff snap was chosen
        clf_glh_east, clf_glh_north  corrected positions in BNG
        clf_glh_lat, clf_glh_lon     same in WGS84
        clf_deviation_m              distance to GPX truth
        clf_improvement_vs_raw_m     = deviation_m − clf_deviation_m
        clf_improvement_vs_stage2_m  = corrected_deviation_m − clf_deviation_m
    """
    out = df.copy()
    needed = [
        "snap_helps_prob", "snap_helps_decision",
        "clf_glh_east", "clf_glh_north",
        "clf_glh_lat", "clf_glh_lon",
        "clf_deviation_m",
        "clf_improvement_vs_raw_m", "clf_improvement_vs_stage2_m",
    ]
    if out.empty:
        for c in needed:
            out[c] = pd.Series(dtype=float)
        return out

    # Build features for all rows (so we can score them).
    X, _, _, _ = prepare_classifier_features(
        out,
        categorical_levels=bundle.categorical_levels,
        snap_required=False,
        matched_only=False,
    )
    for c in bundle.feature_columns:
        if c not in X.columns:
            X[c] = 0.0
    X = X[bundle.feature_columns]

    probs = np.full(len(out), np.nan, dtype=float)
    if len(X):
        probs[X.index] = predict_snap_probability(bundle, X)
    out["snap_helps_prob"] = probs

    # Need both: a Stage-2 snap exists AND classifier says it helps.
    has_snap = (
        out.get("corrected_glh_east", pd.Series(dtype=float)).notna()
        & out.get("corrected_glh_north", pd.Series(dtype=float)).notna()
        & out.get("corrected_snap_distance_m", pd.Series(dtype=float)).notna()
    )
    decision = has_snap & (out["snap_helps_prob"] >= threshold)
    out["snap_helps_decision"] = decision.fillna(False).astype(bool)

    # Start at raw, swap to snap where decision is True.
    out["clf_glh_east"] = out["glh_east"].astype(float)
    out["clf_glh_north"] = out["glh_north"].astype(float)
    out.loc[decision, "clf_glh_east"] = out.loc[decision, "corrected_glh_east"]
    out.loc[decision, "clf_glh_north"] = out.loc[decision, "corrected_glh_north"]

    out["clf_glh_lat"] = np.nan
    out["clf_glh_lon"] = np.nan
    m = out["clf_glh_east"].notna() & out["clf_glh_north"].notna()
    if m.any():
        lat, lon = bng_to_wgs84(out.loc[m, "clf_glh_east"], out.loc[m, "clf_glh_north"])
        out.loc[m, "clf_glh_lat"] = lat
        out.loc[m, "clf_glh_lon"] = lon

    if {"gpx_east", "gpx_north"}.issubset(out.columns):
        out["clf_deviation_m"] = bng_distance(
            out["clf_glh_east"], out["clf_glh_north"],
            out["gpx_east"], out["gpx_north"],
        )
        if "deviation_m" in out.columns:
            out["clf_improvement_vs_raw_m"] = out["deviation_m"] - out["clf_deviation_m"]
        else:
            out["clf_improvement_vs_raw_m"] = np.nan
        if "corrected_deviation_m" in out.columns:
            out["clf_improvement_vs_stage2_m"] = out["corrected_deviation_m"] - out["clf_deviation_m"]
        else:
            out["clf_improvement_vs_stage2_m"] = np.nan
    else:
        out["clf_deviation_m"] = np.nan
        out["clf_improvement_vs_raw_m"] = np.nan
        out["clf_improvement_vs_stage2_m"] = np.nan

    return out


def feature_importance_classifier(
    bundle: XGBClassifierBundle, *, top_n: int = 20,
) -> pd.DataFrame:
    gain = bundle.model.get_booster().get_score(importance_type="gain")
    rows = [(f, gain.get(f, 0.0)) for f in bundle.feature_columns]
    fi = pd.DataFrame(rows, columns=["feature", "gain"]).sort_values(
        "gain", ascending=False).reset_index(drop=True)
    fi["rank"] = np.arange(1, len(fi) + 1)
    return fi.head(top_n) if top_n else fi


def save_classifier_bundle(bundle: XGBClassifierBundle, prefix: str) -> dict[str, str]:
    import json, os
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    model_path = prefix + "_classifier_model.json"
    meta_path = prefix + "_classifier_meta.json"
    bundle.model.save_model(model_path)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({
            "feature_columns": bundle.feature_columns,
            "categorical_levels": bundle.categorical_levels,
            "pos_rate": bundle.pos_rate,
        }, fh, indent=2)
    return {"model": model_path, "meta": meta_path}


def load_classifier_bundle(prefix: str) -> XGBClassifierBundle:
    import json
    import xgboost as xgb
    m = xgb.XGBClassifier()
    m.load_model(prefix + "_classifier_model.json")
    with open(prefix + "_classifier_meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    return XGBClassifierBundle(
        model=m,
        feature_columns=meta["feature_columns"],
        categorical_levels=meta["categorical_levels"],
        pos_rate=meta.get("pos_rate", float("nan")),
    )


def save_bundle(bundle: XGBModelBundle, prefix: str) -> dict[str, str]:
    """Save model and metadata to `<prefix>_model.json` + `<prefix>_meta.json`."""
    import json, os
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    model_path = prefix + "_model.json"
    meta_path = prefix + "_meta.json"
    bundle.model.save_model(model_path)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({
            "feature_columns": bundle.feature_columns,
            "categorical_levels": bundle.categorical_levels,
        }, fh, indent=2)
    return {"model": model_path, "meta": meta_path}


def load_bundle(prefix: str) -> XGBModelBundle:
    import json
    import xgboost as xgb
    m = xgb.XGBRegressor()
    m.load_model(prefix + "_model.json")
    with open(prefix + "_meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    return XGBModelBundle(
        model=m,
        feature_columns=meta["feature_columns"],
        categorical_levels=meta["categorical_levels"],
    )
