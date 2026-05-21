# Stage 3.1 — XGBoost Magnitude Indicator (Model Card)

## What this model does

Given a single GLH point with associated map-context features, predicts the
**magnitude of the GLH error** (in metres, on the original scale) against a
hypothetical high-precision GPS reference. The output is a per-point
uncertainty estimate. **The model does not move the point** — it only
flags how trustworthy the raw GLH coordinate is likely to be.

Internally the regressor is trained on `log1p(deviation_m)` and the
prediction is exponentiated back to metres at inference time.

## Files

Eight Leave-One-Out cross-validation folds are shipped:

```
stage3_1_indicator_fold{1..8}_model.json   # XGBoost saved model
stage3_1_indicator_fold{1..8}_meta.json    # feature schema + categorical levels
```

Each fold was trained on the data of the seven other held-out subsets.
For external use you may either:

- pick any single fold (folds 2–8 were trained on broadly comparable
  urban / suburban Edinburgh-area data; fold 1 is the geographic-transfer
  fold and is the closest analogue for non-Edinburgh / cross-region data),
  or
- average predictions across all 8 folds for a more robust estimate.

## How to load

```python
from glh_correct.model_xgboost import load_bundle, predict_deviation

bundle = load_bundle("models/stage3_1_indicator/stage3_1_indicator_fold1")
# bundle now has bundle.model + bundle.feature_columns + bundle.categorical_levels

predicted_dev_m = predict_deviation(bundle, X_test_df)
```

`X_test_df` must contain the columns listed in `bundle.feature_columns` —
the simplest way to produce them is to run
`glh_correct.feature_engineering.add_map_context_features` over your matched
GLH frame.

## Features expected

Numeric features (a subset of these — see `bundle.feature_columns` for the
exact list per fold):

```
glh_speed_mps, glh_accuracy_m,
dist_to_nearest_network_m, corrected_snap_distance_m,
nearest_car_distance_m, nearest_ped_distance_m,
bearing_to_nearest_car_sin, bearing_to_nearest_car_cos,
bearing_to_nearest_ped_sin, bearing_to_nearest_ped_cos,
nearest_building_m, n_buildings_50m, building_area_50m_m2,
n_buildings_100m, building_area_100m_m2
```

Boolean features:

```
glh_inside_building
```

Categorical features (one-hot expanded internally; see meta.json):

```
glh_layer (raw_signals | timeline_paths)
glh_source (GPS | WIFI | UNKNOWN | …)
corrected_glh_network_kind (carriageway | pedestrian | …)
```

## Calibration

Held-out Pearson correlation between `log1p(predicted_dev)` and
`log1p(actual_dev)`:

| subset                          | Pearson(log) | n      |
|---------------------------------|--------------|--------|
| aggregate (all folds)           | 0.687        | 14,002 |
| Edinburgh-area subset only      | 0.744        |  5,755 |

The Edinburgh-area subset excludes the geographic-transfer fold whose
map-context features are NaN where map coverage is missing — that fold
limits its own per-fold calibration.

## Known limitations

1. **Magnitude only, no direction.** The model predicts how *wrong* the
   point is, not which way to move it. Pair with the Stage 2 rule-based
   snapper or the Stage 5 HMM to obtain corrected positions.
2. **Building footprints only.** No building heights, so urban-canyon
   error patterns are only partially captured.
3. **Edinburgh-trained.** Best calibration is in dense urban UK; expect
   degradation in radically different settings (motorway-only routes,
   rural areas without OSM building data, non-UK).
4. **LOVO honesty.** Each fold has only ~13 k training rows. With more
   data calibration would likely improve.
