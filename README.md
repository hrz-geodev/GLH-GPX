# GLH–GPX Alignment Pipeline

This repository contains a unified, reproducible pipeline for aligning Google Location History (GLH) data with high-accuracy GPX tracks, producing point-level matched datasets suitable for:

- Spatial accuracy analysis
- Machine-learning correction models
- GIS visualisation (ArcGIS / QGIS)

The pipeline supports multiple GLH export formats, handles GPS cleaning and interpolation, performs automated quality verification and filtering, and generates quality-controlled training samples.

---

# 1. What this pipeline does (end-to-end)

For each volunteer:

## 1. Parse GLH data

Supports:

- Legacy list-based GLH exports
- Newer `semanticSegments` Timeline exports

Normalises all formats into a canonical point table.

---

## 2. Parse GPX data

- Reads all GPX files in a volunteer folder
- Extracts timestamped GPS points (UTC)

---

## 3. Clean GPS trajectories

- Removes duplicate points
- Removes implausible jumps (speed threshold)
- Splits GPS into segments by large time gaps

---

## 4. Time-align GPS to GLH

Interpolates GPS positions at exact GLH timestamps.

Only accepts interpolation when:

- GPS points bracket the GLH timestamp
- Both points belong to the same GPS segment
- Bracket gap ≤ configurable threshold (default 120 s)

---

## 5. Quality control & structuring

Assigns:

- `segment_id` (GLH segment)
- `journey_id` (groups segments by temporal continuity)
- `match_quality_tier` (A–D based on interpolation gap)

Generates:

- spatial error statistics
- segment-level diagnostics
- journey-level diagnostics
- quality reports

---

## 6. Automated segment filtering

Low-quality trajectory segments are automatically identified and removed based on spatial error metrics.

Default filtering rule:

```text
remove segment if:
  median_error_m > 100
  OR
  more than 30% of points have error_m > 200
```

This produces cleaner training-ready datasets for downstream modelling.

---

## 7. Export GIS-ready outputs

Exports:

- GLH matched points (GPX)
- GPS matched points (GPX)
- GLH–GPS connection lines (GeoJSON)
- Attribute join table (CSV)

---

# 2. Supported GLH formats

The pipeline automatically detects the GLH format.

---

## Legacy format

Top-level JSON list.

Uses fields such as:

- `activity.start / end`
- `visit.placeLocation`
- `timelinePath.durationMinutesOffsetFromStartTime`

---

## New Timeline format

JSON object with `semanticSegments`.

Uses:

- `semanticSegments.timelinePath.time`

Supports richer metadata when available.

---

Both formats are converted into the same canonical schema so they can be combined within one unified training dataset.

---

# 3. Repository structure

```text
GLH_GPX/
│
├── raw_data/
│   ├── GPX_Volunteer1/
│   │   ├── Timeline.json
│   │   └── *.gpx
│
├── interim/
│   └── <volunteer_id>/
│       ├── glh_points.csv
│       ├── gps_points_clean.csv
│       ├── gps_at_glh_timestamps.csv
│       ├── gps_at_glh_timestamps_with_tiers.csv
│       ├── glh_timeline_segments_with_journeys.csv
│       ├── qc_match.txt
│       ├── qc_match_detailed.txt
│       ├── debug_Vxxx_with_error.csv
│       ├── debug_Vxxx_segment_error_profile.csv
│       ├── debug_Vxxx_journey_error_profile.csv
│       ├── segment_filter_profile.csv
│       ├── bad_segments_to_remove.csv
│       ├── gps_at_glh_timestamps_with_tiers_segment_filtered.csv
│       ├── segment_filter_report.txt
│       ├── exports_glh_matched_points.gpx
│       ├── exports_gps_matched_points.gpx
│       ├── exports_matched_pairs_lines.geojson
│       └── exports_matched_pairs_join_table.csv
│
├── training_data/
│
└── src/
    ├── io_glh_unified.py
    ├── io_gpx_points.py
    ├── run_volunteer_pipeline.py
    ├── run_volunteer_post_qc.py
    ├── export_pairs_to_two_gpx_and_lines.py
    ├── batch_run_all_volunteers_anonymized.py
    ├── run_accuracy_verification_all.py
    ├── debug_volunteer_error_profile.py
    ├── filter_bad_segments_all.py
    ├── run_quality_filter_selected.py
    └── run_quality_filter_problem_cases.py
```

---

# 4. Core scripts (what each file does)

---

## io_glh_unified.py

Unified GLH parser.

Supports both:

- legacy exports
- `semanticSegments` exports

Outputs a canonical GLH point table with optional metadata fields.

---

## io_gpx_points.py

GPX parser.

- extracts timestamped GPS points in UTC
- supports multiple GPX naming structures
- no external GPX dependency required

---

## run_volunteer_pipeline.py

Main volunteer processing script.

For one volunteer:

- loads GLH + GPX files
- cleans GPS trajectories
- interpolates GPS to GLH timestamps
- filters to GPS coverage window

Writes:

- `glh_points.csv`
- `gps_points_clean.csv`
- `gps_at_glh_timestamps.csv`
- `qc_match.txt`

Run:

```bash
python src/run_volunteer_pipeline.py
```

---

## run_volunteer_post_qc.py

Post-processing and quality-control stage.

- derives `journey_id`
- assigns `match_quality_tier`
- generates detailed QC statistics

Writes:

- `gps_at_glh_timestamps_with_tiers.csv`
- `glh_timeline_segments_with_journeys.csv`
- `qc_match_detailed.txt`

Run:

```bash
python src/run_volunteer_post_qc.py
```

---

## export_pairs_to_two_gpx_and_lines.py

GIS export utility.

Exports valid matched points (`gps_interp_ok == TRUE`) to:

- `exports_glh_matched_points.gpx`
- `exports_gps_matched_points.gpx`
- `exports_matched_pairs_lines.geojson`
- `exports_matched_pairs_join_table.csv`

Designed for ArcGIS / QGIS visualisation.

Run:

```bash
python src/export_pairs_to_two_gpx_and_lines.py
```

---

## batch_run_all_volunteers_anonymized.py

Runs the full pipeline for all volunteer folders.

- supports anonymised batch processing
- generates standardised outputs
- handles varying file naming structures

Run:

```bash
python src/batch_run_all_volunteers_anonymized.py
```

---

## run_accuracy_verification_all.py

Runs overall spatial accuracy verification across volunteers.

Calculates:

- mean error
- median error
- percentile statistics
- quality summaries

Run:

```bash
python src/run_accuracy_verification_all.py
```

---

## debug_volunteer_error_profile.py

Detailed volunteer-level diagnostic analysis.

Generates:

- exact timestamp match analysis
- high-error diagnostics
- segment-level error profiles
- journey-level error profiles

Run:

```bash
python src/debug_volunteer_error_profile.py --anon_id V004
```

---

## filter_bad_segments_all.py

Automatically removes low-quality trajectory segments.

Default rule:

```text
median_error_m > 100
OR
more than 30% of points have error_m > 200
```

Outputs:

- cleaned matched datasets
- segment filtering reports
- segment removal summaries

Run:

```bash
python src/filter_bad_segments_all.py
```

Single volunteer:

```bash
python src/filter_bad_segments_all.py --anon_id V004
```

---

## run_quality_filter_selected.py

Runs diagnostic + filtering workflow for standard volunteers.

Processes:

```text
V002
V003
V005
V006
V008
```

---

## run_quality_filter_problem_cases.py

Runs diagnostic + filtering workflow for volunteers requiring additional debugging.

Processes:

```text
V001
V004
V007
```

---

# 5. Output data explanation (important)

---

## gps_at_glh_timestamps_with_tiers.csv

Each row represents one GLH `timelinePath` point, with:

- original GLH location (`lat`, `lon`)
- interpolated GPS location (`gps_lat_interp`, `gps_lon_interp`)
- `gps_interp_ok`
- `bracket_gap_s`
- `match_quality_tier`
- `segment_id`
- `journey_id`

---

## match_quality_tier

```text
A: ≤10 s
B: 10–30 s
C: 30–60 s
D: 60–120 s
```

Smaller interpolation gaps indicate stronger temporal alignment confidence.

---

## gps_at_glh_timestamps_with_tiers_segment_filtered.csv

Final cleaned training-source dataset.

Contains:

- valid matched GLH–GPS pairs
- filtered trajectory segments
- quality metadata
- segment/journey structure

This is the recommended dataset for machine-learning model training.

---

# 6. ArcGIS / QGIS usage

Recommended workflow:

Load:

```text
exports_glh_matched_points.gpx
exports_gps_matched_points.gpx
```

Then load:

```text
exports_matched_pairs_lines.geojson
```

Recommended symbology:

- `error_m`
- `match_quality_tier`

Optional attribute join:

```text
exports_matched_pairs_join_table.csv
```

using:

```text
pair_id
```

This provides a clear spatial visualisation of GLH vs GPS discrepancies.

---

# 7. Training dataset preparation

The filtered outputs can be converted into machine-learning training datasets.

Recommended targets:

```text
target_delta_lat = gps_lat_interp - lat
target_delta_lon = gps_lon_interp - lon
```

Potential input features include:

- GLH coordinates
- activity type
- segment probability
- interpolation quality
- temporal context
- road-network features

---

# 8. Road network integration

Road network data can optionally be integrated for:

- nearest-road distance calculation
- contextual feature engineering
- trajectory plausibility checks
- future map-matching experiments

Recommended usage is during feature engineering rather than initial matching.

---

# 9. Privacy and anonymisation

The workflow uses anonymised volunteer IDs:

```text
V001
V002
V003
...
```

Raw participant identifiers and original location data should remain local and must not be committed to GitHub.

---

# 10. Recommended .gitignore

```gitignore
raw_data/
interim/
training_data/

*.csv
*.json
*.gpx
*.geojson
*.txt

__pycache__/
*.pyc
.venv/
venv/
```

---

# 11. Current status

The pipeline currently supports:

- multiple GLH formats
- anonymised batch volunteer processing
- GPS cleaning and interpolation
- spatial accuracy verification
- automated segment filtering
- GIS export
- generation of training-ready matched datasets

The next planned stage is machine-learning model development for GLH spatial correction and trajectory refinement.

---

# 12. Citation / acknowledgement

If using this workflow in research or publications, please cite or acknowledge the repository appropriately.

