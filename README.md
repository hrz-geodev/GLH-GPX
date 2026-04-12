# GLH–GPX Alignment Pipeline

This repository contains a unified, reproducible pipeline for aligning Google Location History (GLH) data with high-accuracy GPX tracks, producing point-level matched datasets suitable for:

* Spatial accuracy analysis
* Machine-learning correction models
* GIS visualisation (ArcGIS / QGIS)

The pipeline supports multiple GLH export formats, handles GPS cleaning and interpolation, and generates quality-controlled training samples.

---

## 1. What this pipeline does (end-to-end)

For each volunteer:

### 1. Parse GLH data
* Supports:
  * Legacy list-based GLH exports
  * Newer semanticSegments Timeline exports
* Normalises all formats into a canonical point table

### 2. Parse GPX data
* Reads all GPX files in a volunteer folder
* Extracts timestamped GPS points (UTC)

### 3. Clean GPS trajectories
* Removes duplicate points
* Removes implausible jumps (speed threshold)
* Splits GPS into segments by large time gaps

### 4. Time-align GPS to GLH
* Interpolates GPS positions at exact GLH timestamps
* Only accepts interpolation when:
  * GPS points bracket the GLH timestamp
  * Both points belong to the same GPS segment
  * Bracket gap ≤ configurable threshold (default 120 s)

### 5. Quality control & structuring
* Assigns:
  * `segment_id` (GLH segment)
  * `journey_id` (groups segments by temporal continuity)
  * `match_quality_tier` (A–D based on interpolation gap)
* Generates coverage and quality reports

### 6. Export GIS-ready outputs
* GLH matched points (GPX)
* GPS matched points (GPX)
* GLH–GPS connection lines (GeoJSON)
* Attribute join table (CSV)

---

## 2. Supported GLH formats

The pipeline automatically detects the GLH format:

### Legacy format
* Top-level JSON list
* Uses:
  * `activity.start` / `end`
  * `visit.placeLocation`
  * `timelinePath.durationMinutesOffsetFromStartTime`

### New Timeline format
* JSON object with `semanticSegments`
* Uses:
  * `semanticSegments.timelinePath.time`
  * Richer metadata when available

Both formats are converted into the same canonical schema, so they can be used together in one training dataset.

---

## 3. Repository structure

```
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
│       ├── exports_glh_matched_points.gpx
│       ├── exports_gps_matched_points.gpx
│       ├── exports_matched_pairs_lines.geojson
│       └── exports_matched_pairs_join_table.csv
│
└── src/
    ├── io_glh_unified.py
    ├── io_gpx_points.py
    ├── run_volunteer_pipeline.py
    ├── run_volunteer_post_qc.py
    ├── export_pairs_to_two_gpx_and_lines.py
    └── batch_run_all_volunteers_anonymized.py
```

---

## 4. Core scripts (what each file does)

### `io_glh_unified.py`
* Unified GLH parser
* Supports both legacy and semanticSegments formats
* Outputs canonical GLH point table with optional extra fields

### `io_gpx_points.py`
* GPX parser (no external dependencies)
* Extracts timestamped GPS points in UTC

### `run_volunteer_pipeline.py`
**Main processing script**

For one volunteer:
* Loads GLH + all GPX files
* Cleans GPS trajectories
* Interpolates GPS to GLH timestamps
* Filters to GPS coverage window
* Writes:
  * `glh_points.csv`
  * `gps_points_clean.csv`
  * `gps_at_glh_timestamps.csv`
  * `qc_match.txt`

**Run:**
```bash
python src/run_volunteer_pipeline.py
```

### `run_volunteer_post_qc.py`
**Post-processing & quality control**

* Derives `journey_id` from segment time gaps
* Assigns `match_quality_tier`
* Generates detailed QC statistics
* Writes:
  * `gps_at_glh_timestamps_with_tiers.csv`
  * `glh_timeline_segments_with_journeys.csv`
  * `qc_match_detailed.txt`

**Run:**
```bash
python src/run_volunteer_post_qc.py
```

### `export_pairs_to_two_gpx_and_lines.py`
**GIS export**

Exports only valid matched points (`gps_interp_ok == TRUE`) to:
* `exports_glh_matched_points.gpx`
* `exports_gps_matched_points.gpx`
* `exports_matched_pairs_lines.geojson`
* `exports_matched_pairs_join_table.csv`

Designed for ArcGIS / QGIS visualisation.

**Run:**
```bash
python src/export_pairs_to_two_gpx_and_lines.py
```

### `batch_run_all_volunteers_anonymized.py`
* Runs the full pipeline for all volunteer folders
* Ensures consistent outputs
* Designed for anonymised batch processing

---

## 5. Output data explanation (important)

### `gps_at_glh_timestamps_with_tiers.csv`
Each row represents a GLH timelinePath point, with:

* Original GLH location (`lat`, `lon`)
* Interpolated GPS location (`gps_lat_interp`, `gps_lon_interp`)
* `gps_interp_ok` flag
* `bracket_gap_s` (time gap used for interpolation)
* `match_quality_tier`:
  * **A**: ≤10 s
  * **B**: 10–30 s
  * **C**: 30–60 s
  * **D**: 60–120 s
* `segment_id` and `journey_id`

**Only rows with `gps_interp_ok == TRUE` are valid GPS matches.**

---

## 6. ArcGIS / QGIS usage

### Recommended workflow:

1. **Load:**
   * `exports_glh_matched_points.gpx`
   * `exports_gps_matched_points.gpx`

2. **Load:**
   * `exports_matched_pairs_lines.geojson`

3. **Symbolise lines by:**
   * `error_m`
   * or `match_quality_tier`

4. **Join attributes using:**
   * `exports_matched_pairs_join_table.csv` on `pair_id`

This gives a clear spatial view of GLH vs GPS discrepancies.

---

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Contact

[Add contact information here]
