# glh_correct — characterise and correct Google Location History error

`glh_correct` is the inference-time release of a research pipeline that
characterises Google Location History (GLH) positional error against
high-precision GPS reference data and applies trained correctors to new
GLH exports.

It accompanies our paper (in preparation) and is intended to let other
researchers and practitioners:

1. Quantify how trustworthy a given GLH point is (per-point uncertainty
   estimate, in metres) via a trained **XGBoost magnitude indicator**.
2. Snap GLH points to the nearest network edge within a configurable
   radius via a **rule-based corrector** (Stage 2).
3. Map-match a whole GLH session to a road / footway network via a
   **Newson-Krumm-style HMM** (Stage 5).

The pipeline was developed on 8 paired tracks with simultaneous GLH +
sub-metre GPX reference, collected in Edinburgh and across five UK
control regions (Manchester, Glasgow, London, Lancashire, Cumbria).

---

## What's in the box

```
github_release/
├── README.md                        # this file
├── requirements.txt                 # Python dependencies (pinned floors)
├── .gitignore                       # standard Python ignores
├── glh_correct/                     # importable Python package
│   ├── __init__.py
│   ├── glh_parser.py                # parse Google Location History exports
│   ├── gpx_parser.py                # parse high-precision GPX reference tracks
│   ├── projection.py                # WGS84 ↔ British National Grid
│   ├── cleaning.py                  # QC filters
│   ├── sessionize.py                # sessionise GLH and GPX
│   ├── matching.py                  # interpolate GPX truth at GLH times
│   ├── networks.py                  # load OS MasterMap / OSM road networks
│   ├── buildings.py                 # load OSM building footprints
│   ├── snapping.py                  # vectorised nearest-edge snap
│   ├── feature_engineering.py       # per-point map-context features
│   ├── correction_rule_based.py     # Stage 2 rule-based corrector
│   ├── model_xgboost.py             # Stage 3 XGBoost indicator / classifier
│   └── model_hmm_mapmatch.py        # Stage 5 HMM map-matcher
├── examples/
│   ├── 01_load_indicator.py         # score new data with the indicator
│   ├── 02_run_hmm_matching.py       # map-match a session with the HMM
│   └── 03_full_pipeline.py          # parse GLH → match GPX → predict → correct
└── docs/
    └──  METHODOLOGY.md               # what each stage does and why
```

---

## Installation

```bash
git clone <your-repo-url>
cd glh_correct_release
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then either install the package locally:

```bash
pip install -e .
```

…or add the release directory to `PYTHONPATH` and `import glh_correct`
directly.

> **GDAL note.** `geopandas` / `fiona` depend on GDAL. On Windows, install
> with `conda install -c conda-forge geopandas fiona` or grab the wheels
> from Gohlke if pip fails.

---

## Quick start

### Score new GLH data with the magnitude indicator

```python
import pandas as pd
from glh_correct.model_xgboost import load_bundle, predict_deviation
from glh_correct.feature_engineering import add_map_context_features
from glh_correct.networks import load_full_network
from glh_correct.buildings import load_buildings_bng

# 1. Parse your GLH export and project to British National Grid.
#    See examples/03_full_pipeline.py for the full pipeline.
glh_df = ...    # pandas frame with glh_east, glh_north, glh_layer, glh_source, ...

# 2. Load the road + building context (point project_root at the folder
#    holding map/ — see docs/DATA_SOURCES.md for what to put there).
networks  = load_full_network(project_root="/path/to/map_root")
buildings = load_buildings_bng(project_root="/path/to/map_root")

# 3. Annotate with per-point map-context features.
glh_df = add_map_context_features(glh_df, networks, buildings)

# 4. Load any indicator fold (here fold1, the geographic-transfer fold).
bundle = load_bundle("models/stage3_1_indicator/stage3_1_indicator_fold1")

# 5. Predict per-point uncertainty (in metres).
glh_df["predicted_deviation_m"] = predict_deviation(bundle, glh_df)
```

### Map-match a session with the HMM

```python
from glh_correct.model_hmm_mapmatch import map_match_session

corrected_df = map_match_session(
    session_df,            # one session of (east, north, time) rows
    networks,              # combined carriageway + pedestrian network
    max_radius_m=100.0,
    k=5,
    sigma_z=10.0,
    beta=50.0,
)
```

## What this release is *not*

- It is not a training pipeline. The training scripts (`train_stage3*.py`,
  `train_stage4*.py`, `train_stage5_hmm.py`) live in the parent research
  repo and are not shipped here — most external users will only need the
  trained indicator and the HMM (which has no trained parameters).
- It does not bundle OS MasterMap or OSM map data. OS MasterMap is
  licence-restricted and the OSM extracts are large. See
  `docs/DATA_SOURCES.md` for the exact files we used and where to get
  them.
- It does not ship the raw matched parquet files or any reference GPS
  data.

---
