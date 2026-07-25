# Data Availability

The datasets are distributed separately from the source code through Zenodo:

- DOI: [10.5281/zenodo.20736587](https://doi.org/10.5281/zenodo.20736587)

## Minimal Zenodo release

The current Zenodo record contains exactly three CSV files. After downloading
the archive, keep the directory name and place it directly under the repository
`data/` directory:

```text
data/
└── basalt_geochemistry_dataset/
    ├── modern_basalt_geochemistry.csv
    ├── archean_basalt_geochemistry.csv
    └── archean_basalt_geodan_predictions.csv
```

No file renaming or copying into `00_raw/`, `03_combined/`, or
`archean/outputs/` is required. The path configuration automatically reads the
three Zenodo files from `data/basalt_geochemistry_dataset/`.

| File | Expected shape | Used for |
|---|---:|---|
| `modern_basalt_geochemistry.csv` | 30,547 × 43 | Modern training/test split and all downstream training preprocessing |
| `archean_basalt_geochemistry.csv` | 3,483 × 57 | Archean candidate pool with original missing values retained |
| `archean_basalt_geodan_predictions.csv` | 3,012 × 87 | Published GeoDAN Archean prediction table used by downstream figures |

The modern workflow regenerates these intermediate files locally:

```text
data/04_split/
data/05_imputed/
data/06_normalized/
```

They are not part of the Zenodo release and are not required before running
`01_preprocessing/split_train_test.py`.

## Quick start from the three Zenodo files

From the repository root, activate the verified Python environment and run:

```bash
conda activate babeldoc
python 01_preprocessing/split_train_test.py
python 02_imputation/imputation_train_predict.py
python 03_normalization/normalize_major_elements.py
python 03_normalization/selective_smote.py
python 03_normalization/normalize.py
```

The default split script uses `modern_basalt_geochemistry.csv` directly. If a
locally rebuilt GEOROC + PetDB table is available and should be used instead,
set:

```bash
BASALT_USE_COMBINED=1 python 01_preprocessing/split_train_test.py
```

The imputation script reuses cached outputs only after checking that their
metadata and labels match the current split. To force a complete MissForest
refit, set:

```bash
BASALT_REUSE_IMPUTATION_CACHE=0 python 02_imputation/imputation_train_predict.py
```

## Model weights

Model weights are not included in the code repository or the minimal Zenodo
release. Train the model locally with:

```bash
python 04_model/ablation_v4_vit_transformer.py
```

A one-epoch smoke test can be run without overwriting the normal model output:

```bash
GEODAN_EPOCHS=1 GEODAN_SEEDS=42 \
GEODAN_OUTPUT_DIR=data/models/smoke_test \
python 04_model/ablation_v4_vit_transformer.py
```

After training, the main model weight is expected at:

```text
data/models/Full_Model_(ViT+Transformer)_best_seed.pth
```

## Optional historical preprocessing

The original GEOROC/PetDB raw tables and the external convergent-margin
reclassification files are not part of the minimal release. Therefore the
following scripts are optional historical reconstruction tools and require
additional upstream inputs:

```text
01_preprocessing/filter/extract_georoc.py
01_preprocessing/filter/extract_petdb.py
01_preprocessing/combine_list.py
06_archean_application/extended_archean_pool_analysis.py
```

They are not prerequisites for reproducing the modern training workflow from
the three Zenodo files.

## Feature and label definitions

The model uses 36 geochemical features:

```text
NA2O MGO AL2O3 SIO2 P2O5 K2O CAO TIO2 MNO FEOT
RB V CR CO NI BA SR Y ZR NB LA CE PR ND SM EU GD TB DY HO ER YB LU HF TA TH
```

The nine tectonic-setting labels are:

```text
SPREADING_CENTER
OCEAN_ISLAND
CONTINENTAL_RIFT
OCEANIC_PLATEAU
CONTINENTAL_FLOOD_BASALT
BACK-ARC_BASIN
Island_arc
Continental_arc
Intra-oceanic_arc
```
