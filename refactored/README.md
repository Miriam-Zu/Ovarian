# Refactored Pipeline

This folder contains a full engineering rewrite of the published analysis pipeline. The scientific logic is identical to the code in [`../published/`](../published/) — the same models, the same feature selection strategy, the same datasets. What changed is the code structure.

This rewrite was done after publication as a software engineering exercise: taking working research code and bringing it to a standard suitable for a production or collaborative environment.

---

## What Changed and Why

### 1. R pipeline decomposed into modules

The original `immunarch_analysis.Rmd` and `feature_filtering.Rmd` were large notebooks mixing data loading, transformation, plotting, statistics, and feature engineering in a linear structure.

The R code is now split into four reusable modules under `R/`, each with a single responsibility:

- `R/data_processing.R` — loading, subsampling, and preparing TCR repertoire data
- `R/plots.R` — all visualisation functions (clonality, diversity, overlap)
- `R/statistics.R` — statistical tests (t-tests, summaries)
- `R/feature_filtering.R` — feature matrix construction and both filtering methods

Two clean execution scripts in `scripts/` orchestrate these modules:

- `scripts/run_eda.R` — runs the full EDA pipeline (subsampling, plots, statistics)
- `scripts/run_feature_filtering.R` — runs feature engineering (method 1, method 2, or both)

Both scripts are CLI-driven, accepting input/output paths as arguments, mirroring the Python scripts.

```bash
Rscript scripts/run_eda.R /path/to/A_B/folder outputs/eda

Rscript scripts/run_feature_filtering.R outputs/eda outputs/features data/db both
```

### 2. Shared Python utility module (`python/ovarian_utils.py`)

The two original notebooks shared ~80% of the same code: metric computation, confusion matrix plotting, ROC curve generation, and error analysis.

All shared logic now lives in `ovarian_utils.py` and is imported by both pipelines. The functions are documented, typed, and independently testable.

### 3. Evaluation Protocol

The original notebooks computed accuracy, F1, precision, recall, and AUC on `atom.X` and `atom.y` — the **full dataset including training samples**. This is a standard approach in publications. In the refactored code, evaluation is performed exclusively on `atom.X_test` / `atom.y_test`, and the utility functions are designed to make this test-set evaluation the default, aligning with common  machine learning best practices.

### 4. Pipeline orchestrator (`scripts/run_pipeline.py`)

The original pipeline required manually running four scripts in the correct order with no error handling between steps. `run_pipeline.py` encodes the dependency graph, streams output to a timestamped log file, skips stages whose outputs already exist, and halts immediately if any stage fails.

```bash
python scripts/run_pipeline.py                # run all stages, skip completed
python scripts/run_pipeline.py --from-stage 3 # re-run ML stages only
python scripts/run_pipeline.py --force        # force full re-run
python scripts/run_pipeline.py --dry-run      # preview without executing
python scripts/run_pipeline.py --only 2       # run a single stage
```

### 5. Inference script (`scripts/predict.py`)

The original pipeline did not include a dedicated way to generate predictions for new patients. The `predict.py` script completes the end-to-end workflow: given a new TCR repertoire, it applies the same preprocessing transformations, aligns features to the training schema, loads the trained model, and produces both a probability score and a classification.

```bash
python scripts/predict.py \
    --sample-path /data/new_patient/ \
    --model-path  outputs/atom_sfm_600 \
    --schema-path schema_600f.json
```

---

## Repository Structure

```
refactored/
├── renv.lock                          # Pinned R dependencies
├── environment.yml                    # Pinned Python dependencies
│
├── scripts/                           # Execution layer — all runnable scripts
│   ├── run_pipeline.py                # Master orchestrator (all 4 stages)
│   ├── run_eda.R                      # Stage 1: subsampling + EDA
│   ├── run_feature_filtering.R        # Stage 2: feature engineering
│   ├── ML_atom_SFM_600f_refactored.py # Stage 3: 600-feature ML pipeline
│   ├── ML_atom_SFS_16f_refactored.py  # Stage 4: 16-feature ML pipeline
│   └── predict.py                     # Inference on new patient samples
│
├── R/                                 # R utility modules (sourced by R scripts)
│   ├── packages.R                     # load_packages()
│   ├── data_processing.R              # load_and_prepare_data()
│   ├── plots.R                        # plot_unique(), plot_diversity(), plot_overlap(), etc.
│   ├── statistics.R                   # run_t_test()
│   └── feature_filtering.R            # build_feature_matrix(), filter_top_n(), filter_database_features()
│
├── python/                            # Python utility modules
│   └── ovarian_utils.py               # load_data(), evaluate_model(), plot_roc_cv(), etc.
│
├── data/
│   └── db/                            # External database files (required for stage 2, method 2)
│       ├── SearchTable.tsv            # VDJdb
│       ├── mc.10.9.22.xlsx            # McPAS
│       └── trust4_tcga_deidentified_report.tsv  # TCGA Trust4
│
└── outputs/                           # Generated by running the pipeline (not committed)
    ├── eda/
    │   ├── sub.Rdata                  # Subsampled repertoire — input to stage 2
    │   ├── unique_hd.png / unique_oc.png
    │   ├── gini_hd.png / gini_oc.png
    │   ├── true_div_hd.png / true_div_oc.png
    │   ├── inv_simp_hd.png / inv_simp_oc.png
    │   ├── top_hd.png / top_oc.png
    │   ├── rare_hd.png / rare_oc.png
    │   ├── statistical_tests_summary.csv
    │   ├── overlap_heatmap.html
    │   └── eda.done                   # Sentinel: stage 1 completed successfully
    ├── features/
    │   ├── data_600_ab.csv            # Method 1 output: 600-feature matrix
    │   ├── data_16_ab.csv             # Method 2 output: 16 database-derived features
    │   └── feature_filtering.done     # Sentinel: stage 2 completed successfully
    ├── atom_sfm_600                   # Stage 3 output: trained GBM model (600-feature)
    ├── schema_600f.json               # Stage 3 output: feature schema for inference
    ├── atom_sfsLDA-4_16f              # Stage 4 output: trained LDA model (16-feature)
    └── schema_16f.json                # Stage 4 output: feature schema for inference
```

---

## Setup

### First time on any machine

```bash
# Python
conda env create -f environment.yml
conda activate <env_name>

# R (run inside an R session)
install.packages("renv")
renv::restore()

```

### Configuration

Before running, edit the `CONFIG` block at the top of `scripts/run_pipeline.py` to set paths for your environment:

```python
CONFIG = {
    "raw_data_dir": "/path/to/A_B/folder/",    # raw TCR repertoire input
    "db_dir":       "data/db/",                # database files for stage 2
    "feature_filtering_method": "both",        # "1", "2", or "both"
    ...
}
```

### Running the pipeline

```bash
python scripts/run_pipeline.py
```

### Running inference on a new sample

```bash
python scripts/predict.py \
    --sample-path /path/to/new_patient_repertoire/ \
    --model-path  outputs/atom_sfm_600 \
    --schema-path outputs/schema_600f.json
```

---
