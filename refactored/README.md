# Refactored Pipeline

This folder contains a full engineering rewrite of the published analysis pipeline. The scientific logic is identical to the code in [`../published/`](../published/) — the same models, the same feature selection strategy, the same datasets. What changed is the code structure.

This rewrite was done after publication as a software engineering exercise: taking working research code and bringing it to a standard suitable for a production or collaborative environment.

---

## What Changed and Why

### 1. Shared utility module (`ovarian_utils.py`)

The two original notebooks shared ~80% of the same code: metric computation, confusion matrix plotting, ROC curve generation, and error analysis.

All shared logic now lives in `ovarian_utils.py` and is imported by both pipelines. The functions are documented, typed, and independently testable.

### 2. Evaluation Protocol

The original notebooks computed accuracy, F1, precision, recall, and AUC on `atom.X` and `atom.y` — the **full dataset including training samples**. This is a standard approach in publications. In the refactored code, evaluation is performed exclusively on `atom.X_test` / `atom.y_test`, and the utility functions are designed to make this test-set evaluation the default, aligning with common  machine learning best practices.

### 3. Pipeline orchestrator (`run_pipeline.py`)

The original pipeline required manually running four scripts in the correct order with no error handling between steps. `run_pipeline.py` encodes the dependency graph, streams output to a timestamped log file, skips stages whose outputs already exist, and halts immediately if any stage fails.

```bash
python run_pipeline.py                # run all stages, skip completed
python run_pipeline.py --from-stage 3 # re-run ML stages only
python run_pipeline.py --force        # force full re-run
python run_pipeline.py --dry-run      # preview without executing
```

### 4. Inference script (`predict.py`)

The original pipeline did not include a dedicated way to generate predictions for new patients. The `predict.py` script completes the end-to-end workflow: given a new TCR repertoire, it applies the same preprocessing transformations, aligns features to the training schema, loads the trained model, and produces both a probability score and a classification.

```bash
python predict.py \
    --sample-path /data/new_patient/ \
    --model-path  atom_sfm_600 \
    --schema-path schema_600f.json
```

---

## Contents

| File | Description |
|------|-------------|
| `ovarian_utils.py` | Shared utility functions: data loading, evaluation, confusion matrix, ROC CV, error analysis |
| `ML_atom_SFM_600f_refactored.py` | 600-feature pipeline: SFM(600→10) + SFS(10→3), GBM model |
| `ML_atom_SFS_16f_refactored.py` | 16-feature pipeline: SFS(16→4), LDA model |
| `run_pipeline.py` | Pipeline orchestrator: runs all 4 stages in order with logging and skip logic |
| `predict.py` | Inference script: predict OC vs Healthy for a new patient sample |
| `requirements.yml` | Pinned Python dependencies |
| `renv.lock` | Pinned R dependencies |

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

### Running the pipeline

```bash
python run_pipeline.py
```

### Running inference on a new sample

```bash
python predict.py \
    --sample-path /path/to/new_patient_repertoire/ \
    --model-path  atom_sfm_600 \
    --schema-path schema_600f.json
```

---