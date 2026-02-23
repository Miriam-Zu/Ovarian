"""
predict.py
----------
Inference script for the Ovarian Cancer TCR ML pipeline.

Given a new patient's TCR repertoire file, this script:
    1. Subsamples to the training-time clone count (24,840)
    2. Computes public clonotype proportions (same method as training)
    3. Aligns features to the training schema (fills missing clonotypes with 0)
    4. Loads the saved ATOM model and produces a probability + classification

This script requires:
    - A saved ATOM model file (atom_sfm_600 or atom_sfsLDA-4_16f)
    - A saved feature schema file (.json, produced by save_feature_schema()
      during training — see bottom of this file)
    - immunarch installed in R (for preprocessing the new sample)
    - rpy2 installed in Python (to call R preprocessing from Python)

Usage:
    # Predict using the 600-feature GBM model
    python predict.py \\
        --sample-path /path/to/new_patient_repertoire/ \\
        --model-path  atom_sfm_600 \\
        --schema-path schema_600f.json \\
        --threshold   0.412

    # Predict using the 16-feature LDA model
    python predict.py \\
        --sample-path /path/to/new_patient_repertoire/ \\
        --model-path  atom_sfsLDA-4_16f \\
        --schema-path schema_16f.json \\
        --threshold   0.387

    # Output prediction as JSON (for downstream use)
    python predict.py --sample-path ... --model-path ... --schema-path ... --json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Subsampling depth must match the value used during training (see immunarch_analysis.R)
SUBSAMPLE_N = 24840


# ---------------------------------------------------------------------------
# Step 1 — Preprocess a new sample using R/immunarch via rpy2
# ---------------------------------------------------------------------------

def preprocess_sample(sample_path: str) -> pd.DataFrame:
    """
    Run immunarch preprocessing on a new patient's TCR repertoire directory.

    Replicates the R pipeline steps:
        repLoad → repSample (downsample to SUBSAMPLE_N) → pubRep (proportions)

    Requires rpy2 and immunarch to be installed.

    Parameters
    ----------
    sample_path : str
        Path to a directory containing the new patient's repertoire file(s)
        in a format readable by immunarch::repLoad (e.g. MiXCR, AIRR, etc.)

    Returns
    -------
    pd.DataFrame
        Single-row dataframe where columns are CDR3 amino acid sequences
        and the value is the clonotype proportion in this sample.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        pandas2ri.activate()
    except ImportError:
        log.error(
            "rpy2 is not installed. Install with: pip install rpy2\n"
            "Alternatively, preprocess the sample in R and pass the resulting "
            "CSV directly using --csv-path instead of --sample-path."
        )
        sys.exit(1)

    log.info(f"Loading repertoire from: {sample_path}")

    immunarch = importr("immunarch")
    base      = importr("base")

    # Load raw repertoire
    immdata = immunarch.repLoad(sample_path)
    data    = immdata.rx2("data")

    # Check clone count before subsampling
    clones = immunarch.repExplore(data, _method="clones")
    min_clones = int(base.min(clones.rx2("Clones"))[0])
    if min_clones < SUBSAMPLE_N:
        log.warning(
            f"Sample has {min_clones} clones, which is below the subsampling "
            f"threshold of {SUBSAMPLE_N}. The sample will not be subsampled. "
            f"Prediction reliability may be reduced."
        )
        sub = data
    else:
        log.info(f"Subsampling to {SUBSAMPLE_N} clones...")
        sub = immunarch.repSample(data, _method="downsample", _n=SUBSAMPLE_N)

    # Compute public clonotype proportions (same as training)
    log.info("Computing clonotype proportions...")
    pub_prop = immunarch.pubRep(sub, _quant="prop", _verbose=False, _col="aa")

    # Convert to pandas and clean up
    pub_df = pandas2ri.rpy2py(pub_prop)
    pub_df = pub_df.fillna(0)

    # Drop the 'Samples' column if present
    if "Samples" in pub_df.columns:
        pub_df = pub_df.drop(columns=["Samples"])

    # Transpose: rows=samples, cols=clonotypes (same orientation as training)
    pub_df_t = pub_df.set_index(pub_df.columns[0]).T

    log.info(f"Preprocessed sample shape: {pub_df_t.shape}")
    return pub_df_t


def load_csv_sample(csv_path: str) -> pd.DataFrame:
    """
    Load a pre-processed sample CSV (alternative to --sample-path).
    The CSV should have the same format as data_600_ab.csv or data_16_ab.csv
    but for a single new sample (one row, CDR3 sequences as columns).

    Parameters
    ----------
    csv_path : str
        Path to the pre-processed feature CSV for the new sample.
    """
    log.info(f"Loading pre-processed CSV from: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)
    log.info(f"Loaded CSV shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Align new sample to the training feature schema
# ---------------------------------------------------------------------------

def load_feature_schema(schema_path: str) -> dict:
    """
    Load the feature schema saved during training.

    The schema contains:
        - features: list of CDR3 column names used during training
        - model_path: which model this schema belongs to
        - n_features: number of features
        - threshold: optimal classification threshold from training

    Parameters
    ----------
    schema_path : str
        Path to the .json schema file produced by save_feature_schema().
    """
    with open(schema_path) as f:
        schema = json.load(f)
    log.info(f"Loaded feature schema: {len(schema['features'])} features "
             f"from {schema.get('model_path', 'unknown model')}")
    return schema


def align_features(sample_df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """
    Align a new sample's features to the training schema.

    - Columns in the schema but missing from the sample → filled with 0
      (clonotype not observed in this patient)
    - Columns in the sample but not in the schema → dropped
      (not part of the trained feature set)

    Parameters
    ----------
    sample_df : pd.DataFrame
        Feature dataframe for the new sample.
    schema : dict
        Feature schema loaded from save_feature_schema().

    Returns
    -------
    pd.DataFrame
        Single-row dataframe with exactly the columns in the training schema,
        in the same order.
    """
    training_features = schema["features"]

    missing = set(training_features) - set(sample_df.columns)
    extra   = set(sample_df.columns) - set(training_features)

    if missing:
        log.info(f"{len(missing)} training features absent in this sample "
                 f"(filling with 0 — clonotype not observed).")
        for col in missing:
            sample_df[col] = 0.0

    if extra:
        log.info(f"{len(extra)} sample columns not in training schema "
                 f"(dropping).")

    # Reorder to match training column order exactly
    aligned = sample_df[training_features]
    log.info(f"Aligned feature matrix shape: {aligned.shape}")
    return aligned


# ---------------------------------------------------------------------------
# Step 3 — Load model and predict
# ---------------------------------------------------------------------------

def load_model(model_path: str, aligned_sample: pd.DataFrame):
    """
    Load a saved ATOM model and return the winner model object.

    Parameters
    ----------
    model_path : str
        Path to the saved atom model (without extension).
    aligned_sample : pd.DataFrame
        Feature-aligned sample — passed as data so ATOMLoader
        can reconstruct the model without the original training data.
    """
    from atom import ATOMLoader

    log.info(f"Loading ATOM model from: {model_path}")
    # ATOMLoader needs data to reconstruct the model; we pass the new sample
    # as a stand-in so the loader can initialise correctly.
    atom = ATOMLoader(model_path, data=(aligned_sample, None), verbose=0)
    return atom


def predict(atom, aligned_sample: pd.DataFrame,
            threshold: float) -> dict:
    """
    Run inference on an aligned sample.

    Parameters
    ----------
    atom : ATOMLoader object
    aligned_sample : pd.DataFrame
    threshold : float
        Classification threshold. Use the value saved in the schema
        (determined by Youden J on the training set).

    Returns
    -------
    dict with keys:
        probability  — predicted probability of OC (class 1)
        prediction   — 'OC' or 'Healthy'
        threshold    — threshold used
        confidence   — distance of probability from the threshold
    """
    log.info("Running inference...")
    prob = atom.winner.predict_proba(aligned_sample)[:, 1][0]
    label = "OC" if prob >= threshold else "Healthy"
    confidence = abs(prob - threshold)

    result = {
        "probability": round(float(prob), 4),
        "prediction":  label,
        "threshold":   round(float(threshold), 4),
        "confidence":  round(float(confidence), 4),
    }

    return result


# ---------------------------------------------------------------------------
# Schema saving helper — call this at the end of each training script
# ---------------------------------------------------------------------------

def save_feature_schema(atom, model_path: str, threshold: float,
                        schema_path: str) -> None:
    """
    Save the feature schema after training so inference can replicate
    the exact feature set.

    Parameters
    ----------
    atom : ATOMClassifier or ATOMLoader object
    model_path : str
    threshold : float
        Optimal threshold from Youden J (stored in metrics["Threshold"]
        from ovarian_utils.evaluate_model).
    schema_path : str
        Where to save the schema JSON.
    """
    schema = {
        "model_path":  model_path,
        "n_features":  len(atom.features),
        "features":    list(atom.features),
        "threshold":   threshold,
        "subsample_n": SUBSAMPLE_N,
        "created_at":  pd.Timestamp.now().isoformat(),
    }
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)

    log.info(f"Feature schema saved to: {schema_path} "
             f"({len(schema['features'])} features)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict OC vs Healthy from a new TCR repertoire sample.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input: either raw repertoire dir or pre-processed CSV
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--sample-path", type=str,
        help="Path to raw repertoire directory (processed via R/immunarch).",
    )
    input_group.add_argument(
        "--csv-path", type=str,
        help="Path to pre-processed feature CSV (skips R preprocessing).",
    )

    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to the saved ATOM model (e.g. atom_sfm_600).",
    )
    parser.add_argument(
        "--schema-path", type=str, required=True,
        help="Path to the feature schema JSON (e.g. schema_600f.json).",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Classification threshold. Defaults to value stored in schema.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output prediction as JSON (for downstream use).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load and preprocess sample ---
    if args.sample_path:
        sample_df = preprocess_sample(args.sample_path)
    else:
        sample_df = load_csv_sample(args.csv_path)

    # --- Load schema and align features ---
    schema = load_feature_schema(args.schema_path)
    aligned = align_features(sample_df, schema)

    # --- Determine threshold ---
    threshold = args.threshold if args.threshold is not None else schema["threshold"]
    log.info(f"Using threshold: {threshold:.4f}")

    # --- Load model and predict ---
    atom = load_model(args.model_path, aligned)
    result = predict(atom, aligned, threshold)

    # --- Output ---
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "=" * 40)
        print(f"  Prediction  : {result['prediction']}")
        print(f"  Probability : {result['probability']:.4f}")
        print(f"  Threshold   : {result['threshold']:.4f}")
        print(f"  Confidence  : {result['confidence']:.4f}")
        print("=" * 40)

    return result


if __name__ == "__main__":
    main()