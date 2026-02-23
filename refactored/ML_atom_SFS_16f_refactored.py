# %%
"""
ML_atom_SFS_16f_refactored.py
------------------------------
16-feature pipeline: database-derived features + SFS selection + LDA model.
Refactored to use ovarian_utils.py shared module.

Feature source:
    Features derived from VDJdb, McPAS, and TCGA-TRUST4 databases
    (see feature_filtering.R — Method 2). 16 of 25 candidate TCRs
    were present in the study dataset.

Feature selection strategy:
    Step 1 — SFS (backward): 16 → 8 features  [broad search across solvers]
    Step 2 — SFS (backward) with LDA solver:  16 → 4 features  [refinement]

Winning model: LDA with SFS(LDA) down to 4 features.
"""

import pandas as pd
from atom import ATOMClassifier, ATOMLoader
from ovarian_utils import (
    load_data,
    evaluate_model,
    plot_confusion,
    plot_roc_cv,
    report_errors,
    get_best_threshold,
)
from predict import save_feature_schema
# ---------------------------------------------------------------------------
# Configuration — change paths here, not scattered through the code
# ---------------------------------------------------------------------------

DATA_PATH    = "/home/dsi/zuckerm1/r_files/ovarian_original_wo_mistakes/data_16_ab.csv"
SAVE_NAME_1  = "atom_sfs_16f"
SAVE_NAME_2  = "atom_sfsLDA-4_16f"
RANDOM_STATE = 1
TEST_SIZE    = 0.25

# %%
# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

X = load_data(DATA_PATH)
print(X.head(5))
print("Shape:", X.shape)

# %%
# ---------------------------------------------------------------------------
# Step 1 — SFS backward: 16 → 8 features, across 4 solvers x 8 models
# ---------------------------------------------------------------------------

solvers = ["LDA", "XGB", "LR", "LGB"]
models  = ["GBM", "LR", "LDA", "QDA", "RNN", "AdaB", "RF", "XGB"]

atom = ATOMClassifier(X, y="y", test_size=TEST_SIZE, verbose=2,
                      warnings=False, random_state=RANDOM_STATE, index=True)

for solver in solvers:
    atom.branch = f"sfs_{solver}_from_master"
    atom.feature_selection(strategy="SFS", solver=solver, n_features=8,
                           random_state=0, direction="backward")

    models_named = [f"{model}_{solver}" for model in models]
    atom.run(models=models_named, n_calls=50, n_initial_points=10,
             metric="AUC", n_bootstrap=5)

atom.save(SAVE_NAME_1)

# %%
# ---------------------------------------------------------------------------
# Reload and inspect Step 1 results
# ---------------------------------------------------------------------------

atom = ATOMLoader(SAVE_NAME_1)
atom.branch = "master"

print("Winner:", atom.winner)
atom.plot_results()
atom.winner.plot_feature_importance()

# %% [markdown]
# LDA solver produced the best Step 1 results.
# Feature importance shows not all 8 features contribute meaningfully.
# Proceeding to Step 2: SFS(LDA) reducing directly from 16 to 4 features.

# %%
# ---------------------------------------------------------------------------
# Step 2 — SFS(LDA): 16 → 4 features
# ---------------------------------------------------------------------------

N_FEATURES = 4
models      = ["GBM", "LR", "LDA", "QDA", "RNN", "AdaB", "RF", "XGB"]

atom2 = ATOMClassifier(X, y="y", test_size=TEST_SIZE, verbose=2,
                       warnings=False, random_state=RANDOM_STATE, index=True)

atom2.branch = f"sfsLDA_{N_FEATURES}_from_master"
atom2.feature_selection(strategy="SFS", solver="LDA", n_features=N_FEATURES,
                        random_state=0, direction="backward")

model_names = [f"{model}_sfsLDA{N_FEATURES}" for model in models]
atom2.run(models=model_names, n_calls=50, n_initial_points=10,
          metric="AUC", n_bootstrap=5)

atom2.save(SAVE_NAME_2, save_data=False)

# %%
# ---------------------------------------------------------------------------
# Reload and inspect Step 2 results
# ---------------------------------------------------------------------------

atom2 = ATOMLoader(SAVE_NAME_2, data=(X, "y"), verbose=1)
atom2.branch = "master"

print("Winner:", atom2.winner)
atom2.plot_results()

print("Selected features:", atom2.features)
print("Best params:", atom2.winner.best_params)
print("Coefficients:", atom2.winner.estimator.coef_)

atom2.winner.plot_feature_importance()

# Switch to the winning branch for evaluation and the ROC CV loop
atom2.branch = f"sfsLDA_{N_FEATURES}"

# %%
# ---------------------------------------------------------------------------
# Evaluate winner — on TEST SET to avoid data leakage
# ---------------------------------------------------------------------------

metrics = evaluate_model(
    atom_model=atom2.winner,
    X=atom2.X_test,
    y_true=atom2.y_test,
    decision_fn=atom2.LDA_sfsLDA4.decision_function,
)

# %%
# Confusion matrix on test set
y_prob_test = atom2.winner.predict_proba(atom2.X_test)[:, 1]

plot_confusion(
    y_true=atom2.y_test,
    y_prob=y_prob_test,
    threshold=metrics["Threshold"],
    title="Confusion Matrix — Test Set (LDA, 4 features)",
    save_path="cm_test_4f.pdf",
)

# %%
# atom2's built-in test-set evaluation (cross-check)
atom2.winner.evaluate(threshold=metrics["Threshold"]).to_frame().T

# %%
# Error analysis on test set
report_errors(atom2.y_test, y_prob_test, metrics["Threshold"], atom2.y_test.index)

# %%
# ---------------------------------------------------------------------------
# Cross-validated ROC — 9 random splits, LDA on 4 pre-selected features.
# NOTE: pass atom2.X (the feature-selected subset on the winning branch),
#       not the original full X, so the CV loop trains on the same 4 features.
# ---------------------------------------------------------------------------

plot_roc_cv(
    X=atom2.X,
    y=atom2.y,
    model_name="LDA",
    n_splits=9,
    test_size=(20 / 83),
    save_path="roc_4f.pdf",
)

# %%
save_feature_schema(atom2, "atom_sfsLDA-4_16f", metrics["Threshold"], "schema_16f.json")