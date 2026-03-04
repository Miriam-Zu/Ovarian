# %%
"""
ML_atom_SFM_600f_refactored.py
-------------------------------
600-feature pipeline: SFM feature selection + multi-model training.
Refactored to use ovarian_utils.py shared module.

Feature selection strategy:
    Step 1 — SFM (Select From Model): 600 to 10 features
    Step 2 — SFS (Sequential Feature Selector): 10 to 3 features  [2nd loop]

Winning model: GBM with SFM(LGB) + SFS(GBM) down to 3 features.
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

DATA_PATH   = "/home/dsi/zuckerm1/r_files/original_wo_mistakes/data_600_ab.csv"
SAVE_NAME   = "atom_sfm_600"
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
# Step 1 — SFM: 600 to 10 features, across 4 solvers x 8 models
# ---------------------------------------------------------------------------

solvers = ["LDA", "XGB", "LR", "LGB"]
models  = ["GBM", "LR", "LDA", "QDA", "RNN", "AdaB", "RF", "XGB"]

atom = ATOMClassifier(X, y="y", test_size=TEST_SIZE, verbose=2,
                      warnings=False, random_state=RANDOM_STATE, index=True)

for solver in solvers:
    atom.branch = f"sfm_{solver}_from_master"
    atom.feature_selection(strategy="SFM", solver=solver, n_features=10, random_state=0)

    models_named = [f"{model}_{solver}" for model in models]
    atom.run(models=models_named, n_calls=50, n_initial_points=10,
             metric="AUC", n_bootstrap=5)

atom.save(SAVE_NAME, save_data=False)

# %%
# ---------------------------------------------------------------------------
# Reload and inspect Step 1 results
# ---------------------------------------------------------------------------

atom = ATOMLoader(SAVE_NAME, data=(X, "y"))
atom.branch = "master"

atom.plot_results()
atom.winner.plot_feature_importance()

# %% [markdown]
# The feature importance plot shows not all 10 features contribute meaningfully.
# Proceeding to Step 2: add SFS to reduce further, using LGB as the SFM solver
# (best performer from Step 1) and varying the final feature count (2–7).

# %%
# ---------------------------------------------------------------------------
# Step 2 — SFM(LGB) + SFS: 10 to n features, across 3 SFS solvers x 6 counts
# ---------------------------------------------------------------------------

solvers_sfs = ["GBM", "LDA", "LGB"]

for solver in solvers_sfs:
    for n_feat in range(2, 8):
        branch_name = f"sfmLGB_sfs{solver}_{n_feat}_from_master"
        atom.branch = branch_name
        atom.feature_selection(strategy="SFM", solver="LGB", n_features=10, random_state=0)
        atom.feature_selection(strategy="SFS", solver=solver, n_features=n_feat,
                               random_state=0, direction="backward")

        model_name = f"GBM_{solver}_{n_feat}_2step"
        atom.run(models=[model_name], n_calls=50, n_initial_points=10,
                 metric="AUC", n_bootstrap=5)

        atom.save(SAVE_NAME, save_data=False)

# %%
# ---------------------------------------------------------------------------
# Reload and inspect Step 2 results
# ---------------------------------------------------------------------------

atom = ATOMLoader(SAVE_NAME, data=(X, "y"))
atom.branch = "master"
atom.plot_results()

# %% [markdown]
# Winner: GBM_GBM_3_2step
# Pipeline: SFM(LGB): 600→10, then SFS(GBM): 10→3. Final model: GBM.

# %%
# ---------------------------------------------------------------------------
# Inspect winner
# ---------------------------------------------------------------------------

print("Winner:", atom.winner)
print("Features:", atom.winner.features)
print("Best params:", atom.winner.best_params)
atom.winner.plot_feature_importance()

# Switch to the winning branch for the ROC CV loop
atom.branch = "sfmLGB_sfsGBM_3"

# %%
# ---------------------------------------------------------------------------
# Evaluate winner — on TEST SET to avoid data leakage
# ---------------------------------------------------------------------------

metrics = evaluate_model(
    atom_model=atom.winner,
    X=atom.X_test,
    y_true=atom.y_test,
    decision_fn=atom.winner.decision_function,
)

# %%
# Confusion matrix on test set
y_prob_test = atom.winner.predict_proba(atom.X_test)[:, 1]

plot_confusion(
    y_true=atom.y_test,
    y_prob=y_prob_test,
    threshold=metrics["Threshold"],
    title="Confusion Matrix — Test Set (GBM, 3 features)",
    save_path="cm_test_3f.pdf",
)

# %%
# atom's built-in test-set evaluation (cross-check)
atom.winner.evaluate(threshold=metrics["Threshold"]).to_frame().T

# %%
# Error analysis on test set
report_errors(atom.y_test, y_prob_test, metrics["Threshold"], atom.y_test.index)

# %%
# ---------------------------------------------------------------------------
# Cross-validated ROC — 9 random splits, GBM on 3 pre-selected features
# NOTE: pass atom.X (the feature-selected subset on the winning branch),
#       not the original full X, so the CV loop trains on the same features.
# ---------------------------------------------------------------------------

plot_roc_cv(
    X=atom.X,
    y=atom.y,
    model_name="GBM",
    n_splits=9,
    test_size=(20 / 83),
    save_path="roc_3f.pdf",
)

# %%
# Save the feature schema after training so inference can replicate the exact feature set.
save_feature_schema(atom, "atom_sfm_600", metrics["Threshold"], "schema_600f.json")