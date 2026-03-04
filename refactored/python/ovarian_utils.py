"""
ovarian_utils.py
----------------
Shared utility module for the Ovarian Cancer TCR ML pipeline.
Used by both the 600-feature (SFM) and 16-feature (SFS/database) notebooks.

Usage:
    from ovarian_utils import load_data, evaluate_model, plot_roc_cv, plot_confusion
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy import argmax
from numpy import interp  # replaces deprecated scipy.interp

from sklearn.metrics import (
    roc_auc_score, auc, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score, precision_score, recall_score
)
from atom import ATOMClassifier


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load feature CSV and append binary label column.

    Label convention:
        0 = Healthy (row index ends with 'H')
        1 = Ovarian Cancer

    Parameters
    ----------
    csv_path : str
        Path to the feature CSV file (e.g. data_600_ab.csv or data_16_ab.csv).

    Returns
    -------
    pd.DataFrame
        Feature matrix with an appended 'y' column.
    """
    X = pd.read_csv(csv_path, index_col=0)
    X['y'] = [0 if row_name.endswith('H') else 1 for row_name in X.index]
    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1] - 1} features")
    print(f"  Healthy (0): {(X['y'] == 0).sum()}  |  OC (1): {(X['y'] == 1).sum()}")
    return X


# ---------------------------------------------------------------------------
# 2. Threshold selection
# ---------------------------------------------------------------------------

def get_best_threshold(y_true, y_prob) -> tuple:
    """
    Find the optimal classification threshold using the Youden J statistic.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    best_thresh : float
    fpr : array
    tpr : array
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    J = tpr - fpr
    ix = argmax(J)
    best_thresh = thresholds[ix]
    print(f"Best Threshold = {best_thresh:.3f}")
    return best_thresh, fpr, tpr


# ---------------------------------------------------------------------------
# 3. Evaluation on a dataset split
#    NOTE: always call this on the TEST set (atom.X_test / atom.y_test),
#    not on the full dataset, to avoid inflated metrics.
# ---------------------------------------------------------------------------

def evaluate_model(atom_model, X, y_true, decision_fn=None) -> dict:
    """
    Compute and print classification metrics for a fitted ATOM model.

    Parameters
    ----------
    atom_model : ATOM model object (e.g. atom.winner)
        Must expose predict_proba().
    X : pd.DataFrame
        Feature matrix to evaluate on. Use atom.X_test to avoid data leakage.
    y_true : pd.Series or array-like
        True labels corresponding to X. Use atom.y_test.
    decision_fn : callable or None
        If provided, used for AUC (e.g. atom.winner.decision_function).
        Falls back to predict_proba if None.

    Returns
    -------
    dict of metric name -> value
    """
    y_prob = atom_model.predict_proba(X)[:, 1]
    best_thresh, _, _ = get_best_threshold(y_true, y_prob)
    y_pred = (y_prob > best_thresh).astype(float)

    if decision_fn is not None:
        auc_score = roc_auc_score(y_true, decision_fn(X))
    else:
        auc_score = roc_auc_score(y_true, y_prob)

    metrics = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "F1":        f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall":    recall_score(y_true, y_pred),
        "AUC":       auc_score,
        "Threshold": best_thresh,
    }

    print("\n--- Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.3f}")

    return metrics


# ---------------------------------------------------------------------------
# 4. Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion(y_true, y_prob, threshold: float, title: str = "Confusion Matrix",
                   save_path: str = None):
    """
    Plot a confusion matrix using a given threshold.

    Parameters
    ----------
    y_true : array-like
    y_prob : array-like
        Predicted probabilities for the positive class.
    threshold : float
    title : str
    save_path : str or None
        If provided, saves the figure to this path (e.g. 'cm.pdf').
    """
    y_pred = (y_prob > threshold).astype(float)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal')
    plt.grid(False)
    plt.title(title)

    if save_path:
        plt.savefig(save_path, transparent=True)
        print(f"Confusion matrix saved to: {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# 5. Cross-validated ROC curve (multi-split)
# ---------------------------------------------------------------------------

def plot_roc_cv(X, y, model_name: str = "LDA", n_splits: int = 9,
                test_size: float = 0.25, n_calls: int = 50,
                n_initial_points: int = 10, save_path: str = None):
    """
    Train and evaluate a model across multiple random train/test splits,
    then plot the mean ROC curve with ±1 std deviation band.

    This is the correct place to assess generalization — each fold trains
    and evaluates on genuinely held-out data.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (already feature-selected to the desired subset).
    y : pd.Series
        Binary labels.
    model_name : str
        ATOM model name (e.g. 'LDA', 'GBM').
    n_splits : int
        Number of random splits (default 9).
    test_size : float
        Fraction of data to use as test set per split.
    n_calls : int
        Bayesian optimisation calls per split.
    n_initial_points : int
        Initial random points for Bayesian optimisation.
    save_path : str or None
        If provided, saves the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs_list = [], []

    for seed in range(1, n_splits + 1):
        print(f"\n--- Split {seed}/{n_splits} ---")
        atom_cv = ATOMClassifier(X, y=y, test_size=test_size,
                                 verbose=1, warnings=False, random_state=seed)
        atom_cv.run(models=[model_name],
                    n_calls=n_calls,
                    n_initial_points=n_initial_points,
                    metric="AUC",
                    n_bootstrap=5)

        y_prob = atom_cv.winner.predict_proba_test[:, 1]
        fpr, tpr, _ = roc_curve(atom_cv.y_test, y_prob)
        tprs.append(interp(mean_fpr, fpr, tpr))
        auc_score = roc_auc_score(atom_cv.y_test, y_prob)
        aucs_list.append(auc_score)
        ax.plot(fpr, tpr, lw=1.5, alpha=0.3,
                label=f"Split {seed} (AUC = {auc_score:.2f})")

    # Mean ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs_list)
    ax.plot(mean_fpr, mean_tpr, lw=2.5, color='blue',
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc))

    # Std band
    std_tpr = np.std(tprs, axis=0)
    ax.fill_between(mean_fpr,
                    mean_tpr - std_tpr,
                    np.minimum(mean_tpr + std_tpr, 1),
                    color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.")

    ax.plot([0, 1], [0, 1], linestyle='--', color='grey', label='No Skill')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Cross-validated ROC — {model_name}')
    ax.legend(loc='lower right')
    ax.set_facecolor('w')

    if save_path:
        plt.savefig(save_path, transparent=True)
        print(f"ROC plot saved to: {save_path}")

    plt.show()
    print(f"\nMean AUC across {n_splits} splits: {mean_auc:.3f} ± {std_auc:.3f}")


# ---------------------------------------------------------------------------
# 6. Error analysis helpers
# ---------------------------------------------------------------------------

def get_false_negatives(y_true, y_pred_binary, index) -> list:
    """Return sample IDs predicted as Healthy but actually OC (missed cancers)."""
    fn_idx = [i for i in range(len(y_true))
               if y_true.to_numpy()[i] == 1 and y_pred_binary[i] == 0]
    return [index[i] for i in fn_idx]


def get_false_positives(y_true, y_pred_binary, index) -> list:
    """Return sample IDs predicted as OC but actually Healthy."""
    fp_idx = [i for i in range(len(y_true))
               if y_true.to_numpy()[i] == 0 and y_pred_binary[i] == 1]
    return [index[i] for i in fp_idx]


def report_errors(y_true, y_prob, threshold: float, index):
    """
    Print false positives and false negatives for a given threshold.

    Parameters
    ----------
    y_true : pd.Series
    y_prob : array-like
    threshold : float
    index : pd.Index
        Sample index from the original dataframe.
    """
    y_pred = (y_prob > threshold).astype(float)
    fn = get_false_negatives(y_true, y_pred, index)
    fp = get_false_positives(y_true, y_pred, index)
    print(f"False Negatives (OC missed, predicted Healthy): {fn}")
    print(f"False Positives (Healthy predicted as OC):      {fp}")
    return {"FN": fn, "FP": fp}