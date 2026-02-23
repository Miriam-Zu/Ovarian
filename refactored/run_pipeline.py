"""
run_pipeline.py
---------------
Orchestrator for the Ovarian Cancer TCR ML pipeline.

Runs all stages in dependency order, skipping stages whose outputs
already exist (unless --force is passed).

Pipeline stages:
    1. immunarch_analysis.R   — load raw TCR data, subsample, EDA → sub.Rdata
    2. feature_filtering.R    — engineer features, export CSVs
    3. ML_atom_SFM_600f       — 600-feature SFM+SFS ML pipeline
    4. ML_atom_SFS_16f        — 16-feature database-derived ML pipeline

Usage:
    # Run all stages (skip completed ones)
    python run_pipeline.py

    # Force re-run all stages from scratch
    python run_pipeline.py --force

    # Run from a specific stage onwards
    python run_pipeline.py --from-stage 3

    # Run only a specific stage
    python run_pipeline.py --only 2

    # Dry run: print what would be executed without running anything
    python run_pipeline.py --dry-run
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — edit these paths to match your environment
# ---------------------------------------------------------------------------

CONFIG = {
    # Root directory of the project
    "project_dir": Path(__file__).parent.resolve(),

    # Raw TCR data directory (input to stage 1)
    "raw_data_dir": "/path/to/processed/data/",

    # Where intermediate and output files land
    "output_dir": Path(__file__).parent / "outputs",

    # R executable (use full path if Rscript is not on PATH)
    "rscript": "Rscript",

    # R scripts
    "r_analysis_script":  "immunarch_analysis.R",
    "r_filtering_script": "feature_filtering.R",

    # Python scripts (refactored notebooks exported as .py)
    "ml_600_script": "ML_atom_SFM_600f_refactored.py",
    "ml_16_script":  "ML_atom_SFS_16f_refactored.py",

    # Expected output files — used to determine if a stage can be skipped
    "stage_outputs": {
        1: ["sub.Rdata"],
        2: ["data_600_ab.csv", "data_16_ab.csv"],
        3: ["atom_sfm_600"],
        4: ["atom_sfsLDA-4_16f"],
    },
}

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def outputs_exist(stage: int) -> bool:
    """Return True if all expected output files for a stage already exist."""
    output_dir = CONFIG["output_dir"]
    expected = CONFIG["stage_outputs"].get(stage, [])
    return all((output_dir / f).exists() for f in expected)


def run_command(cmd: list, stage_name: str, dry_run: bool = False) -> None:
    """
    Run a shell command, stream output to the log, and raise on failure.

    Parameters
    ----------
    cmd : list of str
        Command and arguments.
    stage_name : str
        Human-readable label for log messages.
    dry_run : bool
        If True, log the command but don't execute it.
    """
    cmd_str = " ".join(str(c) for c in cmd)
    if dry_run:
        log.info(f"[DRY RUN] Would run: {cmd_str}")
        return

    log.info(f"Running: {cmd_str}")
    start = time.time()

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=CONFIG["project_dir"],
    )

    # Stream captured output to log
    for line in result.stdout.splitlines():
        log.info(f"  {line}")

    elapsed = time.time() - start
    if result.returncode != 0:
        log.error(f"Stage '{stage_name}' FAILED after {elapsed:.1f}s "
                  f"(exit code {result.returncode})")
        sys.exit(result.returncode)

    log.info(f"Stage '{stage_name}' completed in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_1_immunarch(dry_run: bool = False) -> None:
    """
    Stage 1: Load raw TCR repertoire data, subsample to 24,840 clones,
    run exploratory analysis (clonality, diversity, overlap).
    Output: sub.Rdata
    """
    script = CONFIG["project_dir"] / CONFIG["r_analysis_script"]
    run_command(
        [CONFIG["rscript"], str(script), CONFIG["raw_data_dir"]],
        stage_name="immunarch_analysis",
        dry_run=dry_run,
    )


def stage_2_feature_filtering(dry_run: bool = False) -> None:
    """
    Stage 2: Engineer features from sub.Rdata.
    Method 1 — top 600 most shared clonotypes → data_600_ab.csv
    Method 2 — 16 database-derived clonotypes → data_16_ab.csv
    """
    script = CONFIG["project_dir"] / CONFIG["r_filtering_script"]
    run_command(
        [CONFIG["rscript"], str(script)],
        stage_name="feature_filtering",
        dry_run=dry_run,
    )


def stage_3_ml_600(dry_run: bool = False) -> None:
    """
    Stage 3: 600-feature ML pipeline.
    SFM (600→10) + SFS (10→3), GBM model.
    Output: atom_sfm_600
    """
    script = CONFIG["project_dir"] / CONFIG["ml_600_script"]
    run_command(
        [sys.executable, str(script)],
        stage_name="ML_SFM_600f",
        dry_run=dry_run,
    )


def stage_4_ml_16(dry_run: bool = False) -> None:
    """
    Stage 4: 16-feature (database-derived) ML pipeline.
    SFS (16→4), LDA model.
    Output: atom_sfsLDA-4_16f
    """
    script = CONFIG["project_dir"] / CONFIG["ml_16_script"]
    run_command(
        [sys.executable, str(script)],
        stage_name="ML_SFS_16f",
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# Stage registry — order matters
# ---------------------------------------------------------------------------

STAGES = {
    1: {"name": "immunarch_analysis",  "fn": stage_1_immunarch},
    2: {"name": "feature_filtering",   "fn": stage_2_feature_filtering},
    3: {"name": "ML_SFM_600f",         "fn": stage_3_ml_600},
    4: {"name": "ML_SFS_16f",          "fn": stage_4_ml_16},
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(from_stage: int = 1, only_stage: int = None,
                 force: bool = False, dry_run: bool = False) -> None:
    """
    Execute pipeline stages in order, with skip logic and error handling.
    """
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Ovarian Cancer TCR ML Pipeline")
    log.info(f"Log file: {log_file}")
    log.info(f"Force re-run: {force}  |  Dry run: {dry_run}")
    log.info("=" * 60)

    stages_to_run = (
        [only_stage] if only_stage
        else [s for s in STAGES if s >= from_stage]
    )

    for stage_num in stages_to_run:
        stage = STAGES[stage_num]
        name  = stage["name"]

        log.info(f"\n--- Stage {stage_num}: {name} ---")

        if not force and outputs_exist(stage_num):
            log.info(f"Output already exists — skipping. "
                     f"(Use --force to re-run)")
            continue

        stage["fn"](dry_run=dry_run)

    log.info("\n" + "=" * 60)
    log.info("Pipeline complete.")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ovarian Cancer TCR ML Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--from-stage", type=int, default=1, metavar="N",
        help="Start from stage N (1-4). Stages before N are skipped.",
    )
    parser.add_argument(
        "--only", type=int, default=None, metavar="N",
        help="Run only stage N, regardless of dependency state.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run all stages even if outputs already exist.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands that would be run without executing them.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.only and args.from_stage != 1:
        log.error("--only and --from-stage cannot be used together.")
        sys.exit(1)

    run_pipeline(
        from_stage=args.from_stage,
        only_stage=args.only,
        force=args.force,
        dry_run=args.dry_run,
    )