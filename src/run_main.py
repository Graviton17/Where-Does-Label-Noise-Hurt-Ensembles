"""Phase 5 main experiment runner (RQ1, RQ2).

Runs every (dataset, model, noise type, rate) condition with R rounds and
appends one result row per condition to results/raw/main/<dataset>.csv.
The prediction matrix P of every condition is also saved, so later phases
(e.g. the margin-stratified analysis) can reuse it without re-training.

Resumable: conditions already in the CSV are skipped, so the script can be
stopped and restarted at any time.

Run from the project folder:
    conda run -n labelnoise python src/run_main.py
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import (MODEL_FAMILY, MODEL_NAMES, NOISE_TYPES, decompose, load_dataset,  # noqa: E402
                  reference_probs, run_condition)

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "results" / "raw" / "main"
PRED_DIR = BASE / "results" / "raw" / "main_preds"
RATES = [0.05, 0.10, 0.20, 0.30]
R = 51


def conditions():
    """The 13 noise conditions: clean + 3 noise types x 4 rates."""
    yield "none", 0.0
    for noise_type in NOISE_TYPES:
        for rho in RATES:
            yield noise_type, rho


def run_dataset(name, seed, log):
    out_csv = OUT_DIR / f"{name}.csv"
    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = set(zip(prev.model, prev.noise_type, prev.rho.round(4)))

    X_tr, y_tr, X_te, y_te = load_dataset(name, DATA_DIR)
    p_ref = reference_probs(X_tr, y_tr, seed=seed)
    (PRED_DIR / name).mkdir(parents=True, exist_ok=True)

    for model in MODEL_NAMES:
        for noise_type, rho in conditions():
            if (model, noise_type, round(rho, 4)) in done:
                continue
            t = time.perf_counter()
            P, stats = run_condition(X_tr, y_tr, X_te, model, noise_type, rho,
                                     R=R, seed=seed, p_ref=p_ref)
            row = {"dataset": name, "model": model, "family": MODEL_FAMILY[model],
                   "noise_type": noise_type, "rho": rho, "R": R,
                   "n_train": len(y_tr), "n_test": len(y_te),
                   "pos_share_clean": float(y_tr.mean()),
                   **decompose(P, y_te), **stats,
                   "seconds": round(time.perf_counter() - t, 2),
                   "finished_at": datetime.now().isoformat(timespec="seconds")}
            np.savez_compressed(PRED_DIR / name / f"{model}__{noise_type}__{rho:.2f}.npz", P=P)
            # append immediately, so a crash loses at most one condition
            pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
            log(f"{name:24s} {model:8s} {noise_type:9s} {rho:.2f}  error {row['error']:.4f}  "
                f"bias {row['bias']:.4f}  net_var {row['net_var']:+.4f}  ({row['seconds']:.1f}s)")


def run_all(datasets=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed = json.loads((BASE / "config.json").read_text())["SEED"]
    if datasets is None:   # smallest first, so early results arrive quickly
        datasets = pd.read_csv(BASE / "results" / "tables" / "datasets.csv") \
                     .sort_values("rows")["dataset"].tolist()
    log_path = OUT_DIR / "progress.log"

    def log(msg):
        line = f"{datetime.now():%H:%M:%S}  {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    t0 = time.perf_counter()
    log(f"START  {len(datasets)} datasets x {len(MODEL_NAMES)} models x 13 conditions, R = {R}")
    for name in datasets:
        run_dataset(name, seed, log)
    log(f"DONE   in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    run_all(sys.argv[1:] or None)
