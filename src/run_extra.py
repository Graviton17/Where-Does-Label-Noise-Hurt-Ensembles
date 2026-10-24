"""Phase 6 runner: RQ3 (sources of variance), RQ4 (regularisation) and the
capacity control. Resumable in the same way as run_main.py.

Run from the project folder:
    conda run -n labelnoise python src/run_extra.py            # all three, in order
    conda run -n labelnoise python src/run_extra.py rq4         # only one experiment
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_main  # noqa: E402
from core import (CAPACITY_MODELS, DESIGNS, MODEL_FAMILY, MODEL_NAMES, NOISE_TYPES,  # noqa: E402
                  REG_MODELS, TREE_DEPTH, decompose, load_dataset, reference_probs, run_design)

BASE = run_main.BASE
RAW = BASE / "results" / "raw"
EXTRA_RATES = [0.10, 0.20, 0.30]
RQ3_RHO = 0.20
R = run_main.R


def datasets_small_first():
    return pd.read_csv(BASE / "results" / "tables" / "datasets.csv").sort_values("rows")["dataset"].tolist()


def make_logger(path):
    def log(msg):
        line = f"{datetime.now():%H:%M:%S}  {msg}"
        print(line, flush=True)
        with open(path, "a") as f:
            f.write(line + "\n")
    return log


def run_grid(experiment, models):
    """RQ4 and capacity: same loop as Phase 5, other models, rates {10, 20, 30 %} + clean."""
    out_dir = RAW / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(out_dir / "progress.log")
    seed = json.loads((BASE / "config.json").read_text())["SEED"]
    log(f"START {experiment}: {len(models)} models x 10 conditions, R = {R}")
    for name in datasets_small_first():
        run_main.run_dataset(name, seed, log, models=models, rates=EXTRA_RATES,
                             out_dir=out_dir, pred_dir=None)
    log(f"DONE {experiment}")


def run_rq3():
    out_dir = RAW / "rq3"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(out_dir / "progress.log")
    seed = json.loads((BASE / "config.json").read_text())["SEED"]
    log(f"START rq3: {len(DESIGNS)} designs x {len(MODEL_NAMES)} models x 3 noise types, rho = {RQ3_RHO}, R = {R}")
    for name in datasets_small_first():
        out_csv = out_dir / f"{name}.csv"
        done = set()
        if out_csv.exists():
            prev = pd.read_csv(out_csv)
            done = set(zip(prev.design, prev.model, prev.noise_type))
        X_tr, y_tr, X_te, y_te = load_dataset(name, run_main.DATA_DIR)
        p_ref = reference_probs(X_tr, y_tr, seed=seed)
        for design in DESIGNS:
            for model in MODEL_NAMES:
                for noise_type in NOISE_TYPES:
                    if (design, model, noise_type) in done:
                        continue
                    t = time.perf_counter()
                    P, stats = run_design(X_tr, y_tr, X_te, design, model, noise_type, RQ3_RHO,
                                          R=R, seed=seed, p_ref=p_ref)
                    row = {"dataset": name, "design": design, "model": model,
                           "family": MODEL_FAMILY[model], "noise_type": noise_type, "rho": RQ3_RHO,
                           "R": R, **decompose(P, y_te), **stats,
                           "seconds": round(time.perf_counter() - t, 2),
                           "finished_at": datetime.now().isoformat(timespec="seconds")}
                    pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
                    log(f"{name:24s} {design:13s} {model:8s} {noise_type:9s}  error {row['error']:.4f}  "
                        f"var {row['var_unbiased'] + row['var_biased']:.4f}  ({row['seconds']:.1f}s)")
    log("DONE rq3")


EXPERIMENTS = {
    "rq4": lambda: run_grid("rq4", list(REG_MODELS)),
    "capacity": lambda: run_grid("capacity", list(CAPACITY_MODELS)),
    "rq3": run_rq3,
}

if __name__ == "__main__":
    for exp in (sys.argv[1:] or ["rq4", "capacity", "rq3"]):
        EXPERIMENTS[exp]()
