"""Phase 6B (RQ5): does the binary pattern hold for multiclass data?

Contents
    select_and_prepare   apply the selection rule to OpenML-CC18 and save data/multiclass/<name>.npz
    reference_probs_mc   clean multinomial logistic-regression probabilities (for boundary noise)
    inject_noise_mc      symmetric / pair-flip / boundary noise for K classes
    decompose_mc         Domingos 0-1 decomposition for K classes
    run_all              resumable experiment runner -> results/raw/rq5/<name>.csv

Run from the project folder:
    conda run -n labelnoise python src/multiclass.py
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import MODEL_FAMILY, MODEL_NAMES, make_model  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data" / "multiclass"
OUT_DIR = BASE / "results" / "raw" / "rq5"
TABLES = BASE / "results" / "tables"
NOISE_TYPES_MC = ["symmetric", "pair_flip", "boundary"]
RATES = [0.10, 0.20, 0.30]
R = 51


# ---------------------------------------------------------------- data
def select_and_prepare(seed):
    """Same rule as Phase 2, for multiclass: all OpenML-CC18 datasets with > 2 classes,
    <= 50,000 rows, <= 100 features, no missing values, smallest class >= 5 %.
    Preparation is identical to Phase 2 (stratified 70/30 split first; imputer and
    one-hot encoder fitted on the training part only). Classes are relabelled
    0..K-1 from most to least frequent, which fixes the order used by pair-flip noise."""
    import openml
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    openml.config.set_root_cache_directory(str(BASE / "data" / "openml_cache"))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cand = openml.datasets.list_datasets(data_id=list(openml.study.get_suite(99).data),
                                         output_format="dataframe")
    cand["features"] = cand.NumberOfFeatures - 1
    cand["min_class_share"] = (cand.MinorityClassSize / cand.NumberOfInstances).round(3)

    def reason(r):
        if r.NumberOfClasses <= 2:          return "binary (used in the main study)"
        if r.NumberOfInstances > 50_000:    return "more than 50,000 rows"
        if r.features > 100:                return "more than 100 features"
        if r.NumberOfMissingValues > 0:     return "has missing values"
        if r.min_class_share < 0.05:        return "smallest class below 5%"
        return ""
    cand["excluded_because"] = cand.apply(reason, axis=1)
    cand["selected"] = cand.excluded_because == ""
    cand[cand.NumberOfClasses > 2][["did", "name", "NumberOfInstances", "features", "NumberOfClasses",
                                    "NumberOfMissingValues", "min_class_share", "selected",
                                    "excluded_because"]] \
        .sort_values(["selected", "NumberOfInstances"], ascending=[False, True]) \
        .to_csv(TABLES / "dataset_selection_multiclass.csv", index=False)

    rows = []
    for did, name in cand[cand.selected].sort_values("NumberOfInstances")[["did", "name"]].itertuples(index=False):
        short = name.split("_raw_")[0] if name.startswith("jungle") else name
        path = DATA_DIR / f"{short}.npz"
        d = openml.datasets.get_dataset(int(did), download_data=True, download_qualities=False,
                                        download_features_meta_data=False)
        X, y, is_cat, cols = d.get_data(target=d.default_target_attribute)
        order = y.value_counts().index.tolist()                  # most frequent class first
        y_int = y.map({c: i for i, c in enumerate(order)}).astype(int).to_numpy()
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_int, test_size=0.30, stratify=y_int, random_state=seed)
        cat = [c for c, k in zip(cols, is_cat) if k]
        num = [c for c, k in zip(cols, is_cat) if not k]
        prep = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num),
            ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                              ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])
        X_tr = prep.fit_transform(X_tr).astype(np.float64)
        X_te = prep.transform(X_te).astype(np.float64)
        np.savez_compressed(path, X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te)
        rows.append({"dataset": short, "openml_id": int(did), "rows": len(y_int), "classes": len(order),
                     "features_after_encoding": X_tr.shape[1],
                     "smallest_class_share": round(np.bincount(y_int).min() / len(y_int), 3),
                     "train_rows": len(y_tr), "test_rows": len(y_te)})
    pd.DataFrame(rows).to_csv(TABLES / "datasets_multiclass.csv", index=False)
    return pd.DataFrame(rows)


def load_mc(name):
    f = np.load(DATA_DIR / f"{name}.npz")
    return f["X_train"], f["y_train"], f["X_test"], f["y_test"]


# ---------------------------------------------------------------- noise
def reference_probs_mc(X_tr, y_tr, seed):
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=seed))
    return lr.fit(X_tr, y_tr).predict_proba(X_tr)                 # shape (n, K)


def inject_noise_mc(noise_type, y, rho, rng, K, P_ref=None):
    """Flip exactly round(rho * n) labels.
    symmetric  -> a uniformly random OTHER class
    pair_flip  -> class k becomes k + 1 (mod K): the standard class-conditional multiclass noise
    boundary   -> rows chosen with chance proportional to 1 - (reference top-class probability);
                  each gets the reference model's most likely class other than its true one"""
    y = y.copy()
    if rho == 0:
        return y
    k = int(round(rho * len(y)))
    if noise_type == "symmetric":
        idx = rng.choice(len(y), k, replace=False)
        y[idx] = (y[idx] + rng.integers(1, K, size=k)) % K
    elif noise_type == "pair_flip":
        idx = rng.choice(len(y), k, replace=False)
        y[idx] = (y[idx] + 1) % K
    elif noise_type == "boundary":
        w = 1.0 - P_ref.max(axis=1) + 1e-6
        idx = rng.choice(len(y), k, replace=False, p=w / w.sum())
        p = P_ref[idx].copy()
        p[np.arange(k), y[idx]] = -1.0                             # exclude the true class
        y[idx] = p.argmax(axis=1)
    else:
        raise ValueError(noise_type)
    return y


# ---------------------------------------------------------------- one condition
def _round(r, X_tr, y_tr, X_te, model, noise_type, rho, seed, K, P_ref):
    boot = np.random.default_rng([seed, r, 0]).integers(0, len(y_tr), len(y_tr))
    y_clean = y_tr[boot]
    y_noisy = inject_noise_mc(noise_type, y_clean, rho, np.random.default_rng([seed, r, 1]), K,
                              None if P_ref is None else P_ref[boot])
    m = make_model(model, seed=r)
    m.fit(X_tr[boot], y_noisy)
    return m.predict(X_te).astype(np.int16), float(np.mean(y_noisy != y_clean))


def run_condition_mc(X_tr, y_tr, X_te, model, noise_type, rho, K, P_ref, seed=42, R=R, n_jobs=-1):
    out = Parallel(n_jobs=n_jobs)(delayed(_round)(r, X_tr, y_tr, X_te, model, noise_type, rho, seed, K, P_ref)
                                  for r in range(R))
    return np.vstack([o[0] for o in out]), {"flip_rate": float(np.mean([o[1] for o in out]))}


# ---------------------------------------------------------------- decomposition
def decompose_mc(P, y, K):
    """Domingos (2000) 0-1 decomposition for K classes (clean test labels, noise term 0).
    Per test point: main = most frequent prediction, B = [main != y], V = P(pred != main).
      unbiased point (B = 0): E[L] = V
      biased point   (B = 1): E[L] = 1 - P(pred = y)
    So, on biased points, variance only lowers the error when a disagreeing model hits the
    TRUE class: the helpful part is P(pred = y) <= V. With 2 classes it equals V exactly.
        error = bias + var_unbiased - var_biased_helpful"""
    P, y = np.asarray(P), np.asarray(y)
    counts = np.stack([(P == k).sum(axis=0) for k in range(K)])     # (K, n_test)
    main = counts.argmax(axis=0)
    bias = (main != y).astype(float)
    var = (P != main).mean(axis=0)
    hit = (P == y).mean(axis=0)
    out = {"error": float((P != y).mean()), "bias": float(bias.mean()),
           "var_unbiased": float((var * (1 - bias)).mean()),
           "var_biased": float((var * bias).mean()),
           "var_biased_helpful": float((hit * bias).mean())}
    out["net_var"] = out["var_unbiased"] - out["var_biased_helpful"]
    out["identity_gap"] = out["error"] - (out["bias"] + out["net_var"])
    return out


# ---------------------------------------------------------------- runner
def conditions():
    yield "none", 0.0
    for nt in NOISE_TYPES_MC:
        for rho in RATES:
            yield nt, rho


def run_all(datasets=None):
    seed = json.loads((BASE / "config.json").read_text())["SEED"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "progress.log"

    def log(msg):
        line = f"{datetime.now():%H:%M:%S}  {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    table = TABLES / "datasets_multiclass.csv"
    if not table.exists():
        log("preparing multiclass datasets")
        select_and_prepare(seed)
    info = pd.read_csv(table).sort_values("rows")
    names = datasets or info.dataset.tolist()
    log(f"START rq5: {len(names)} datasets x {len(MODEL_NAMES)} models x 10 conditions, R = {R}")
    for name in names:
        out_csv = OUT_DIR / f"{name}.csv"
        done = set()
        if out_csv.exists():
            prev = pd.read_csv(out_csv)
            done = set(zip(prev.model, prev.noise_type, prev.rho.round(4)))
        X_tr, y_tr, X_te, y_te = load_mc(name)
        K = int(max(y_tr.max(), y_te.max()) + 1)
        P_ref = reference_probs_mc(X_tr, y_tr, seed)
        for model in MODEL_NAMES:
            for nt, rho in conditions():
                if (model, nt, round(rho, 4)) in done:
                    continue
                t = time.perf_counter()
                P, stats = run_condition_mc(X_tr, y_tr, X_te, model, nt, rho, K, P_ref, seed=seed)
                row = {"dataset": name, "classes": K, "model": model, "family": MODEL_FAMILY[model],
                       "noise_type": nt, "rho": rho, "R": R, "n_train": len(y_tr), "n_test": len(y_te),
                       **decompose_mc(P, y_te, K), **stats,
                       "seconds": round(time.perf_counter() - t, 2),
                       "finished_at": datetime.now().isoformat(timespec="seconds")}
                pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
                log(f"{name:22s} K={K:2d} {model:8s} {nt:9s} {rho:.2f}  error {row['error']:.4f}  "
                    f"bias {row['bias']:.4f}  net_var {row['net_var']:+.4f}  ({row['seconds']:.1f}s)")
    log("DONE rq5")


if __name__ == "__main__":
    run_all(sys.argv[1:] or None)
