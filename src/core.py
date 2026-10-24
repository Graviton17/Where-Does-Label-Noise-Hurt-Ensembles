"""Core building blocks for the label-noise / bias-variance study.

Every experiment notebook (Phases 4-6) imports from this file, so all
experiments use exactly the same models, noise injection and decomposition.

Contents
    load_dataset      read a prepared dataset from data/<name>.npz
    make_model        the six models, with frozen hyperparameters
    reference_probs   clean logistic-regression probabilities (for boundary noise)
    symmetric_noise, one_sided_noise, boundary_noise, inject_noise
    run_condition     train one model R times under one noise setting -> prediction matrix P
    decompose         Domingos 0-1 bias-variance decomposition of P
"""
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              GradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

MODEL_NAMES = ["Tree", "Bagging", "RF", "AdaBoost", "GB", "XGB"]
# Phase 6.2 (RQ4): regularised versions of four base models
REG_MODELS = {"GB_reg": "GB", "XGB_reg": "XGB", "AdaBoost_reg": "AdaBoost", "Bagging_small": "Bagging"}
# Phase 6.3: capacity control (tree depth matched across families)
CAPACITY_MODELS = {"Bagging_d3": "Bagging", "RF_d3": "RF", "AdaBoost_d3": "AdaBoost",
                   "XGB_d3": "XGB", "GB_d8": "GB", "XGB_d8": "XGB"}
TREE_DEPTH = {"Tree": None, "Bagging": None, "RF": None, "AdaBoost": 1, "GB": 3, "XGB": 4,
              "Bagging_d3": 3, "RF_d3": 3, "AdaBoost_d3": 3, "XGB_d3": 3, "GB_d8": 8, "XGB_d8": 8,
              "GB_reg": 3, "XGB_reg": 4, "AdaBoost_reg": 1, "Bagging_small": None}
MODEL_FAMILY = {"Tree": "single", "Bagging": "parallel", "RF": "parallel",
                "AdaBoost": "sequential", "GB": "sequential", "XGB": "sequential"}
MODEL_FAMILY.update({v: MODEL_FAMILY[b] for v, b in {**REG_MODELS, **CAPACITY_MODELS}.items()})
NOISE_TYPES = ["symmetric", "one_sided", "boundary"]


# ---------------------------------------------------------------- data
def load_dataset(name, data_dir):
    f = np.load(Path(data_dir) / f"{name}.npz", allow_pickle=False)
    return f["X_train"], f["y_train"], f["X_test"], f["y_test"]


# ---------------------------------------------------------------- models
def make_model(name, seed):
    """A fresh, untrained model. Hyperparameters are fixed here and never
    changed between noise conditions, so differences come from the noise.

    Each model uses one CPU thread (n_jobs=1); run_condition runs several
    rounds in parallel instead, which is faster for small datasets."""
    if name == "Tree":
        return DecisionTreeClassifier(random_state=seed)
    if name == "Bagging":
        return BaggingClassifier(DecisionTreeClassifier(), n_estimators=100,
                                 n_jobs=1, random_state=seed)
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=seed)
    if name == "AdaBoost":
        return AdaBoostClassifier(n_estimators=100, random_state=seed)
    if name == "GB":
        return GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                          max_depth=3, random_state=seed)
    if name == "XGB":
        return _xgb(max_depth=4, seed=seed)

    # ---- Phase 6.2: regularised versions (RQ4)
    if name == "GB_reg":
        return GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                          subsample=0.8, n_iter_no_change=10,
                                          validation_fraction=0.1, random_state=seed)
    if name == "XGB_reg":
        return EarlyStoppingXGB(seed=seed)
    if name == "AdaBoost_reg":
        return AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=seed)
    if name == "Bagging_small":
        return BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, max_samples=0.2,
                                 n_jobs=1, random_state=seed)

    # ---- Phase 6.3: capacity control (matched tree depth)
    if name == "Bagging_d3":
        return BaggingClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=100,
                                 n_jobs=1, random_state=seed)
    if name == "RF_d3":
        return RandomForestClassifier(n_estimators=100, max_depth=3, n_jobs=1, random_state=seed)
    if name == "AdaBoost_d3":
        return AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=100,
                                  random_state=seed)
    if name == "XGB_d3":
        return _xgb(max_depth=3, seed=seed)
    if name == "GB_d8":
        return GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=8,
                                          random_state=seed)
    if name == "XGB_d8":
        return _xgb(max_depth=8, seed=seed)
    raise ValueError(f"unknown model {name!r}")


def _xgb(max_depth, seed, **kw):
    params = dict(n_estimators=100, max_depth=max_depth, learning_rate=0.1, tree_method="hist",
                  n_jobs=1, random_state=seed, eval_metric="logloss")
    params.update(kw)
    return XGBClassifier(**params)


class EarlyStoppingXGB:
    """XGBoost with early stopping on a 10 % validation split taken from the
    (noisy) training data, as a practitioner would do: they cannot know which
    labels are wrong, so the validation labels are noisy too."""

    def __init__(self, seed, val_fraction=0.1, patience=10):
        self.seed, self.val_fraction, self.patience = seed, val_fraction, patience

    def fit(self, X, y):
        from sklearn.model_selection import train_test_split
        X_fit, X_val, y_fit, y_val = train_test_split(
            X, y, test_size=self.val_fraction, stratify=y, random_state=self.seed)
        self.model_ = _xgb(max_depth=4, seed=self.seed, n_estimators=300, learning_rate=0.05,
                           subsample=0.8, early_stopping_rounds=self.patience)
        self.model_.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def reference_probs(X_train, y_train, seed=0):
    """P(class 1) for every training row from a logistic regression fitted
    on the CLEAN training labels. Points near 0.5 are near the decision
    boundary; boundary noise flips those most often."""
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=seed))
    lr.fit(X_train, y_train)
    return lr.predict_proba(X_train)[:, 1]


# ---------------------------------------------------------------- noise
def symmetric_noise(y, rho, rng):
    """Flip exactly round(rho * n) labels chosen uniformly at random."""
    y = y.copy()
    k = int(round(rho * len(y)))
    idx = rng.choice(len(y), k, replace=False)
    y[idx] = 1 - y[idx]
    return y


def one_sided_noise(y, rho, rng):
    """Flip exactly round(rho * n_positive) labels of class 1 to class 0.
    Class 1 is the minority class (Phase 2), so this models missed positives."""
    y = y.copy()
    pos = np.flatnonzero(y == 1)
    k = int(round(rho * len(pos)))
    idx = rng.choice(pos, k, replace=False)
    y[idx] = 0
    return y


def boundary_noise(y, rho, rng, p_ref):
    """Flip exactly round(rho * n) labels, choosing each point with chance
    proportional to how close the reference model is to 50/50 on it."""
    y = y.copy()
    k = int(round(rho * len(y)))
    w = 1.0 - 2.0 * np.abs(p_ref - 0.5) + 1e-6      # 1 at the boundary, ~0 far away
    idx = rng.choice(len(y), k, replace=False, p=w / w.sum())
    y[idx] = 1 - y[idx]
    return y


def inject_noise(noise_type, y, rho, rng, p_ref=None):
    if rho == 0:
        return y.copy()
    if noise_type == "symmetric":
        return symmetric_noise(y, rho, rng)
    if noise_type == "one_sided":
        return one_sided_noise(y, rho, rng)
    if noise_type == "boundary":
        return boundary_noise(y, rho, rng, p_ref)
    raise ValueError(f"unknown noise type {noise_type!r}")


# ---------------------------------------------------------------- one experiment condition
def _one_round(r, X_tr, y_tr, X_te, model_name, noise_type, rho, seed, p_ref):
    # Common random numbers: the bootstrap sample and the noise mask depend only on
    # (seed, round), never on the model. Every model therefore sees exactly the same
    # training samples and the same flipped labels, so model comparisons are fair.
    boot_rng = np.random.default_rng([seed, r, 0])
    noise_rng = np.random.default_rng([seed, r, 1])

    idx = boot_rng.integers(0, len(y_tr), len(y_tr))          # bootstrap sample
    y_clean = y_tr[idx]
    y_noisy = inject_noise(noise_type, y_clean, rho, noise_rng,
                           None if p_ref is None else p_ref[idx])

    model = make_model(model_name, seed=r)
    model.fit(X_tr[idx], y_noisy)
    pred = model.predict(X_te).astype(np.int8)
    stats = {"flip_rate": float(np.mean(y_noisy != y_clean)),
             "pos_share_noisy": float(np.mean(y_noisy))}
    return pred, stats


def run_condition(X_tr, y_tr, X_te, model_name, noise_type, rho,
                  R=51, seed=42, p_ref=None, n_jobs=-1):
    """Train `model_name` R times, each time on a fresh bootstrap sample of the
    training pool with fresh label noise, and predict the clean test set.

    Returns
        P      int8 array (R, n_test): row r = predictions of the round-r model
        stats  dict: average realised flip rate and class-1 share after noise
    """
    if noise_type == "boundary" and rho > 0 and p_ref is None:
        raise ValueError("boundary noise needs p_ref = reference_probs(X_tr, y_tr)")
    out = Parallel(n_jobs=n_jobs)(
        delayed(_one_round)(r, X_tr, y_tr, X_te, model_name, noise_type, rho, seed, p_ref)
        for r in range(R))
    P = np.vstack([o[0] for o in out])
    stats = {k: float(np.mean([o[1][k] for o in out])) for k in out[0][1]}
    return P, stats


# ---------------------------------------------------------------- RQ3: sources of variance
DESIGNS = ["model_only", "noise_only", "sampling_only"]


def _one_round_design(r, design, X_tr, y_tr, X_te, model_name, noise_type, rho, seed, p_ref):
    """One round of an RQ3 design. Each design re-draws only some sources of
    randomness; the others are held fixed across all R rounds:
        model_only     fixed sample (whole pool), fixed noise mask, new model seed
        noise_only     fixed sample (whole pool), NEW noise mask,   new model seed
        sampling_only  NEW bootstrap sample,      fixed noise mask, new model seed
    The fixed noise mask is drawn once on the whole training pool (round-independent)."""
    n = len(y_tr)
    fixed_rng = np.random.default_rng([seed, 999_999, 1])            # same for every round
    y_fixed_noisy = inject_noise(noise_type, y_tr, rho, fixed_rng, p_ref)
    if design == "model_only":
        idx, y_fit = np.arange(n), y_fixed_noisy
    elif design == "noise_only":
        idx = np.arange(n)
        y_fit = inject_noise(noise_type, y_tr, rho, np.random.default_rng([seed, r, 1]), p_ref)
    elif design == "sampling_only":
        idx = np.random.default_rng([seed, r, 0]).integers(0, n, n)
        y_fit = y_fixed_noisy[idx]
    else:
        raise ValueError(f"unknown design {design!r}")
    model = make_model(model_name, seed=r)
    model.fit(X_tr[idx], y_fit)
    return model.predict(X_te).astype(np.int8), float(np.mean(y_fit != y_tr[idx]))


def run_design(X_tr, y_tr, X_te, design, model_name, noise_type, rho,
               R=51, seed=42, p_ref=None, n_jobs=-1):
    """Like run_condition, but only the sources of randomness named by `design` vary."""
    out = Parallel(n_jobs=n_jobs)(
        delayed(_one_round_design)(r, design, X_tr, y_tr, X_te, model_name, noise_type, rho, seed, p_ref)
        for r in range(R))
    return np.vstack([o[0] for o in out]), {"flip_rate": float(np.mean([o[1] for o in out]))}


# ---------------------------------------------------------------- decomposition
def decompose(P, y_test):
    """Domingos (2000) 0-1 loss decomposition for binary labels, measured
    against clean test labels (so the noise term is zero).

    Per test point x:
        main(x)  = majority prediction over the R models
        B(x)     = 1 if main(x) is wrong, else 0
        V(x)     = share of models that disagree with main(x)
        E[L(x)]  = B(x) + (1 - 2 B(x)) V(x)
    Averaged over the test set:
        error = bias + var_unbiased - var_biased
    """
    P = np.asarray(P)
    y_test = np.asarray(y_test)
    if P.shape[0] % 2 == 0:
        raise ValueError("use an odd number of rounds R so the majority vote never ties")
    main = (P.mean(axis=0) > 0.5).astype(int)
    bias = (main != y_test).astype(float)
    var = (P != main).mean(axis=0)
    out = {
        "error": float((P != y_test).mean()),
        "bias": float(bias.mean()),
        "var_unbiased": float((var * (1 - bias)).mean()),
        "var_biased": float((var * bias).mean()),
    }
    out["net_var"] = out["var_unbiased"] - out["var_biased"]
    out["identity_gap"] = out["error"] - (out["bias"] + out["net_var"])   # should be ~0
    return out
