"""Scoring, calibration and uncertainty for out-of-fold xG predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import config

EPS = 1e-6


def _clip(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """Fit y ~ Bernoulli(expit(a + b*logit(p))). Perfect calibration: a=0, b=1.

    b < 1 means predictions are too extreme (over-confident), b > 1 too timid.
    """
    y = np.asarray(y, dtype=float)
    z = logit(_clip(p))

    def nll(theta):
        q = _clip(expit(theta[0] + theta[1] * z))
        return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))

    res = minimize(nll, x0=np.array([0.0, 1.0]), method="BFGS")
    return float(res.x[0]), float(res.x[1])


def score(y, p) -> dict:
    y = np.asarray(y, dtype=int)
    p = _clip(p)
    intercept, slope = calibration_slope_intercept(y, p)
    goals = int(y.sum())
    return {
        "n_shots": int(len(y)),
        "goals": goals,
        "xg_total": float(p.sum()),
        "xg_minus_goals": float(p.sum() - goals),
        "auc": float(roc_auc_score(y, p)) if 0 < goals < len(y) else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "cal_intercept": intercept,
        "cal_slope": slope,
    }


def compare_models(y, preds: dict[str, np.ndarray], labels: dict[str, str],
                   reference: np.ndarray | None = None) -> pd.DataFrame:
    """One row per model. The StatsBomb row is a reference, not a competitor.

    All rows are scored on the same shots (those with a StatsBomb xG value when a
    reference is given) so the numbers are directly comparable.
    """
    y = np.asarray(y)
    mask = np.ones(len(y), dtype=bool)
    if reference is not None:
        mask = ~np.isnan(np.asarray(reference, dtype=float))
    rows = []
    for key, p in preds.items():
        rows.append({"key": key, "model": labels[key], "is_reference": False,
                     **score(y[mask], np.asarray(p)[mask])})
    if reference is not None:
        rows.append({"key": "statsbomb", "model": "StatsBomb xG (reference)",
                     "is_reference": True, **score(y[mask], np.asarray(reference)[mask])})
    return pd.DataFrame(rows)


def select_main(table: pd.DataFrame) -> str:
    """Main model = lowest out-of-fold log loss among *our* models."""
    ours = table[~table["is_reference"]]
    return str(ours.loc[ours["log_loss"].idxmin(), "key"])


def _wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1 + z ** 2 / n
    centre = (phat + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(centre - half), float(centre + half)


def calibration_table(y, p, n_bins: int = config.CALIBRATION_BINS) -> pd.DataFrame:
    """Quantile bins (equal number of shots per bin) with 95% Wilson intervals."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n_bins = max(1, min(n_bins, len(p)))
    ranks = pd.Series(p).rank(method="first")
    bins = pd.qcut(ranks, n_bins, labels=False)
    rows = []
    for b in range(n_bins):
        m = (bins == b).to_numpy()
        k, n = y[m].sum(), m.sum()
        lo, hi = _wilson(k, n)
        rows.append({"bin": b, "n": int(n), "mean_pred": float(p[m].mean()),
                     "obs_rate": float(k / n), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def _per_shot_log_loss(y, p) -> np.ndarray:
    p = _clip(p)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_bootstrap(y, p_model, p_base, groups, reps: int = config.BOOTSTRAP_REPS,
                     seed: int = config.RANDOM_STATE) -> dict:
    """Match-level (cluster) bootstrap of the log-loss difference model - base.

    Negative = the model is better. Resampling whole matches respects the fact
    that shots inside one match are not independent.
    """
    d = _per_shot_log_loss(y, p_model) - _per_shot_log_loss(y, p_base)
    frame = pd.DataFrame({"g": np.asarray(groups), "d": d})
    agg = frame.groupby("g")["d"].agg(["sum", "count"])
    sums, counts = agg["sum"].to_numpy(), agg["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(agg), size=(reps, len(agg)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return {"delta": float(d.mean()), "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5)),
            "p_better": float((boot < 0).mean())}
