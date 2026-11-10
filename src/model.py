"""Four xG models, added one block of information at a time (ablation).

1. Logistic regression on distance + angle
2. + shot context (lateral offset, flags, body part, shot type, technique, play pattern)
3. + freeze-frame features (keeper and defender positions)
4. HistGradientBoosting on the same inputs as (3)

Penalties are not modelled: they get the observed penalty conversion rate.
All evaluation uses out-of-fold predictions from GroupKFold grouped by match,
so no shot is ever scored by a model that saw other shots from the same match.
Hyper-parameters are fixed up front (no tuning) to keep the CV estimate honest.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from src import features as F


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    kind: str = "logreg"  # "logreg" | "hgb"

    @property
    def columns(self) -> list[str]:
        return self.numeric + self.categorical


_CONTEXT_NUM = F.GEOMETRY + F.LATERAL + F.SHOT_FLAGS
_FULL_NUM = _CONTEXT_NUM + F.FF_NUMERIC + F.FF_FLAGS

MODEL_SPECS: list[ModelSpec] = [
    ModelSpec("m1_geometry", "1 · Logreg: distance + angle", numeric=list(F.GEOMETRY)),
    ModelSpec("m2_context", "2 · Logreg: + shot context",
              numeric=list(_CONTEXT_NUM), categorical=list(F.CATEGORICAL)),
    ModelSpec("m3_freeze", "3 · Logreg: + freeze-frame",
              numeric=list(_FULL_NUM), categorical=list(F.CATEGORICAL)),
    ModelSpec("m4_hgb", "4 · HistGradientBoosting (all features)",
              numeric=list(_FULL_NUM), categorical=list(F.CATEGORICAL), kind="hgb"),
]
SPEC_BY_KEY = {s.key: s for s in MODEL_SPECS}


def build_pipeline(spec: ModelSpec, random_state: int = config.RANDOM_STATE) -> Pipeline:
    transformers = []
    if spec.numeric:
        transformers.append(("num", SimpleImputer(strategy="median"), spec.numeric))
    if spec.categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                  min_frequency=30, sparse_output=False),
                             spec.categorical))
    pre = ColumnTransformer(transformers, verbose_feature_names_out=True)
    if spec.kind == "logreg":
        # Scaling *every* column (dummies included) makes the coefficients
        # comparable: each one is the change in log-odds per 1 SD of the input.
        return Pipeline([("pre", pre), ("scale", StandardScaler()),
                         ("clf", LogisticRegression(C=1.0, max_iter=5000))])
    if spec.kind == "hgb":
        return Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=250, max_leaf_nodes=15, min_samples_leaf=50,
            l2_regularization=1.0, early_stopping=False, random_state=random_state))])
    raise ValueError(f"Unknown model kind {spec.kind!r}")


def make_folds(groups: pd.Series, n_splits: int = config.N_FOLDS,
               random_state: int = config.RANDOM_STATE) -> list[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold by match. The same folds are reused for every model (paired comparison)."""
    n_groups = groups.nunique()
    n_splits = min(n_splits, n_groups)
    if n_splits < 2:
        raise ValueError("Need at least 2 matches for cross-validation.")
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    dummy = np.zeros(len(groups))
    return list(gkf.split(dummy, dummy, groups=groups.to_numpy()))


def oof_predict(df: pd.DataFrame, spec: ModelSpec, folds) -> np.ndarray:
    X, y = df[spec.columns], df[F.LABEL].to_numpy()
    oof = np.full(len(df), np.nan)
    for train_idx, test_idx in folds:
        pipe = build_pipeline(spec).fit(X.iloc[train_idx], y[train_idx])
        oof[test_idx] = pipe.predict_proba(X.iloc[test_idx])[:, 1]
    assert not np.isnan(oof).any(), "every shot must receive exactly one OOF prediction"
    return oof


def fit_full(df: pd.DataFrame, spec: ModelSpec) -> Pipeline:
    return build_pipeline(spec).fit(df[spec.columns], df[F.LABEL].to_numpy())


def _pretty(name: str) -> str:
    name = name.split("__", 1)[-1]
    for cat in F.CATEGORICAL:
        prefix = cat + "_"
        if name.startswith(prefix):
            level = name[len(prefix):]
            if level == "infrequent_sklearn":
                level = "rare (<30 shots, pooled)"
            return f"{cat} = {level}"
    return name


def logreg_coefficients(pipe: Pipeline) -> pd.DataFrame:
    """Coefficients on standardised inputs (log-odds change per 1 SD)."""
    names = pipe.named_steps["pre"].get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_.ravel()
    df = pd.DataFrame({"feature": [_pretty(n) for n in names], "coef": coefs})
    df["abs"] = df["coef"].abs()
    return df.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)
