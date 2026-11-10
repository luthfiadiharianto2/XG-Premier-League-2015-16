"""Team, player and form analysis built on out-of-fold xG."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

import config
from src import features as F


def poisson_outcome_probs(lam_home: float, lam_away: float,
                          max_goals: int = config.MAX_GOALS_POISSON) -> tuple[float, float, float]:
    """P(home win), P(draw), P(away win) with independent Poisson goal counts.

    Mass above ``max_goals`` is folded into the last cell so the grid sums to 1.
    """
    k = np.arange(max_goals + 1)
    ph = poisson.pmf(k, max(lam_home, 0.0))
    pa = poisson.pmf(k, max(lam_away, 0.0))
    ph[-1] += 1 - ph.sum()
    pa[-1] += 1 - pa.sum()
    grid = np.outer(ph, pa)
    return float(np.tril(grid, -1).sum()), float(np.trace(grid)), float(np.triu(grid, 1).sum())


def _points(gf: int, ga: int) -> int:
    return 3 if gf > ga else (1 if gf == ga else 0)


def match_table(matches: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """One row per match with xG for each side and expected points."""
    xg = shots.groupby(["match_id", "team"])["xg"].sum()
    rows = []
    for m in matches.itertuples(index=False):
        hxg = float(xg.get((m.match_id, m.home_team), 0.0))
        axg = float(xg.get((m.match_id, m.away_team), 0.0))
        pw, pd_, pl = poisson_outcome_probs(hxg, axg)
        rows.append({
            "match_id": m.match_id, "match_date": m.match_date,
            "home_team": m.home_team, "away_team": m.away_team,
            "home_score": m.home_score, "away_score": m.away_score,
            "home_xg": hxg, "away_xg": axg,
            "home_xpts": 3 * pw + pd_, "away_xpts": 3 * pl + pd_,
            "p_home": pw, "p_draw": pd_, "p_away": pl,
        })
    return pd.DataFrame(rows)


def team_match_long(mt: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match: one from each team's perspective."""
    home = pd.DataFrame({
        "match_id": mt.match_id, "match_date": mt.match_date, "team": mt.home_team,
        "opponent": mt.away_team, "venue": "H", "gf": mt.home_score, "ga": mt.away_score,
        "xg": mt.home_xg, "xga": mt.away_xg, "xpts": mt.home_xpts})
    away = pd.DataFrame({
        "match_id": mt.match_id, "match_date": mt.match_date, "team": mt.away_team,
        "opponent": mt.home_team, "venue": "A", "gf": mt.away_score, "ga": mt.home_score,
        "xg": mt.away_xg, "xga": mt.home_xg, "xpts": mt.away_xpts})
    long = pd.concat([home, away], ignore_index=True)
    long["pts"] = [_points(a, b) for a, b in zip(long.gf, long.ga)]
    return long.sort_values(["match_date", "match_id", "venue"]).reset_index(drop=True)


def team_table(long: pd.DataFrame) -> pd.DataFrame:
    g = long.groupby("team")
    t = pd.DataFrame({
        "matches": g.size(), "gf": g.gf.sum(), "ga": g.ga.sum(),
        "xg": g.xg.sum(), "xga": g.xga.sum(), "pts": g.pts.sum(), "xpts": g.xpts.sum(),
    })
    t["gd"] = t.gf - t.ga
    t["xgd"] = t.xg - t.xga
    for col in ["xg", "xga", "xgd"]:
        t[f"{col}_pm"] = t[col] / t.matches
    t["pts_minus_xpts"] = t.pts - t.xpts
    t = t.sort_values(["pts", "gd", "gf"], ascending=False)
    t["rank_pts"] = np.arange(1, len(t) + 1)
    t["rank_xpts"] = t.xpts.rank(ascending=False, method="min").astype(int)
    t = t.sort_values("xgd_pm", ascending=False)
    return t.reset_index()


def player_table(shots: pd.DataFrame, min_shots: int = 1) -> pd.DataFrame:
    """Non-penalty finishing: goals vs npxG with a Poisson-binomial z-score.

    z = (G - npxG) / sqrt(sum p(1-p)) treats each shot as an independent
    Bernoulli with the model's probability. |z| < 2 is consistent with luck.
    """
    np_shots = shots[~F.is_penalty(shots)].copy()
    np_shots["var"] = np_shots.xg * (1 - np_shots.xg)
    g = np_shots.groupby("player_id")
    t = pd.DataFrame({
        "player": g.player.agg(lambda s: s.mode().iat[0]),
        "team": g.team.agg(lambda s: s.mode().iat[0]),
        "shots": g.size(), "np_goals": g.is_goal.sum(), "npxg": g.xg.sum(), "var": g["var"].sum(),
    })
    t["diff"] = t.np_goals - t.npxg
    t["z"] = t["diff"] / np.sqrt(t["var"].clip(lower=1e-9))
    t["npxg_per_shot"] = t.npxg / t.shots
    t = t[t.shots >= min_shots].drop(columns="var")
    return t.sort_values("diff", ascending=False).reset_index(drop=True)


def team_form(long: pd.DataFrame, team: str, window: int = config.ROLLING_WINDOW) -> pd.DataFrame:
    """Match-by-match xG/xGA with a trailing rolling mean.

    The first window-1 points average fewer than ``window`` matches (min_periods=1).
    """
    f = long[long.team == team].sort_values(["match_date", "match_id"]).reset_index(drop=True)
    f["matchday"] = np.arange(1, len(f) + 1)
    for col in ["xg", "xga", "gf", "ga", "pts", "xpts"]:
        f[f"{col}_roll"] = f[col].rolling(window, min_periods=1).mean()
    return f
