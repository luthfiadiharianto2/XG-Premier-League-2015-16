"""Turn StatsBomb shot events into a modelling table.

Only information that is known *at the moment the shot is struck* is used as a
feature. Post-shot information (end_location, deflected, outcome, saved_to_post,
StatsBomb's own xG) is either not extracted at all or kept strictly as a label /
benchmark column that the models never see.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import config

# ----------------------------------------------------------- feature catalogue
GEOMETRY = ["distance", "angle"]
LATERAL = ["lateral_offset"]
SHOT_FLAGS = ["under_pressure", "first_time", "one_on_one", "aerial_won",
              "follows_dribble", "open_goal"]
CATEGORICAL = ["body_part", "shot_type", "technique", "play_pattern"]
FF_NUMERIC = ["gk_dist_to_goal", "defenders_in_cone", "nearest_opp_dist"]
FF_FLAGS = ["gk_in_cone", "has_freeze_frame"]

ALL_FEATURES = GEOMETRY + LATERAL + SHOT_FLAGS + CATEGORICAL + FF_NUMERIC + FF_FLAGS

# Columns that exist in the shot table but must never be used as features.
LABEL = "is_goal"
NON_FEATURE_COLUMNS = {"is_goal", "outcome", "statsbomb_xg", "xg"}
# Post-shot StatsBomb fields that are deliberately not extracted (data leakage).
LEAKY_STATSBOMB_FIELDS = {"end_location", "deflected", "outcome", "saved_to_post",
                          "saved_off_target", "redirect", "statsbomb_xg"}

PENALTY = "Penalty"


# ------------------------------------------------------------------ geometry
def shot_geometry(x: float, y: float) -> dict:
    """Distance (yards) to goal centre, open goal-mouth angle (radians), |y-40|."""
    dx = config.GOAL_X - x
    dist = math.hypot(dx, config.GOAL_CENTER_Y - y)
    a1 = math.atan2(config.POST_HIGH_Y - y, dx)
    a2 = math.atan2(config.POST_LOW_Y - y, dx)
    angle = abs(a1 - a2)
    if angle > math.pi:  # only possible for shots from behind the goal line
        angle = 2 * math.pi - angle
    return {"distance": dist, "angle": angle,
            "lateral_offset": abs(y - config.GOAL_CENTER_Y)}


def _in_triangle(px, py, a, b, c) -> np.ndarray:
    """Vectorised point-in-triangle test (boundary counts as inside)."""
    def cross(p1x, p1y, p2, p3):
        return (p1x - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1y - p3[1])
    d1 = cross(px, py, a, b)
    d2 = cross(px, py, b, c)
    d3 = cross(px, py, c, a)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)


def freeze_frame_features(x: float, y: float, frame: list[dict] | None) -> dict:
    """Features from the 360-style snapshot StatsBomb attaches to each shot.

    gk_dist_to_goal   distance of the defending keeper from the goal centre
    gk_in_cone        1 if the keeper stands inside the shooter->posts triangle
    defenders_in_cone outfield opponents inside that triangle
    nearest_opp_dist  distance from the shooter to the closest opponent
    has_freeze_frame  1 if a frame exists (otherwise the others are NaN and get
                      imputed inside the model pipeline, fold by fold)
    """
    out = {"gk_dist_to_goal": np.nan, "gk_in_cone": np.nan, "defenders_in_cone": np.nan,
           "nearest_opp_dist": np.nan, "has_freeze_frame": 0}
    if not frame:
        return out
    opp = [p for p in frame if not p.get("teammate", False) and p.get("location")]
    out["has_freeze_frame"] = 1
    shooter = (x, y)
    post_a = (config.GOAL_X, config.POST_LOW_Y)
    post_b = (config.GOAL_X, config.POST_HIGH_Y)
    if not opp:
        out["defenders_in_cone"] = 0
        return out
    is_gk = np.array([p.get("position", {}).get("name") == "Goalkeeper" for p in opp])
    loc = np.array([p["location"][:2] for p in opp], dtype=float)
    inside = _in_triangle(loc[:, 0], loc[:, 1], shooter, post_a, post_b)
    out["defenders_in_cone"] = int(inside[~is_gk].sum())
    out["nearest_opp_dist"] = float(np.min(np.hypot(loc[:, 0] - x, loc[:, 1] - y)))
    if is_gk.any():
        gk = loc[is_gk][0]
        out["gk_dist_to_goal"] = float(math.hypot(config.GOAL_X - gk[0], config.GOAL_CENTER_Y - gk[1]))
        out["gk_in_cone"] = int(inside[is_gk][0])
    return out


def _body_part(name: str | None) -> str:
    # Left vs right foot is not informative without knowing the player's
    # stronger foot, so both collapse to "Foot".
    if name in ("Right Foot", "Left Foot"):
        return "Foot"
    if name == "Head":
        return "Head"
    return "Other"


# ---------------------------------------------------------------- extraction
def extract_shots(events: list[dict], match: dict) -> list[dict]:
    """All shots of one match, excluding penalty shoot-outs (period 5)."""
    home, away = match["home_team"], match["away_team"]
    rows = []
    for ev in events:
        if ev.get("type", {}).get("name") != "Shot" or ev.get("period") == 5:
            continue
        shot = ev.get("shot", {})
        loc = ev.get("location")
        if not loc:
            continue
        x, y = float(loc[0]), float(loc[1])
        team = ev.get("team", {}).get("name")
        outcome = shot.get("outcome", {}).get("name", "Unknown")
        row = {
            "match_id": int(match["match_id"]),
            "event_id": ev.get("id"),
            "period": ev.get("period"),
            "minute": ev.get("minute"),
            "second": ev.get("second"),
            "team": team,
            "opponent": away if team == home else home,
            "player": ev.get("player", {}).get("name", "Unknown"),
            "player_id": ev.get("player", {}).get("id", -1),
            "x": x,
            "y": y,
            "shot_type": shot.get("type", {}).get("name", "Unknown"),
            "body_part": _body_part(shot.get("body_part", {}).get("name")),
            "technique": shot.get("technique", {}).get("name", "Unknown"),
            "play_pattern": ev.get("play_pattern", {}).get("name", "Unknown"),
            "under_pressure": int(bool(ev.get("under_pressure", False))),
            **{f: int(bool(shot.get(f, False))) for f in SHOT_FLAGS if f != "under_pressure"},
            "outcome": outcome,
            "is_goal": int(outcome == "Goal"),
            "statsbomb_xg": shot.get("statsbomb_xg", np.nan),
        }
        row.update(shot_geometry(x, y))
        row.update(freeze_frame_features(x, y, shot.get("freeze_frame")))
        rows.append(row)
    return rows


def shots_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["statsbomb_xg"] = pd.to_numeric(df["statsbomb_xg"], errors="coerce")
    return df.reset_index(drop=True)


def is_penalty(df: pd.DataFrame) -> pd.Series:
    return df["shot_type"] == PENALTY


def penalty_conversion(df: pd.DataFrame) -> tuple[float, int, int]:
    """Observed penalty conversion in the sample -> used as penalty xG."""
    pens = df[is_penalty(df)]
    n, g = len(pens), int(pens["is_goal"].sum())
    return (g / n if n else float("nan")), g, n


def feature_correlations(df: pd.DataFrame, threshold: float = 0.8) -> tuple[pd.DataFrame, list[tuple]]:
    """Pearson correlations between numeric features, plus pairs above threshold."""
    cols = [c for c in GEOMETRY + LATERAL + FF_NUMERIC + SHOT_FLAGS + FF_FLAGS if c in df]
    corr = df[cols].astype(float).corr()
    flagged = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= threshold:
                flagged.append((a, b, float(r)))
    return corr, flagged
