"""A small synthetic league in StatsBomb JSON format, for offline tests only.

Goal probability follows a KNOWN logistic formula (TRUE_COEFS), so tests can
check that the pipeline recovers sensible behaviour. Nothing generated here is
ever used for the numbers in the README.
"""
from __future__ import annotations

import gzip
import json
import math
import uuid
from itertools import permutations
from pathlib import Path

import numpy as np

COMP_ID, SEASON_ID = 9001, 9002
COMP_NAME, SEASON_NAME = "Synthetic League", "2030/2031"
TEAMS = ["Red FC", "Blue United", "Green Rovers", "Yellow Town", "Black City", "White Athletic"]

TRUE_COEFS = {"intercept": 1.0, "distance": -0.13, "angle": 1.2, "head": -0.9,
              "one_on_one": 0.9, "defenders_in_cone": -0.35}


def true_probability(distance, angle, head, one_on_one, defenders_in_cone):
    z = (TRUE_COEFS["intercept"] + TRUE_COEFS["distance"] * distance + TRUE_COEFS["angle"] * angle
         + TRUE_COEFS["head"] * head + TRUE_COEFS["one_on_one"] * one_on_one
         + TRUE_COEFS["defenders_in_cone"] * defenders_in_cone)
    return 1 / (1 + math.exp(-z))


def _angle(x, y):
    dx = 120 - x
    return abs(math.atan2(44 - y, dx) - math.atan2(36 - y, dx))


def _in_cone(px, py, x, y):
    if px < x or px > 120:
        return False
    t = (px - x) / (120 - x) if x < 120 else 1
    lo, hi = y + t * (36 - y), y + t * (44 - y)
    return lo <= py <= hi


def _shot_event(rng, team, player, minute, period, kind="Open Play"):
    if kind == "Penalty":
        x, y, head, one, n_def = 108.0, 40.0, 0, 0, 0
    else:
        x = float(rng.uniform(88, 118))
        y = float(np.clip(rng.normal(40, 9), 2, 78))
        head = int(rng.random() < 0.2)
        one = int(rng.random() < 0.08)
    # freeze frame: keeper near goal line + random defenders
    frame = [{"location": [float(rng.uniform(116, 119.5)), float(rng.normal(40, 1.5))],
              "teammate": False, "player": {"id": 1, "name": "GK"},
              "position": {"id": 1, "name": "Goalkeeper"}}]
    if kind != "Penalty":
        for _ in range(int(rng.integers(2, 7))):
            frame.append({"location": [float(rng.uniform(x, 120)), float(rng.uniform(25, 55))],
                          "teammate": False, "player": {"id": 2, "name": "D"},
                          "position": {"id": 3, "name": "Centre Back"}})
        frame.append({"location": [float(rng.uniform(95, 115)), float(rng.uniform(20, 60))],
                      "teammate": True, "player": {"id": 3, "name": "T"},
                      "position": {"id": 23, "name": "Center Forward"}})
        n_def = sum(_in_cone(p["location"][0], p["location"][1], x, y)
                    for p in frame if not p["teammate"] and p["position"]["name"] != "Goalkeeper")
        p_goal = true_probability(math.hypot(120 - x, 40 - y), _angle(x, y), head, one, n_def)
    else:
        p_goal = 0.78
    goal = bool(rng.random() < p_goal)
    shot = {
        "statsbomb_xg": round(p_goal, 4),
        "end_location": [120.0, 40.0, 1.0],
        "outcome": {"id": 97 if goal else 98, "name": "Goal" if goal else "Off T"},
        "type": {"id": 87 if kind == "Open Play" else 88, "name": kind},
        "body_part": {"id": 37 if head else 40, "name": "Head" if head else "Right Foot"},
        "technique": {"id": 93, "name": "Normal"},
        "freeze_frame": frame,
    }
    if kind != "Penalty" and one:
        shot["one_on_one"] = True
    ev = {"id": str(uuid.UUID(int=int(rng.integers(0, 2**63)))), "period": period,
          "minute": minute, "second": 0, "type": {"id": 16, "name": "Shot"},
          "team": {"id": 0, "name": team}, "player": {"id": player[0], "name": player[1]},
          "location": [x, y], "play_pattern": {"id": 1, "name": "Regular Play"}, "shot": shot}
    return ev, goal


def build_league(cache_dir: Path, seed: int = 7, rounds: int = 2, shots_per_team=(10, 16),
                 shootout_match: bool = True) -> dict:
    """Write competitions/matches/events (gzip, same layout as the real cache)."""
    rng = np.random.default_rng(seed)
    cache_dir = Path(cache_dir)
    players = {t: [(i * 100 + j, f"{t.split()[0]} Player {j}") for j in range(1, 8)]
               for i, t in enumerate(TEAMS, start=1)}
    fixtures = list(permutations(TEAMS, 2)) * (rounds // 2 or 1)
    matches, n_shootout = [], 0
    for k, (home, away) in enumerate(fixtures):
        mid = 10_000 + k
        events, score = [], {home: 0, away: 0}
        for team in (home, away):
            for _ in range(int(rng.integers(*shots_per_team))):
                ev, goal = _shot_event(rng, team, players[team][int(rng.integers(0, 7))],
                                       int(rng.integers(1, 90)), 1 if rng.random() < .5 else 2)
                events.append(ev)
                score[team] += goal
            if rng.random() < 0.15:
                ev, goal = _shot_event(rng, team, players[team][0], 60, 2, kind="Penalty")
                events.append(ev)
                score[team] += goal
        if shootout_match and k == 0:  # shoot-out kicks must be ignored
            for team in (home, away):
                for _ in range(5):
                    ev, _ = _shot_event(rng, team, players[team][1], 120, 5, kind="Penalty")
                    events.append(ev)
                    n_shootout += 1
        events.append({"id": "x", "period": 1, "minute": 0, "second": 0,
                       "type": {"id": 30, "name": "Pass"}, "team": {"id": 0, "name": home}})
        _write(cache_dir / "events" / f"{mid}.json.gz", events)
        matches.append({"match_id": mid, "match_date": f"2030-{8 + k // 20:02d}-{1 + k % 20:02d}",
                        "kick_off": "15:00:00.000", "match_week": k // 3 + 1,
                        "home_team": {"home_team_name": home}, "away_team": {"away_team_name": away},
                        "home_score": score[home], "away_score": score[away]})
    _write(cache_dir / "matches" / str(COMP_ID) / f"{SEASON_ID}.json.gz", matches)
    _write(cache_dir / "competitions.json.gz", [
        {"competition_id": COMP_ID, "season_id": SEASON_ID, "competition_name": COMP_NAME,
         "season_name": SEASON_NAME, "country_name": "Nowhere", "competition_gender": "male"}])
    return {"n_matches": len(matches), "n_shootout": n_shootout}


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(obj, fh)
