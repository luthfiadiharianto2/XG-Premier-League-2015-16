"""Unit tests for geometry, features, loader, evaluation and analysis helpers."""
import gzip
import json
import math

import numpy as np
import pandas as pd
import pytest
import requests

from src import analysis as A
from src import evaluate as E
from src import features as F
from src import model as M
from src.data_loader import DataNotAvailableError, StatsBombClient, resolve_competition


# ------------------------------------------------------------------ geometry
def test_geometry_central_shot():
    g = F.shot_geometry(108, 40)
    assert g["distance"] == pytest.approx(12.0)
    assert g["angle"] == pytest.approx(2 * math.atan(4 / 12))
    assert g["lateral_offset"] == 0


def test_angle_shrinks_with_distance_and_width():
    assert F.shot_geometry(100, 40)["angle"] < F.shot_geometry(110, 40)["angle"]
    assert F.shot_geometry(110, 10)["angle"] < F.shot_geometry(110, 40)["angle"]


def test_freeze_frame_features():
    frame = [
        {"location": [119, 40], "teammate": False, "position": {"name": "Goalkeeper"}},
        {"location": [112, 40], "teammate": False, "position": {"name": "Centre Back"}},  # in cone
        {"location": [112, 70], "teammate": False, "position": {"name": "Left Back"}},    # outside
        {"location": [109, 40], "teammate": True, "position": {"name": "Center Forward"}},
    ]
    f = F.freeze_frame_features(108, 40, frame)
    assert f["has_freeze_frame"] == 1
    assert f["gk_dist_to_goal"] == pytest.approx(1.0)
    assert f["gk_in_cone"] == 1
    assert f["defenders_in_cone"] == 1
    assert f["nearest_opp_dist"] == pytest.approx(4.0)  # teammate at 1 yd is ignored


def test_missing_freeze_frame_is_nan_with_flag():
    f = F.freeze_frame_features(100, 40, None)
    assert f["has_freeze_frame"] == 0
    assert all(np.isnan(f[k]) for k in ["gk_dist_to_goal", "defenders_in_cone", "nearest_opp_dist"])


def _shot(period=1, kind="Open Play", outcome="Goal"):
    return {"id": "a", "period": period, "minute": 10, "second": 0,
            "type": {"name": "Shot"}, "team": {"name": "H"}, "player": {"id": 1, "name": "P"},
            "location": [110, 40], "play_pattern": {"name": "Regular Play"},
            "shot": {"type": {"name": kind}, "outcome": {"name": outcome},
                     "body_part": {"name": "Left Foot"}, "technique": {"name": "Normal"},
                     "end_location": [120, 40, 1], "statsbomb_xg": 0.3, "first_time": True}}


def test_extract_drops_shootouts_and_maps_fields():
    match = {"match_id": 1, "home_team": "H", "away_team": "A"}
    rows = F.extract_shots([_shot(), _shot(period=5, kind="Penalty")], match)
    assert len(rows) == 1
    r = rows[0]
    assert r["opponent"] == "A" and r["body_part"] == "Foot" and r["first_time"] == 1
    assert r["is_goal"] == 1 and r["one_on_one"] == 0


def test_no_post_shot_information_in_any_model():
    extracted = set(F.extract_shots([_shot()], {"match_id": 1, "home_team": "H", "away_team": "A"})[0])
    assert "end_location" not in extracted and "deflected" not in extracted
    for spec in M.MODEL_SPECS:
        cols = set(spec.columns)
        assert not cols & F.NON_FEATURE_COLUMNS, spec.key
        assert not cols & F.LEAKY_STATSBOMB_FIELDS, spec.key
        assert cols <= set(F.ALL_FEATURES), spec.key


def test_ablation_feature_sets_are_nested():
    cols = [set(s.columns) for s in M.MODEL_SPECS]
    assert cols[0] == {"distance", "angle"}
    assert cols[0] < cols[1] < cols[2] and cols[2] == cols[3]
    assert {"distance", "log_distance"} - cols[3] == {"log_distance"}  # no redundant transforms


def test_penalty_conversion():
    df = pd.DataFrame({"shot_type": ["Penalty"] * 4 + ["Open Play"], "is_goal": [1, 1, 1, 0, 1]})
    rate, goals, n = F.penalty_conversion(df)
    assert (rate, goals, n) == (0.75, 3, 4)


# -------------------------------------------------------------------- loader
class FlakySession:
    """Fails twice with a 503, then succeeds."""

    def __init__(self, payload, failures=2):
        self.payload, self.failures, self.calls = payload, failures, 0

    def get(self, url, timeout=None):
        self.calls += 1
        resp = requests.Response()
        if self.calls <= self.failures:
            resp.status_code = 503
            resp._content = b"busy"
        else:
            resp.status_code = 200
            resp._content = json.dumps(self.payload).encode()
        return resp


def test_retry_then_cache(tmp_path):
    session = FlakySession([{"ok": 1}])
    c = StatsBombClient(base_url="https://example.invalid", cache_dir=tmp_path, backoff=0,
                        retries=3, session=session)
    assert c.get_json("competitions.json") == [{"ok": 1}]
    assert session.calls == 3
    with gzip.open(tmp_path / "competitions.json.gz", "rt") as fh:
        assert json.load(fh) == [{"ok": 1}]
    assert c.get_json("competitions.json") == [{"ok": 1}]
    assert session.calls == 3  # second read served from disk


def test_retry_gives_up(tmp_path):
    c = StatsBombClient(base_url="https://example.invalid", cache_dir=tmp_path, backoff=0,
                        retries=2, session=FlakySession([], failures=99))
    with pytest.raises(ConnectionError):
        c.get_json("competitions.json")


def test_offline_without_cache_raises(tmp_path):
    with pytest.raises(DataNotAvailableError):
        StatsBombClient(cache_dir=tmp_path, offline=True).get_json("competitions.json")


COMPS = [{"competition_id": 2, "season_id": 27, "competition_name": "Premier League",
          "season_name": "2015/2016"},
         {"competition_id": 2, "season_id": 44, "competition_name": "Premier League",
          "season_name": "2003/2004"}]


@pytest.mark.parametrize("season", ["2015/2016", "2015/16", "2015-16", "27"])
def test_resolve_season_variants(season):
    assert resolve_competition(COMPS, "premier league", season)["season_id"] == 27


def test_resolve_unknown_season_lists_alternatives():
    with pytest.raises(DataNotAvailableError, match="2015/2016"):
        resolve_competition(COMPS, "Premier League", "2025/26")


# ---------------------------------------------------------------- evaluation
def test_metrics_and_calibration_on_known_probabilities():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.6, 20000)
    y = (rng.random(p.size) < p).astype(int)
    s = E.score(y, p)
    assert s["cal_slope"] == pytest.approx(1.0, abs=0.1)
    assert s["cal_intercept"] == pytest.approx(0.0, abs=0.1)
    assert abs(s["xg_total"] - s["goals"]) < 4 * math.sqrt((p * (1 - p)).sum())
    cal = E.calibration_table(y, p, 10)
    assert cal.n.sum() == p.size and cal.n.min() >= p.size // 10 - 1
    assert np.allclose(cal.mean_pred, cal.obs_rate, atol=0.03)


def test_calibration_slope_detects_overconfidence():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.5, 20000)
    y = (rng.random(p.size) < p).astype(int)
    from scipy.special import expit, logit
    assert E.score(y, expit(2 * logit(p)))["cal_slope"] < 0.7


def test_select_main_ignores_reference():
    t = pd.DataFrame({"key": ["a", "b", "statsbomb"], "log_loss": [0.3, 0.28, 0.1],
                      "is_reference": [False, False, True]})
    assert E.select_main(t) == "b"


def test_paired_bootstrap_sign():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.5, 5000)
    y = (rng.random(p.size) < p).astype(int)
    groups = np.repeat(np.arange(250), 20)
    b = E.paired_bootstrap(y, p, np.full_like(p, y.mean()), groups, reps=300)
    assert b["delta"] < 0 and b["ci_high"] < 0


def test_group_folds_never_split_a_match():
    groups = pd.Series(np.repeat(np.arange(40), 7))
    folds = M.make_folds(groups, 5)
    assert len(folds) == 5
    seen = np.zeros(len(groups), dtype=int)
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te])
        seen[te] += 1
    assert (seen == 1).all()


# ------------------------------------------------------------------ analysis
def test_poisson_probabilities():
    w, d, l = A.poisson_outcome_probs(1.3, 1.3)
    assert w + d + l == pytest.approx(1.0)
    assert w == pytest.approx(l)
    assert A.poisson_outcome_probs(0, 0) == pytest.approx((0.0, 1.0, 0.0))
    w2, _, l2 = A.poisson_outcome_probs(2.5, 0.4)
    assert w2 > 0.75 and l2 < 0.06


def test_xpts_bounds():
    for lh, la in [(0.2, 3.0), (1.0, 1.0), (2.8, 0.1)]:
        w, d, l = A.poisson_outcome_probs(lh, la)
        total = 3 * w + d + 3 * l + d
        assert 2.0 <= total <= 3.0


def test_player_zscore():
    shots = pd.DataFrame({"player_id": [1] * 4 + [2] * 2, "player": ["A"] * 4 + ["B"] * 2,
                          "team": ["T"] * 6, "shot_type": ["Open Play"] * 5 + ["Penalty"],
                          "is_goal": [1, 1, 0, 0, 0, 1], "xg": [0.5] * 4 + [0.2, 0.78]})
    t = A.player_table(shots).set_index("player")
    assert t.loc["A", "np_goals"] == 2 and t.loc["A", "npxg"] == pytest.approx(2.0)
    assert t.loc["A", "z"] == pytest.approx(0.0)
    assert t.loc["B", "shots"] == 1  # penalty excluded
    assert t.loc["B", "z"] == pytest.approx(-0.2 / math.sqrt(0.16))
