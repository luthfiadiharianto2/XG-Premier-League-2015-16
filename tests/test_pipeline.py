"""End-to-end test: the full pipeline on a synthetic StatsBomb-format league.

The synthetic goal probabilities come from a known formula, so we can check
that the models learn the right shape. These numbers are NOT reported anywhere.
"""
import json

import numpy as np
import pytest

from synthetic import TEAMS, TRUE_COEFS


def test_shootouts_dropped_and_counts_consistent(pipeline_result):
    result, _, info = pipeline_result
    meta = result["meta"]
    assert meta["n_shootout_dropped"] == info["n_shootout"] > 0
    assert meta["n_matches"] == info["n_matches"]
    assert meta["n_shots"] == meta["n_np_shots"] + meta["n_penalties"]
    assert (result["shots"]["period"] != 5).all()


def test_every_model_has_one_oof_prediction_per_shot(pipeline_result):
    result, _, _ = pipeline_result
    folds = result["folds"]
    n = len(result["np_shots"])
    test_idx = np.concatenate([te for _, te in folds])
    assert sorted(test_idx) == list(range(n))
    groups = result["np_shots"].match_id.to_numpy()
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te])


def test_models_learn_signal_and_are_calibrated(pipeline_result):
    result, _, _ = pipeline_result
    t = result["models"].set_index("key")
    for key in ["m1_geometry", "m2_context", "m3_freeze", "m4_hgb"]:
        assert t.loc[key, "auc"] > 0.65, key
        # total xG within 15% of goals
        assert abs(t.loc[key, "xg_total"] / t.loc[key, "goals"] - 1) < 0.15, key
    # the true model includes freeze-frame information, so model 3 must beat model 1
    assert t.loc["m3_freeze", "log_loss"] < t.loc["m1_geometry", "log_loss"]
    # the generating probabilities are the best possible reference
    assert t.loc["statsbomb", "log_loss"] <= t.loc["m3_freeze", "log_loss"] + 0.01
    assert result["meta"]["main_model"] in t.index


def test_coefficient_signs_match_generating_formula(pipeline_result):
    result, _, _ = pipeline_result
    c = result["coefficients"].set_index("feature")["coef"]
    assert TRUE_COEFS["distance"] < 0 and c["distance"] < 0
    assert c["defenders_in_cone"] < 0
    assert c["body_part = Head"] < 0
    assert c["one_on_one"] > 0


def test_penalty_xg_is_observed_conversion(pipeline_result):
    result, _, _ = pipeline_result
    shots = result["shots"]
    pens = shots[shots.shot_type == "Penalty"]
    assert np.allclose(pens.xg, pens.is_goal.mean())
    assert result["meta"]["penalty_xg"] == pytest.approx(pens.is_goal.mean())


def test_team_table_reconciles_with_shots(pipeline_result):
    result, _, _ = pipeline_result
    teams = result["teams"].set_index("team")
    shots = result["shots"]
    assert set(teams.index) == set(TEAMS)
    assert teams.xg.sum() == pytest.approx(shots.xg.sum())
    assert teams.xg.sum() == pytest.approx(teams.xga.sum())
    assert (teams.matches == 2 * (len(TEAMS) - 1)).all()
    # goals in the synthetic league come only from shots, so they must match
    assert teams.gf.sum() == shots.is_goal.sum()


def test_form_rolling_average(pipeline_result):
    result, _, _ = pipeline_result
    f = result["form"]
    assert f.xg_roll.iloc[4] == pytest.approx(f.xg.iloc[:5].mean())
    assert f.xg_roll.iloc[-1] == pytest.approx(f.xg.iloc[-5:].mean())


def test_outputs_written(pipeline_result):
    _, out, _ = pipeline_result
    figs = {p.name for p in (out / "outputs" / "figures").glob("*.png")}
    assert {"calibration.png", "coefficients.png", "team_xgd.png"} <= figs
    assert any(n.startswith("shot_map_") for n in figs)
    assert any(n.startswith("rolling_form_") for n in figs)
    assert (out / "outputs" / "results.md").read_text(encoding="utf-8").count("|") > 50
    js = (out / "dashboard" / "data.js").read_text(encoding="utf-8")
    payload = json.loads(js.split("window.XG_DATA = ", 1)[1].rstrip().rstrip(";"))
    assert {"meta", "models", "calibration", "teams", "players", "form", "shots"} <= set(payload)
    assert len(payload["shots"]["rows"]) == payload["meta"]["n_shots"]
    assert set(payload["form"]) == set(TEAMS)


def test_cli_rejects_unknown_season(synthetic_cache, tmp_path):
    import run_pipeline
    cache, _ = synthetic_cache
    code = run_pipeline.main(["--competition", "Synthetic League", "--season", "1999/2000",
                              "--cache-dir", str(cache), "--offline",
                              "--output-dir", str(tmp_path), "--dashboard-dir", str(tmp_path)])
    assert code == 2
