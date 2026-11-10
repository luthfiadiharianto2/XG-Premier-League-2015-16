"""xG Lab — one command from raw StatsBomb events to figures, tables and dashboard.

    python run_pipeline.py                                   # PL 2015/16, Man United focus
    python run_pipeline.py --team Arsenal --max-matches 100
    python run_pipeline.py --competition "La Liga" --season 2015/2016 --team Barcelona
    python run_pipeline.py --list                            # what StatsBomb publishes
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src import analysis as A
from src import evaluate as E
from src import export as X
from src import features as F
from src import model as M
from src import plots as PL
from src.data_loader import (DataNotAvailableError, StatsBombClient, iter_events,
                             matches_frame, resolve_competition)

log = logging.getLogger("xglab")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build an xG model and analysis from StatsBomb Open Data.")
    p.add_argument("--competition", default=config.DEFAULT_COMPETITION, help="name or id")
    p.add_argument("--season", default=config.DEFAULT_SEASON, help="e.g. 2015/2016, 2015/16 or id")
    p.add_argument("--team", default=config.DEFAULT_TEAM, help="focus team for shot map / form")
    p.add_argument("--max-matches", type=int, default=config.DEFAULT_MAX_MATCHES,
                   help="use only the first N matches chronologically")
    p.add_argument("--folds", type=int, default=config.N_FOLDS)
    p.add_argument("--workers", type=int, default=config.DOWNLOAD_WORKERS)
    p.add_argument("--cache-dir", type=Path, default=config.CACHE_DIR)
    p.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    p.add_argument("--dashboard-dir", type=Path, default=config.DASHBOARD_DIR)
    p.add_argument("--base-url", default=config.BASE_URL)
    p.add_argument("--offline", action="store_true", help="use cached files only")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--bootstrap", type=int, default=config.BOOTSTRAP_REPS)
    p.add_argument("--list", action="store_true", help="list available competitions/seasons and exit")
    return p.parse_args(argv)


def list_competitions(client: StatsBombClient) -> None:
    comps = pd.DataFrame(client.competitions())
    comps = comps[["competition_id", "competition_name", "season_id", "season_name"]]
    print(comps.sort_values(["competition_name", "season_name"]).to_string(index=False))


def run(args: argparse.Namespace) -> dict:
    t0 = time.time()
    client = StatsBombClient(base_url=args.base_url, cache_dir=args.cache_dir, offline=args.offline)

    # ------------------------------------------------------------------ data
    comp = resolve_competition(client.competitions(), args.competition, args.season)
    log.info("Competition: %s %s (ids %d/%d)", comp["competition_name"], comp["season_name"],
             comp["competition_id"], comp["season_id"])
    matches = matches_frame(client.matches(comp["competition_id"], comp["season_id"]))
    if args.max_matches:
        matches = matches.head(args.max_matches).reset_index(drop=True)
    teams_present = sorted(set(matches.home_team) | set(matches.away_team))
    if args.team not in teams_present:
        raise DataNotAvailableError(
            f"Team '{args.team}' not found in the selected matches. Teams: {', '.join(teams_present)}")
    log.info("Loading events for %d matches (cache: %s)", len(matches), args.cache_dir)

    rows, shootout_dropped = [], 0
    match_lookup = matches.set_index("match_id").to_dict(orient="index")
    for mid, events in iter_events(client, matches.match_id, workers=args.workers):
        shootout_dropped += sum(1 for e in events
                                if e.get("type", {}).get("name") == "Shot" and e.get("period") == 5)
        rows.extend(F.extract_shots(events, {"match_id": mid, **match_lookup[mid]}))
    shots = F.shots_frame(rows)
    pen_mask = F.is_penalty(shots)
    np_shots = shots[~pen_mask].reset_index(drop=True)
    pen_xg, pen_goals, pen_n = F.penalty_conversion(shots)
    log.info("Shots: %d (non-penalty %d, penalties %d, shoot-out dropped %d)",
             len(shots), len(np_shots), pen_n, shootout_dropped)

    corr, flagged = F.feature_correlations(np_shots)
    for a, b, r in flagged:
        log.info("  note: |r| >= 0.8 between %s and %s (r=%.2f)", a, b, r)

    # ---------------------------------------------------------------- models
    folds = M.make_folds(np_shots.match_id, n_splits=args.folds)
    y = np_shots[F.LABEL].to_numpy()
    preds, labels = {}, {}
    for spec in M.MODEL_SPECS:
        ts = time.time()
        preds[spec.key] = M.oof_predict(np_shots, spec, folds)
        labels[spec.key] = spec.label
        log.info("  %-40s OOF done in %.1fs", spec.label, time.time() - ts)
    labels["statsbomb"] = "StatsBomb xG (reference)"
    sb = np_shots["statsbomb_xg"].to_numpy(dtype=float)
    has_sb = not np.isnan(sb).all()
    model_table = E.compare_models(y, preds, labels, reference=sb if has_sb else None)
    main_key = E.select_main(model_table)
    log.info("Main model: %s", labels[main_key])

    groups = np_shots.match_id.to_numpy()
    boot_rows = []
    for key in [k for k in preds if k != "m1_geometry"]:
        b = E.paired_bootstrap(y, preds[key], preds["m1_geometry"], groups, reps=args.bootstrap)
        boot_rows.append({"comparison": f"{labels[key]} vs model 1", **b})
    if has_sb:
        m = ~np.isnan(sb)
        b = E.paired_bootstrap(y[m], preds[main_key][m], sb[m], groups[m], reps=args.bootstrap)
        boot_rows.append({"comparison": f"{labels[main_key]} vs StatsBomb xG", **b})
    boot = pd.DataFrame(boot_rows)

    cal = {k: E.calibration_table(y, p) for k, p in preds.items()}
    if has_sb:
        m = ~np.isnan(sb)
        cal["statsbomb"] = E.calibration_table(y[m], sb[m])

    logreg_key = main_key if M.SPEC_BY_KEY[main_key].kind == "logreg" else "m3_freeze"
    coefs = M.logreg_coefficients(M.fit_full(np_shots, M.SPEC_BY_KEY[logreg_key]))

    # -------------------------------------------------------------- analysis
    np_shots["xg"] = preds[main_key]
    pens = shots[pen_mask].copy()
    pens["xg"] = pen_xg
    all_shots = pd.concat([np_shots, pens], ignore_index=True)
    mt = A.match_table(matches, all_shots)
    long = A.team_match_long(mt)
    teams = A.team_table(long)
    players_all = A.player_table(all_shots, min_shots=1)
    players = players_all[players_all.shots >= config.MIN_SHOTS_PLAYER].reset_index(drop=True)
    forms = {t: A.team_form(long, t) for t in teams_present}
    focus_form = forms[args.team]

    # ---------------------------------------------------------------- output
    out, fig_dir = args.output_dir, args.output_dir / "figures"
    meta = {
        "competition": comp["competition_name"], "season": comp["season_name"],
        "team": args.team, "n_matches": int(len(matches)), "n_shots": int(len(shots)),
        "n_np_shots": int(len(np_shots)), "n_penalties": int(pen_n),
        "n_shootout_dropped": int(shootout_dropped),
        "freeze_frame_coverage": float(np_shots.has_freeze_frame.mean()),
        "penalty_xg": pen_xg, "penalty_goals": pen_goals, "penalty_attempts": pen_n,
        "main_model": main_key, "main_model_label": labels[main_key],
        "coef_model_label": labels[logreg_key], "folds": len(folds),
        "min_shots_player": config.MIN_SHOTS_PLAYER, "rolling_window": config.ROLLING_WINDOW,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "correlated_pairs": [{"a": a, "b": b, "r": r} for a, b, r in flagged],
        "source": "StatsBomb Open Data (github.com/statsbomb/open-data)",
    }
    X.write_tables(out, {
        "model_comparison": model_table, "bootstrap": boot, "coefficients": coefs,
        "teams": teams, "players": players_all, "matches": mt,
        "team_form_" + args.team.lower().replace(" ", "_"): focus_form,
        "feature_correlation": corr.reset_index().rename(columns={"index": "feature"}),
        **{f"calibration_{k}": v for k, v in cal.items()},
    })
    all_shots.drop(columns=["event_id"]).to_csv(out / "tables" / "shots.csv", index=False,
                                                 float_format="%.4f")
    X.write_summary(out, {"meta": meta, "models": model_table.to_dict(orient="records"),
                          "bootstrap": boot.to_dict(orient="records")})
    X.write_results_markdown(out, meta, model_table, boot, teams, players, coefs)

    if not args.no_plots:
        slug = args.team.lower().replace(" ", "_")
        PL.shot_map(all_shots, args.team, f"{args.team} shot map · {comp['season_name']}",
                    fig_dir / f"shot_map_{slug}.png")
        PL.calibration_plot(cal, labels, fig_dir / "calibration.png")
        PL.coefficient_plot(coefs, f"Standardised coefficients · {labels[logreg_key]}",
                            fig_dir / "coefficients.png")
        PL.team_xgd_plot(teams, args.team, fig_dir / "team_xgd.png")
        PL.rolling_form_plot(focus_form, args.team, config.ROLLING_WINDOW,
                             fig_dir / f"rolling_form_{slug}.png")

    form_cols = ["matchday", "match_date", "opponent", "venue", "gf", "ga", "xg", "xga",
                 "pts", "xpts", "xg_roll", "xga_roll"]
    payload = {
        "meta": meta,
        "models": X._records(model_table, 4),
        "bootstrap": X._records(boot, 4),
        "calibration": {k: X._records(v, 4) for k, v in cal.items()},
        "coefficients": X._records(coefs, 4),
        "teams": X._records(teams, 3),
        "players": X._records(players_all, 3),
        "matches": X._records(mt[["match_id", "match_date", "home_team", "away_team",
                                  "home_score", "away_score", "home_xg", "away_xg",
                                  "home_xpts", "away_xpts"]], 3),
        "form": {t: X._records(f[form_cols], 3) for t, f in forms.items()},
        "shots": X.shots_payload(all_shots),
    }
    X.write_dashboard_data(args.dashboard_dir / "data.js", payload)
    log.info("Done in %.1fs. Outputs: %s, dashboard data: %s", time.time() - t0, out,
             args.dashboard_dir / "data.js")
    return {"meta": meta, "models": model_table, "bootstrap": boot, "teams": teams,
            "players": players_all, "shots": all_shots, "coefficients": coefs,
            "calibration": cal, "folds": folds, "np_shots": np_shots, "form": focus_form}


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args(argv)
    try:
        if args.list:
            list_competitions(StatsBombClient(base_url=args.base_url, cache_dir=args.cache_dir,
                                              offline=args.offline))
            return 0
        result = run(args)
    except DataNotAvailableError as exc:
        log.error("%s", exc)
        return 2
    except ConnectionError as exc:
        log.error("%s\nCheck your internet connection and re-run; files already downloaded "
                  "are cached in %s.", exc, args.cache_dir)
        return 3
    print("\n" + result["models"][["model", "auc", "brier", "log_loss", "xg_total", "goals"]]
          .round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
