"""Static matplotlib figures (colour-blind-safe Okabe-Ito palette)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Arc, Rectangle  # noqa: E402

import config  # noqa: E402

P = config.PALETTE
CREDIT = "Data: StatsBomb Open Data"


def _style(ax, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis, color="#000000", alpha=0.08, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, path: Path) -> Path:
    fig.text(0.995, 0.005, CREDIT, ha="right", va="bottom", fontsize=7, color="#666666")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_half_pitch(ax, line="#555555") -> None:
    """Vertical attacking half: horizontal axis = StatsBomb y, vertical = x."""
    kw = dict(fill=False, edgecolor=line, linewidth=1)
    ax.add_patch(Rectangle((0, 60), 80, 60, **kw))
    ax.add_patch(Rectangle((18, 102), 44, 18, **kw))
    ax.add_patch(Rectangle((30, 114), 20, 6, **kw))
    ax.add_patch(Rectangle((36, 120), 8, 1.6, **kw))
    ax.add_patch(Arc((40, 108), 20, 20, theta1=217, theta2=323, color=line, linewidth=1))
    ax.plot([40], [108], marker="o", color=line, markersize=2)
    ax.set_xlim(-2, 82)
    ax.set_ylim(59.5, 122.5)  # attacking half; bottom band holds the legend
    ax.set_aspect("equal")
    ax.axis("off")


def shot_map(shots: pd.DataFrame, team: str, title: str, path: Path) -> Path:
    s = shots[shots.team == team]
    fig, ax = plt.subplots(figsize=(7, 5.0))
    draw_half_pitch(ax)
    far = int((s.x < 60).sum())
    miss = s[s.is_goal == 0]
    goal = s[s.is_goal == 1]
    size = lambda xg: 20 + 600 * xg  # noqa: E731
    ax.scatter(miss.y, miss.x, s=size(miss.xg), facecolors="none", edgecolors=P["blue"],
               linewidths=0.9, alpha=0.7, zorder=3)
    ax.scatter(goal.y, goal.x, s=size(goal.xg), c=P["vermillion"], edgecolors="black",
               linewidths=0.6, alpha=0.9, zorder=4)
    handles = [
        Line2D([], [], marker="o", ls="", mfc="none", mec=P["blue"], ms=8, label=f"No goal ({len(miss)})"),
        Line2D([], [], marker="o", ls="", mfc=P["vermillion"], mec="black", ms=8, label=f"Goal ({len(goal)})"),
    ]
    for v in (0.05, 0.2, 0.5):
        handles.append(Line2D([], [], marker="o", ls="", mfc="#dddddd", mec="#777777",
                              ms=np.sqrt(size(v)), label=f"xG {v:.2f}"))
    ax.legend(handles=handles, loc="lower center", ncol=5, frameon=True, fontsize=8,
              bbox_to_anchor=(0.5, 0.005), handletextpad=0.3, columnspacing=1.0,
              facecolor="white", edgecolor="none", framealpha=0.95)
    ax.set_title(f"{title}\n{len(s)} shots · {s.xg.sum():.1f} xG · {int(s.is_goal.sum())} goals",
                 fontsize=11, loc="left")
    return _save(fig, path)


def calibration_plot(cal: dict[str, pd.DataFrame], labels: dict[str, str], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    hi = max(max(t.mean_pred.max(), t.obs_rate.max()) for t in cal.values()) * 1.08
    ax.plot([0, hi], [0, hi], color="#444444", ls=":", lw=1, label="Perfect calibration")
    markers = {"m1_geometry": "s", "m2_context": "^", "m3_freeze": "D", "m4_hgb": "o", "statsbomb": "x"}
    for key, t in cal.items():
        color = config.MODEL_COLORS.get(key, P["black"])
        ls = "--" if key == "statsbomb" else "-"
        ax.plot(t.mean_pred, t.obs_rate, marker=markers.get(key, "o"), ms=5, lw=1.4,
                ls=ls, color=color, label=labels[key])
        if key != "statsbomb":
            ax.fill_between(t.mean_pred, t.ci_low, t.ci_high, color=color, alpha=0.08, lw=0)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("Mean predicted xG in bin")
    ax.set_ylabel("Observed goal rate in bin")
    ax.set_title("Calibration (out-of-fold, 10 quantile bins)", loc="left", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    _style(ax)
    return _save(fig, path)


def coefficient_plot(coefs: pd.DataFrame, title: str, path: Path, top: int = 20) -> Path:
    c = coefs.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(c) + 1.4))
    colors = [P["blue"] if v > 0 else P["vermillion"] for v in c.coef]
    ax.barh(c.feature, c.coef, color=colors, height=0.65)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_xlabel("Coefficient (change in log-odds per 1 SD of feature)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=P["blue"], label="Raises xG"),
                       Rectangle((0, 0), 1, 1, color=P["vermillion"], label="Lowers xG")],
              frameon=False, fontsize=8, loc="lower right")
    ax.tick_params(axis="y", labelsize=8)
    _style(ax, "x")
    return _save(fig, path)


def team_xgd_plot(teams: pd.DataFrame, focus: str, path: Path) -> Path:
    t = teams.sort_values("xgd_pm")
    fig, ax = plt.subplots(figsize=(7.5, 0.33 * len(t) + 1.5))
    colors = [P["blue"] if v >= 0 else P["vermillion"] for v in t.xgd_pm]
    ax.barh(t.team, t.xgd_pm, color=colors, height=0.65, alpha=0.9, label="xGD per match")
    ax.scatter(t.gd / t.matches, t.team, marker="D", s=22, color=P["black"], zorder=3,
               label="Actual GD per match")
    ax.axvline(0, color="#333333", lw=0.8)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() == focus:
            lbl.set_fontweight("bold")
    ax.set_xlabel("Goals per match")
    ax.set_title("Expected goal difference per match", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.tick_params(axis="y", labelsize=8)
    _style(ax, "x")
    return _save(fig, path)


def rolling_form_plot(form: pd.DataFrame, team: str, window: int, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = form.matchday
    ax.plot(x, form.xg_roll, color=P["blue"], lw=2, label=f"xG for ({window}-match avg)")
    ax.plot(x, form.xga_roll, color=P["vermillion"], lw=2, ls="--", label=f"xG against ({window}-match avg)")
    ax.scatter(x, form.xg, color=P["blue"], s=10, alpha=0.35, label="xG for (single match)")
    ax.scatter(x, form.xga, color=P["vermillion"], marker="x", s=12, alpha=0.45,
               label="xG against (single match)")
    ax.fill_between(x, form.xg_roll, form.xga_roll, where=form.xg_roll >= form.xga_roll,
                    color=P["blue"], alpha=0.08, interpolate=True)
    ax.fill_between(x, form.xg_roll, form.xga_roll, where=form.xg_roll < form.xga_roll,
                    color=P["vermillion"], alpha=0.08, interpolate=True)
    ax.set_xlabel("Matchday")
    ax.set_ylabel("Expected goals")
    ax.set_xlim(0.5, x.max() + 0.5)
    ax.set_ylim(0, max(form.xg.max(), form.xga.max()) * 1.32)  # headroom for legend
    ax.set_title(f"{team}: rolling form", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    _style(ax)
    return _save(fig, path)
