"""Central configuration for xG Lab.

Everything that a user might reasonably want to change lives here; the CLI in
``run_pipeline.py`` can override the data selection and paths.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- data
BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
CACHE_DIR = ROOT / "data" / "cache"
OUTPUT_DIR = ROOT / "outputs"
DASHBOARD_DIR = ROOT / "dashboard"

# The original brief targeted Premier League 2025/26, but StatsBomb Open Data
# only publishes two Premier League seasons (2003/04 and 2015/16, checked
# November 2026). 2015/16 is the only complete 380-match PL season, so it is the
# default. Pass --season to pick another one; unknown seasons fail loudly with
# the list of seasons that do exist.
DEFAULT_COMPETITION = "Premier League"
DEFAULT_SEASON = "2015/2016"
DEFAULT_TEAM = "Manchester United"
DEFAULT_MAX_MATCHES: int | None = None

HTTP_TIMEOUT = 30          # seconds per request
HTTP_RETRIES = 4           # attempts per file
HTTP_BACKOFF = 1.5         # seconds; doubles after each failed attempt
DOWNLOAD_WORKERS = 8

# --------------------------------------------------------------------- modelling
N_FOLDS = 5
RANDOM_STATE = 42
CALIBRATION_BINS = 10
BOOTSTRAP_REPS = 1000

# -------------------------------------------------------------------- analysis
ROLLING_WINDOW = 5
MIN_SHOTS_PLAYER = 20      # used for README / dashboard default filter
MAX_GOALS_POISSON = 10     # truncation of the Poisson score grid (tail mass folded in)

# ---------------------------------------------------- StatsBomb pitch geometry
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_X = 120.0
GOAL_CENTER_Y = 40.0
POST_LOW_Y = 36.0
POST_HIGH_Y = 44.0

# --------------------------------------------------------------------- styling
# Okabe & Ito (2008) colour-blind-safe palette.
PALETTE = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
}
MODEL_COLORS = {
    "m1_geometry": PALETTE["sky"],
    "m2_context": PALETTE["green"],
    "m3_freeze": PALETTE["blue"],
    "m4_hgb": PALETTE["vermillion"],
    "statsbomb": PALETTE["grey"],
}
