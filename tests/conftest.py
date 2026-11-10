import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic import COMP_NAME, SEASON_NAME, TEAMS, build_league  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_cache(tmp_path_factory):
    cache = tmp_path_factory.mktemp("cache")
    info = build_league(cache, shots_per_team=(14, 22))
    return cache, info


@pytest.fixture(scope="session")
def pipeline_result(synthetic_cache, tmp_path_factory):
    """Run the full pipeline once, offline, on the synthetic league."""
    import run_pipeline

    cache, info = synthetic_cache
    out = tmp_path_factory.mktemp("out")
    args = run_pipeline.parse_args([
        "--competition", COMP_NAME, "--season", SEASON_NAME, "--team", TEAMS[0],
        "--cache-dir", str(cache), "--output-dir", str(out / "outputs"),
        "--dashboard-dir", str(out / "dashboard"), "--offline", "--bootstrap", "200",
        "--workers", "2",
    ])
    result = run_pipeline.run(args)
    return result, out, info
