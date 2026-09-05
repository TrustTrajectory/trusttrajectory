import json

from conftest import ScriptedClient

from trusttrajectory import cli
from trusttrajectory.config import PRESETS
from trusttrajectory.models import ModelConfig
from trusttrajectory.runner import run_scenario
from trusttrajectory.scenarios import load_scenarios
from trusttrajectory.scoring import make_scorer


def test_list_scenarios(capsys):
    assert cli.main(["list-scenarios", "--tiers", "easy"]) == 0
    out = capsys.readouterr().out
    assert "easy_01" in out and "10 scenarios" in out


def test_analyze_end_to_end(tmp_path, capsys):
    cfg = PRESETS["v4or_t13"]
    scorer = make_scorer(cfg)
    traj = [run_scenario(ScriptedClient(str(i)), ModelConfig("fake/m", "fake"), sc, 1, cfg, scorer,
                         log=lambda s: None, sleep=lambda s: None)
            for i, sc in enumerate(load_scenarios(ids=["easy_01", "flight_hard_01", "adversarial_02"]))]
    raw = tmp_path / "raw_demo.json"
    raw.write_text(json.dumps(traj, default=str))
    out = tmp_path / "analysis"
    assert cli.main(["analyze", str(raw), "--out", str(out), "--preset", "v4or_t13", "--n-boot", "200", "--min-n", "1"]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["n_trajectories"] == 3 and "pre_post" in report
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert any(p.startswith("fig1_trust_decay") for p in pngs) and any(p.startswith("fig3_taxonomy") for p in pngs)
    assert "RESULTS SUMMARY" in capsys.readouterr().out


def test_run_config_overrides_parse():
    parser = cli.build_parser()
    args = parser.parse_args(["run", "--model", "x/y", "--pivot-range", "11", "15", "--gate", "none", "--max-turns", "25"])
    cfg = cli._build_config(args)
    assert cfg.pivot_turn == (11, 15) and cfg.gate_mode == "none" and cfg.max_turns == 25
