import csv
import json

import pytest
from conftest import ScriptedClient

from trusttrajectory import cli
from trusttrajectory.analysis import annotation_report, compare_conditions, export_sample, fleiss_kappa
from trusttrajectory.analysis.annotation import read_labels, write_csv
from trusttrajectory.config import PRESETS
from trusttrajectory.models import ModelConfig, load_dotenv
from trusttrajectory.runner import run_scenario
from trusttrajectory.scenarios import load_scenarios
from trusttrajectory.scoring import make_scorer


@pytest.fixture(scope="module")
def two_models():
    cfg = PRESETS["v4or_t13"]
    scorer = make_scorer(cfg)
    out = {}
    for label, salt in (("ModelA", "a"), ("ModelB", "b")):
        out[label] = [run_scenario(ScriptedClient(salt + str(i), confab=(i % 3 == 0)), ModelConfig("fake/" + label, label), sc, 1,
                                   cfg, scorer, log=lambda s: None, sleep=lambda s: None)
                      for i, sc in enumerate(load_scenarios(ids=["easy_02", "medium_03", "hard_04", "flight_conflict_02", "adversarial_01", "flight_easy_04"]))]
    return out


def test_export_sample_is_blind_stratified_and_keyed(two_models):
    trajs = two_models["ModelA"] + two_models["ModelB"]
    blind, key = export_sample(trajs, n=40, seed=1)
    assert len(blind) == len(key) == 40
    assert all(row["label"] == "" and "auto_label" not in row for row in blind)
    assert {k["auto_label"] for k in key} <= {"NONE", "PARAM_DRIFT", "PREMATURE_COMMIT", "CONSTRAINT_IGNORE", "CONFIDENT_CONFAB", "MEMORY_COLLAPSE"}
    strata = {(k["model"], k["difficulty"], k["phase"]) for k in key}
    assert len(strata) >= 10  # round-robin across model x tier x phase
    assert export_sample(trajs, n=40, seed=1)[1] == key  # seeded


def test_annotation_report(tmp_path, two_models):
    blind, key = export_sample(two_models["ModelA"], n=30, seed=2)
    write_csv(key, tmp_path / "key.csv")
    auto = read_labels(str(tmp_path / "key.csv"), column="auto_label")
    perfect = dict(auto)
    noisy = {sid: ("NONE" if lab != "NONE" and i % 4 == 0 else lab) for i, (sid, lab) in enumerate(auto.items())}
    rep = annotation_report(auto, {"ann1": perfect, "ann2": noisy})
    assert rep["auto_vs_annotator"]["ann1"]["kappa"] == 1.0 and rep["auto_vs_annotator"]["ann1"]["agreement"] == 1.0
    assert rep["auto_vs_annotator"]["ann2"]["agreement"] <= 1.0
    assert "ann1|ann2" in rep["pairwise_kappa"] and "fleiss_kappa" in rep
    assert rep["n_majority_items"] <= 30


def test_fleiss_kappa():
    assert fleiss_kappa([["A", "A"], ["B", "B"], ["A", "A"]]) == 1.0
    assert fleiss_kappa([["A", "B"], ["B", "A"]]) < 0.5
    assert fleiss_kappa([]) != fleiss_kappa([])  # nan


def test_compare_conditions_writes_summary(tmp_path, two_models):
    public = compare_conditions(two_models, str(tmp_path), PRESETS["v4or_t13"], n_boot=100, min_n=1, figures=True)
    assert set(public) == {"ModelA", "ModelB"}
    assert (tmp_path / "summary.json").exists() and (tmp_path / "fig6_decay_by_condition_all.png").exists()
    assert public["ModelA"]["pivot_turns"] == [13]


def test_cli_compare_rescore_annotate_and_context_gap(tmp_path, two_models, capsys):
    paths = []
    for label, traj in two_models.items():
        p = tmp_path / f"raw_{label}.json"
        p.write_text(json.dumps(traj, default=str))
        paths.append(str(p))
    assert cli.main(["compare", *paths, "--labels", "A,B", "--out", str(tmp_path / "cmp"), "--n-boot", "50", "--min-n", "1", "--no-figures"]) == 0
    assert "A " in capsys.readouterr().out
    assert cli.main(["rescore", paths[0], "--scorer", "regex", "--out", str(tmp_path / "rs")]) == 0
    assert "agreement: 1.000" in capsys.readouterr().out
    assert cli.main(["annotate-export", *paths, "--out", str(tmp_path / "ann"), "--n", "20"]) == 0
    rows = list(csv.DictReader(open(tmp_path / "ann" / "annotation_sample.csv")))
    assert len(rows) == 20 and (tmp_path / "ann" / "GUIDELINES.md").exists()
    for r in rows:  # simulate a perfect annotator from the key
        pass
    key = {k["sample_id"]: k["auto_label"] for k in csv.DictReader(open(tmp_path / "ann" / "annotation_key.csv"))}
    for r in rows:
        r["label"] = key[r["sample_id"]]
    write_csv(rows, tmp_path / "ann" / "alice.csv")
    assert cli.main(["annotate-score", "--key", str(tmp_path / "ann" / "annotation_key.csv"), str(tmp_path / "ann" / "alice.csv"),
                     "--out", str(tmp_path / "ann" / "report.json")]) == 0
    assert "kappa 1.000" in capsys.readouterr().out
    assert cli.main(["context-gap", *paths, "--source", "medium", "--target", "hard"]) == 0
    assert "--filler-tokens" in capsys.readouterr().out
    assert cli.main(["estimate", "--models", "google/gemini-3.5-flash", "--runs", "1"]) == 0
    assert "TOTAL" in capsys.readouterr().out


def test_dotenv_loading(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nOPENROUTER_API_KEY=sk-or-test\nOPENROUTER_APP_TITLE='Bench'\n")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "keep-me")
    loaded = load_dotenv(str(env))
    assert loaded == {"OPENROUTER_API_KEY": "sk-or-test"}
    import os
    assert os.environ["OPENROUTER_APP_TITLE"] == "keep-me"
