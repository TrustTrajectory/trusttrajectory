import json
import os
import random

from conftest import FailingClient, ScriptedClient

from trusttrajectory.config import PRESETS, RunConfig
from trusttrajectory.models import ModelConfig
from trusttrajectory.runner import chat, make_tag, run_benchmark, run_scenario, window_messages
from trusttrajectory.scenarios import load_scenarios, smoke_scenarios
from trusttrajectory.scoring import make_scorer

MODEL = ModelConfig(name="fake/model", display_name="fake")


def run(client, scenario_id="easy_01", cfg=None, **kw):
    cfg = cfg or PRESETS["v4or_t13"]
    sc = load_scenarios(ids=[scenario_id])[0]
    return run_scenario(client, MODEL, sc, 1, cfg, make_scorer(cfg), log=lambda s: None, sleep=lambda s: None, **kw)


def test_trajectory_reaches_the_pivot_books_and_answers_probes(cfg):
    t = run(ScriptedClient())
    assert t["pivot_injected"] and t["complication_given"] and t["booking_done"]
    assert t["pivot_turn"] == 13 and 13 < t["turns_used"] <= cfg.max_turns
    turns = t["turn_records"]
    assert [r["turn_idx"] for r in turns] == list(range(1, len(turns) + 1))
    assert not turns[12 - 1]["pivot_occurred"] and turns[13 - 1]["pivot_occurred"]
    assert all("issues" in r and "param_total" in r for r in turns)
    assert t["scorer"] == "regex" and t["config"]["pivot_turn"] == 13
    assert t["first_hall_turn"] is None or 1 <= t["first_hall_turn"] <= t["turns_used"]


def test_run_is_deterministic_for_a_deterministic_model():
    a, b = run(ScriptedClient("x")), run(ScriptedClient("x"))
    strip = lambda t: {k: v for k, v in t.items() if k != "turn_records"}
    assert strip(a) == strip(b)
    assert [(r["turn_idx"], r["hallucination_type"]) for r in a["turn_records"]] == \
           [(r["turn_idx"], r["hallucination_type"]) for r in b["turn_records"]]


def test_execution_gate_blocks_early_tool_calls_but_still_logs_them():
    gated = run(ScriptedClient(tool_turn=3), cfg=RunConfig(min_booking_turn=8))
    early = [r for r in gated["turn_records"] if r["tool_called"] and r["turn_idx"] < 8]
    assert early, "the scripted model calls the tool before turn 8"
    assert gated["early_booking_turn"] is None or gated["early_booking_turn"] >= 8
    open_gate = run(ScriptedClient(tool_turn=3), cfg=RunConfig(gate_mode="none", min_booking_turn=8))
    assert open_gate["early_booking_turn"] is not None and open_gate["early_booking_turn"] < 8


def test_implied_confab_does_not_advance_state():
    t = run(ScriptedClient(confab=True))
    confab = [r for r in t["turn_records"] if r["detect_reason"] == "implied_confab"]
    assert confab and confab[0]["hallucination_type"] == "CONFIDENT_CONFAB"
    assert t["early_booking_turn"] != confab[0]["turn_idx"]


def test_randomised_pivot_is_recorded_and_seeded():
    cfg = RunConfig(pivot_turn=(11, 15))
    t1 = run(ScriptedClient(), cfg=cfg, rng=random.Random(5))
    t2 = run(ScriptedClient(), cfg=cfg, rng=random.Random(5))
    assert t1["pivot_turn"] == t2["pivot_turn"] and 11 <= t1["pivot_turn"] <= 15
    first_post = next(r["turn_idx"] for r in t1["turn_records"] if r["pivot_occurred"])
    assert first_post == t1["pivot_turn"]


def test_pivot_still_fires_if_it_coincides_with_the_complication():
    sc = load_scenarios(ids=["easy_01"])[0]  # complication at T10
    cfg = RunConfig(pivot_turn=10)
    t = run_scenario(ScriptedClient(), MODEL, sc, 1, cfg, make_scorer(cfg), log=lambda s: None, sleep=lambda s: None)
    assert t["complication_given"] and t["pivot_injected"]
    assert next(r["turn_idx"] for r in t["turn_records"] if r["pivot_occurred"]) == 11


def test_t15_preset_runs_thirty_turn_budget():
    t = run(ScriptedClient(), cfg=PRESETS["v4or_t15"])
    assert t["pivot_turn"] == 15 and t["turns_used"] <= 30


def test_chat_retries_then_reports_error(cfg):
    client = FailingClient()
    sleeps = []
    text, latency, err, usage = chat(client, MODEL, [{"role": "user", "content": "hi"}], cfg, log=lambda s: None, sleep=sleeps.append)
    assert text is None and "429" in err and usage is None
    assert client.attempts == cfg.retry_limit and len(sleeps) == cfg.retry_limit - 1
    assert all(s >= 30 for s in sleeps)  # 429 backoff


def test_window_keeps_system_prompt_and_recent_messages():
    msgs = [{"role": "system", "content": "s"}] + [{"role": "user", "content": str(i)} for i in range(30)]
    w = window_messages(msgs, 20)
    assert len(w) == 20 and w[0]["content"] == "s" and w[-1]["content"] == "29" and w[1]["content"] == "11"
    assert window_messages(msgs, None) is msgs


def test_run_benchmark_checkpoints_and_resumes(tmp_path):
    cfg = PRESETS["v4or_t13"].with_overrides(scenario_delay_s=0)
    scenarios = smoke_scenarios()[:2]
    kwargs = dict(cfg=cfg, n_runs=1, out_dir=str(tmp_path), tag="t", log=lambda s: None, sleep=lambda s: None)
    first = run_benchmark(ScriptedClient(), MODEL, scenarios[:1], **kwargs)
    assert len(first) == 1 and os.path.exists(tmp_path / "checkpoint_t.json")

    client = ScriptedClient()
    second = run_benchmark(client, MODEL, scenarios, **kwargs)
    assert [t["scenario_id"] for t in second] == [s["id"] for s in scenarios]
    # the first scenario was resumed from the checkpoint, so the model never saw its opening message again
    openings = {m[1]["content"] for m in client.calls if len(m) == 2}
    assert scenarios[0]["initial"] not in openings and scenarios[1]["initial"] in openings
    raw = json.load(open(tmp_path / "raw_t.json"))
    assert len(raw) == 2

    fresh = run_benchmark(ScriptedClient(), MODEL, scenarios[:1], resume=False, **kwargs)
    assert len(fresh) == 1


def test_make_tag_is_filesystem_safe():
    assert make_tag(ModelConfig(name="google/gemini-3.5-flash"), "smoke").startswith("gemini-3.5-flash_smoke_")


def test_records_carry_what_rescoring_needs():
    t = run(ScriptedClient())
    r = t["turn_records"][0]
    assert r["assistant_text_full"].startswith(r["assistant_text"]) and "committed_facts_before" in r
    assert r["expected_state"] == load_scenarios(ids=["easy_01"])[0]["expected_pre"]
    post = next(r for r in t["turn_records"] if r["pivot_occurred"])
    assert post["expected_state"] == load_scenarios(ids=["easy_01"])[0]["expected_post"]
    assert t["model_config"]["reasoning"] is None


def test_reasoning_setting_is_sent_as_extra_body():
    class Capture(ScriptedClient):
        def create(self, **kwargs):
            self.kwargs = kwargs
            return super().create(**kwargs)
    client = Capture()
    chat(client, ModelConfig("x/y", reasoning="off"), [{"role": "user", "content": "hi"}], PRESETS["v4or_t13"], log=lambda s: None)
    assert client.kwargs["extra_body"] == {"reasoning": {"enabled": False}}
    chat(client, ModelConfig("x/y", reasoning="low"), [{"role": "user", "content": "hi"}], PRESETS["v4or_t13"], log=lambda s: None)
    assert client.kwargs["extra_body"] == {"reasoning": {"effort": "low"}}
    chat(client, ModelConfig("x/y"), [{"role": "user", "content": "hi"}], PRESETS["v4or_t13"], log=lambda s: None)
    assert "extra_body" not in client.kwargs


def test_workers_give_the_same_results_in_scenario_order(tmp_path):
    cfg = PRESETS["v4or_t13"].with_overrides(scenario_delay_s=0)
    scenarios = smoke_scenarios()
    seq = run_benchmark(ScriptedClient(), MODEL, scenarios, cfg=cfg, n_runs=2, out_dir=str(tmp_path / "s"), tag="t",
                        log=lambda s: None, sleep=lambda s: None)
    par = run_benchmark(ScriptedClient(), MODEL, scenarios, cfg=cfg, n_runs=2, out_dir=str(tmp_path / "p"), tag="t",
                        workers=4, log=lambda s: None, sleep=lambda s: None)
    key = lambda t: (t["scenario_id"], t["run_idx"], t["first_hall_turn"], t["total_hallucinations"], t["booking_done"])
    assert [key(t) for t in seq] == [key(t) for t in par]
    assert [t["scenario_id"] for t in par] == [s["id"] for s in scenarios for _ in (1, 2)]


def test_filler_pads_only_pre_pivot_user_turns_and_avoids_scorer_vocab():
    from trusttrajectory.extraction import CUISINE_SYNONYMS, KNOWN_CITIES, KNOWN_DAYS, KNOWN_TIMES, NUMBER_WORDS, extract_number
    from trusttrajectory.simulator import FILLER_SENTENCES
    banned = list(KNOWN_CITIES) + list(KNOWN_DAYS) + list(KNOWN_TIMES) + list(CUISINE_SYNONYMS) + list(NUMBER_WORDS)
    for sent in FILLER_SENTENCES:
        low = sent.lower()
        assert "?" not in sent and extract_number(low) is None
        assert not any(f" {b} " in f" {low} " for b in banned), sent
    client = ScriptedClient()
    t = run(client, cfg=PRESETS["v4or_t13"].with_overrides(filler_tokens_per_turn=60))
    final = client.calls[-1]
    pre = [m for m in final if m["role"] == "user" and m["content"] not in ("Please continue.",) and not m["content"].startswith("TOOL_RESPONSE")]
    padded = [m for m in pre if any(f in m["content"] for f in FILLER_SENTENCES)]
    assert padded, "drip answers should carry filler"
    pivot_msg = next(m for m in final if m["role"] == "user" and "Change of plans" in m["content"])
    assert not any(f in pivot_msg["content"] for f in FILLER_SENTENCES)
    assert t["config"]["filler_tokens_per_turn"] == 60
    assert all(not any(f in m["content"] for f in FILLER_SENTENCES) for m in final[final.index(pivot_msg) + 1:] if m["role"] == "user")


def test_state_card_is_prepended_to_the_pivot():
    client = ScriptedClient()
    t = run(client, cfg=PRESETS["v4or_t13"].with_overrides(state_card_at_pivot=True))
    pivot_msg = next(m for m in client.calls[-1] if m["role"] == "user" and "Change of plans" in m["content"])
    assert pivot_msg["content"].startswith("Before I make a change, here is the confirmed state")
    assert "people: 2" in pivot_msg["content"] and "day: friday" in pivot_msg["content"]
    assert t["config"]["state_card_at_pivot"] is True


def test_api_failures_are_retried_and_never_kept_as_data(tmp_path):
    class FlakyClient(ScriptedClient):
        def __init__(self):
            super().__init__()
            self.fail_next = {"easy_01": 1}   # first attempt at easy_01 dies at its 3rd call
            self.count = {}
        def create(self, model, messages, temperature, max_tokens, extra_headers=None, **kw):
            sid = "easy_01" if "restaurant for tonight" in messages[1]["content"] else "other"
            self.count[sid] = self.count.get(sid, 0) + 1
            if sid == "easy_01" and self.fail_next.get(sid) and self.count[sid] == 3:
                self.fail_next[sid] = 0
                raise RuntimeError("Error code: 500 - boom")
            return super().create(model, messages, temperature, max_tokens, extra_headers, **kw)

    cfg = PRESETS["v4or_t13"].with_overrides(scenario_delay_s=0, retry_limit=1)
    scenarios = load_scenarios(ids=["easy_01", "easy_02"])
    logs = []
    out = run_benchmark(FlakyClient(), MODEL, scenarios, cfg=cfg, n_runs=1, out_dir=str(tmp_path), tag="t",
                        retry_pause_s=0, log=logs.append, sleep=lambda s: None)
    assert [t["scenario_id"] for t in out] == ["easy_01", "easy_02"] and all(t["error"] is None for t in out)
    assert any("Retrying 1 trajectories" in l for l in logs)
    failed = json.load(open(tmp_path / "failed_t.json"))
    assert len(failed) == 1 and failed[0]["scenario_id"] == "easy_01" and "boom" in failed[0]["error"]
    raw = json.load(open(tmp_path / "raw_t.json"))
    assert len(raw) == 2


def test_post_booking_confirmation_is_not_labelled_confab():
    class ConfirmingClient(ScriptedClient):
        def _reply(self, messages):
            # the harness follows a tool response with a confirmation prompt, so look back two messages
            if any(m["content"].startswith("TOOL_RESPONSE") for m in messages[-3:] if m["role"] == "user"):
                self.calls.append([dict(m) for m in messages])
                return "Your booking is confirmed! Table for 2 people in new york on friday at 8 PM, italian."
            return super()._reply(messages)
    t = run(ConfirmingClient())
    after_booking = [r for r in t["turn_records"] if r["detect_reason"] == "confirmation_after_booking"]
    assert after_booking and all(r["hallucination_type"] != "CONFIDENT_CONFAB" for r in after_booking)
    assert all("booking_executed" in r for r in t["turn_records"])


def test_chat_surfaces_upstream_error_bodies():
    from types import SimpleNamespace
    from trusttrajectory.runner import chat
    from trusttrajectory.config import get_preset
    from trusttrajectory.models import ModelConfig

    class Completions:
        def create(self, **kwargs):
            return SimpleNamespace(choices=None, error={"code": 429, "message": "Provider returned error"})

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    logs = []
    cfg = get_preset("v4or_t13")
    text, latency, err, usage = chat(client, ModelConfig(name="x/y"), [{"role": "user", "content": "hi"}], cfg,
                                     log=logs.append, sleep=lambda s: None)
    assert text is None and "Provider returned error" in err and "429" in err
    assert any("no choices in response" in l for l in logs)
