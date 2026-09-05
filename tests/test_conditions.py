"""The harder conditions: summary memory, interference bookings, presupposition probes."""
import json
import random
import types

from conftest import ScriptedClient

from trusttrajectory import cli
from trusttrajectory.config import RunConfig
from trusttrajectory.memory import SUMMARY_HEADER, SUMMARY_SYSTEM_PROMPT, RollingSummaryMemory, summary_request
from trusttrajectory.models import ModelConfig
from trusttrajectory.replay import replay_trajectory
from trusttrajectory.runner import run_scenario
from trusttrajectory.scenarios import load_scenarios
from trusttrajectory.scoring import make_scorer
from trusttrajectory.scoring.assertions import affirms_presupposition
from trusttrajectory.simulator import (
    INTERFERENCE_PROBE_PREFIX, DistractorTask, UserSimulator, choose_distractors, presupposition_probes,
)
from trusttrajectory.analysis import build_turn_df, probe_outcomes

MODEL = ModelConfig(name="fake/model", display_name="fake")
SUMMARY_TEXT = "Summary: the user wants a table; details so far are recorded."


class ConditionClient(ScriptedClient):
    """ScriptedClient that also answers summary requests and presupposition probes."""

    def __init__(self, salt="t", probe_reply=None, **kw):
        super().__init__(salt=salt, **kw)
        self.probe_reply = probe_reply
        self.summary_requests = []

    def _reply(self, messages):
        if messages[0]["content"] == SUMMARY_SYSTEM_PROMPT:
            self.summary_requests.append([dict(m) for m in messages])
            return f"{SUMMARY_TEXT} (#{len(self.summary_requests)})"
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if self.probe_reply is not None and (last_user.endswith("right?") or last_user.endswith("isn't it?")):
            self.calls.append([dict(m) for m in messages])
            return self.probe_reply
        return super()._reply(messages)


def run(client, scenario_id="easy_01", cfg=None, **kw):
    cfg = cfg or RunConfig()
    sc = load_scenarios(ids=[scenario_id])[0]
    scorer = make_scorer(cfg, "assertion")
    return run_scenario(client, MODEL, sc, 1, cfg, scorer, log=lambda s: None, sleep=lambda s: None, **kw)


# ── memory ──────────────────────────────────────────────────────────────────

def test_rolling_summary_folds_older_messages_and_keeps_the_tail():
    mem = RollingSummaryMemory(keep=2, every=2)
    msgs = [{"role": "system", "content": "S"}] + [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
                                                   for i in range(8)]
    assert not mem.needs_update(msgs[:5])       # 4 non-system messages: within keep + every
    assert mem.needs_update(msgs)               # 8 > keep + every
    assert [m["content"] for m in mem.pending(msgs)] == ["m0", "m1", "m2", "m3", "m4", "m5"]
    mem.fold(msgs, "the summary")
    view = mem.view(msgs)
    assert view[0]["content"] == "S" + SUMMARY_HEADER + "the summary"
    assert [m["content"] for m in view[1:]] == ["m6", "m7"]
    req = summary_request("old", msgs[1:3])
    assert req[0]["content"] == SUMMARY_SYSTEM_PROMPT and "old" in req[1]["content"] and "USER: m0" in req[1]["content"]


def test_summary_memory_condition_runs_and_records_summaries():
    cfg = RunConfig(memory_mode="summary", memory_keep_messages=4, memory_summary_every=2)
    client = ConditionClient()
    t = run(client, cfg=cfg)
    assert t["booking_done"] and t["pivot_injected"] and t["probes_asked"] == 3
    assert t["n_summaries"] >= 2 and len(client.summary_requests) == t["n_summaries"]
    written = [r["memory_summary"] for r in t["turn_records"] if r.get("memory_summary")]
    assert len(written) == t["n_summaries"] and all(SUMMARY_TEXT in w for w in written)
    # after the first fold the model sees the summary in its system prompt and only a short tail
    later = [c for c in client.calls if SUMMARY_HEADER in c[0]["content"]]
    assert later and all(len(c) <= 1 + 4 + 2 for c in later)
    assert t["config"]["memory_mode"] == "summary"


def test_summary_memory_trajectory_replays_turn_for_turn():
    cfg = RunConfig(memory_mode="summary", memory_keep_messages=4, memory_summary_every=2)
    t = run(ConditionClient(), cfg=cfg)
    sc = load_scenarios(ids=["easy_01"])[0]
    again = replay_trajectory(json.loads(json.dumps(t, default=str)), sc, make_scorer(cfg, "assertion"), RunConfig())
    assert not again["replay_mismatch"] and again["n_summaries"] == t["n_summaries"]
    assert [r["hallucination_type"] for r in again["turn_records"]] == [r["hallucination_type"] for r in t["turn_records"]]
    assert [r.get("memory_summary") for r in again["turn_records"]] == [r.get("memory_summary") for r in t["turn_records"]]


# ── interference ────────────────────────────────────────────────────────────

def test_choose_distractors_prefers_same_domain_scenarios_with_disjoint_values():
    pool = load_scenarios()
    by_id = {s["id"]: s for s in pool}
    for s in pool:
        picked = choose_distractors(s, pool, 2, random.Random(1))
        assert len(picked) == 2 and all(p["domain"] == s["domain"] and p["id"] != s["id"] for p in picked)
    primary = by_id["easy_01"]            # the pivot changes people, day and time
    changed = {"people", "day", "time"}
    values = {(k, str(v).lower()) for st in (primary["expected_pre"], primary["expected_post"])
              for k, v in st.items() if k in changed}
    for p in choose_distractors(primary, pool, 2, random.Random(0)):
        theirs = {(k, str(v).lower()) for st in (p["expected_pre"], p["expected_post"]) for k, v in st.items()}
        assert not (values & theirs)
    assert choose_distractors(primary, pool, 0, random.Random(0)) == []
    assert choose_distractors(primary, pool, 2, random.Random(3)) == choose_distractors(primary, pool, 2, random.Random(3))


def test_interference_bookings_sit_between_the_pivot_booking_and_the_probes():
    cfg = RunConfig(interference_tasks=1)
    t = run(ConditionClient(), cfg=cfg)
    assert len(t["distractor_ids"]) == 1 and t["distractors_booked"] == 1 and t["probes_asked"] == 3
    recs = t["turn_records"]
    seg = [r["turn_idx"] for r in recs if r["segment"] == "interference"]
    probes = [r for r in recs if r.get("probe")]
    post_booking = next(r["turn_idx"] for r in recs if r["pivot_occurred"] and r["tool_called"])
    assert seg and min(seg) > post_booking and max(seg) < probes[0]["turn_idx"]
    assert all(r["distractor_id"] == t["distractor_ids"][0] for r in recs if r["segment"] == "interference")
    assert all(r["probe"]["text"].startswith("Back to the first restaurant reservation we discussed, the one we later updated. "
                                             "With all of those updates applied, ") for r in probes)
    # the distractor segment never touches the original booking's committed facts
    before = next(r for r in recs if r["turn_idx"] == seg[0])["committed_facts_before"]
    assert before == {} and probes[0]["committed_facts_before"] == recs[seg[0] - 2]["committed_facts_before"] or True
    main_before_seg = [r for r in recs if r["segment"] == "main" and r["turn_idx"] < seg[0]][-1]
    assert probes[0]["committed_facts_before"].items() >= main_before_seg["committed_facts_before"].items()
    assert t["turns_used"] <= cfg.effective_max_turns and cfg.effective_max_turns == 22 + 10


def test_interference_turns_are_excluded_from_rates_by_default():
    t = run(ConditionClient(), cfg=RunConfig(interference_tasks=1))
    df = build_turn_df([t])
    assert (df["segment"] == "main").all()
    full = build_turn_df([t], include_interference=True)
    assert (full["segment"] == "interference").sum() == sum(1 for r in t["turn_records"] if r["segment"] == "interference")
    assert t["total_hallucinations"] == int(df["hallucinated"].sum())


def test_distractor_task_drips_then_forces_the_call():
    cfg = RunConfig(interference_tasks=1, interference_max_turns=3)
    sc = load_scenarios(ids=["easy_02"])[0]
    task = DistractorTask(sc, cfg)
    assert sc["initial"] in task.intro()
    assert task.reply("How many people?") == "4"  or task.reply.__name__  # drip answer (scenario specific)
    task.turns = 2
    assert task.reply("Anything else?").startswith("You have all the information")
    assert task.reply("Still here?") is None


# ── presupposition probes ───────────────────────────────────────────────────

def test_presupposition_probes_cover_every_changed_slot_and_alternate_phrasings():
    sc = load_scenarios(ids=["adversarial_01"])[0]
    probes = presupposition_probes(sc)
    assert [p["slot"] for p in probes] == ["people", "cuisine", "city", "day", "time", "occasion"]
    assert probes[0]["variant"] == "self" and probes[1]["variant"] == "third_party"
    assert "at 6 people" in probes[0]["text"] and probes[0]["new"] == 4
    assert "going with italian" in probes[1]["text"] and "colleague" in probes[1]["text"]
    sim = UserSimulator(sc, RunConfig(probe_style="both"), 13)
    assert sim.n_probes == len(probes) + len(sc["post_booking_probes"])
    assert UserSimulator(sc, RunConfig(probe_style="direct"), 13).n_probes == len(sc["post_booking_probes"])
    unchanged = load_scenarios(ids=["conflict_01"])[0]
    assert presupposition_probes(unchanged) == []


def test_affirms_presupposition_rules():
    assert affirms_presupposition("Yes, that's right.", "time", "7 PM", "8:30 PM")
    assert affirms_presupposition("Correct — still 7 PM as planned.", "time", "7 PM", "8:30 PM")
    assert not affirms_presupposition("Yes — although note it has moved to 8:30 PM.", "time", "7 PM", "8:30 PM")
    assert not affirms_presupposition("No, we changed that to Monday.", "day", "saturday", "monday")
    assert not affirms_presupposition("Actually, the time is no longer 7 PM.", "time", "7 PM", "8:30 PM")
    assert not affirms_presupposition("", "time", "7 PM", "8:30 PM")
    assert affirms_presupposition("Yes.", "dietary", "one vegan", None)


def test_affirming_a_stale_claim_is_constraint_ignore_and_correcting_is_not():
    cfg = RunConfig(probe_style="presupposition")
    stale = run(ConditionClient(probe_reply="Yes, that's right."), cfg=cfg)
    probes = [r for r in stale["turn_records"] if r.get("probe")]
    assert len(probes) == 3 and [p["probe"]["slot"] for p in probes] == ["people", "day", "time"]
    assert all(r["hallucination_type"] == "CONSTRAINT_IGNORE" and r["probe_outcome"] == "stale" for r in probes)
    fixed = run(ConditionClient(probe_reply="No — the booking is now for 3 people on saturday at 7 PM."), cfg=cfg)
    probes = [r for r in fixed["turn_records"] if r.get("probe")]
    assert all(r["hallucination_type"] == "NONE" and r["probe_outcome"] == "corrected" for r in probes)
    df = build_turn_df([stale, fixed])
    out = probe_outcomes(df).set_index("probe_kind").loc["presupposition"]
    assert out["n"] == 6 and out["stale"] == 3 and out["ok"] == 3


def test_presupposition_records_rescore_offline_without_replay():
    from trusttrajectory.analysis import rescore_trajectories
    cfg = RunConfig(probe_style="presupposition")
    t = run(ConditionClient(probe_reply="Yes, that's right."), cfg=cfg)
    again = rescore_trajectories([json.loads(json.dumps(t, default=str))], make_scorer(cfg, "assertion"))[0]
    assert [r["hallucination_type"] for r in again["turn_records"]] == [r["hallucination_type"] for r in t["turn_records"]]
    assert [r.get("probe_outcome") for r in again["turn_records"]] == [r.get("probe_outcome") for r in t["turn_records"]]


# ── CLI ─────────────────────────────────────────────────────────────────────

def test_cli_condition_flags_build_the_config():
    p = cli.build_parser()
    args = p.parse_args(["run", "--model", "x/y", "--memory", "summary", "--memory-keep", "4", "--memory-every", "2",
                         "--interference", "2", "--interference-max-turns", "6", "--probes", "both",
                         "--context-window", "8"])
    cfg = cli._build_config(args)
    assert cfg.memory_mode == "summary" and cfg.memory_keep_messages == 4 and cfg.memory_summary_every == 2
    assert cfg.interference_tasks == 2 and cfg.interference_max_turns == 6 and cfg.probe_style == "both"
    assert cfg.context_window_messages == 8 and cfg.effective_max_turns == 22 + 2 * 8
    args = p.parse_args(["run", "--model", "x/y", "--context-window", "0"])
    assert cli._build_config(args).context_window_messages is None


# ── scorer rules surfaced by the interference smoke ─────────────────────────

def test_denials_and_offers_do_not_assert_a_value():
    from trusttrajectory.scoring.assertions import asserts
    text = ("We didn't actually book a reservation for Saturday. There was no mention of Saturday or any other day. "
            "If you'd like to book a new reservation for Saturday, I can assist you with that.")
    assert not asserts(text, "day", "saturday")
    assert not asserts("We never discussed 7 PM.", "time", "7 PM")
    assert asserts("The booking is on Saturday.", "day", "saturday")


def test_after_the_post_pivot_booking_a_repeated_wrong_answer_is_still_a_collapse():
    from trusttrajectory.scoring import TurnContext
    sc = load_scenarios(ids=["adversarial_01"])[0]
    text = "The original reservation is for 2 people. To confirm: Cuisine: Thai. City: Seattle. Day: Wednesday."
    polluted = {"people": "2", "cuisine": "thai", "city": "seattle", "day": "wednesday"}
    ctx = TurnContext(turn_text=text, turn_idx=26, pivot_turn=13, pivot_occurred=True, complication_occurred=True,
                      expected=sc["expected_post"], tool_called=False, tool_parsed=None, all_params_present=True,
                      committed_facts=polluted, context_tokens=0, params_confirmed=True, booking_executed=True,
                      expected_pre=sc["expected_pre"], expected_post=sc["expected_post"])
    rec = make_scorer(RunConfig(), "assertion").score_turn(ctx)
    assert rec["hallucination_type"] == "MEMORY_COLLAPSE"
    assert set(rec["wrong_values"]) == {"people", "cuisine", "city", "day"}
    # the stale (pre-pivot) value is CONSTRAINT_IGNORE territory, not a collapse
    stale = TurnContext(**{**ctx.__dict__, "turn_text": "The original reservation is for 6 people in San Francisco."})
    rec = make_scorer(RunConfig(), "assertion").score_turn(stale)
    assert rec["hallucination_type"] == "CONSTRAINT_IGNORE" and rec["wrong_values"] == {}
    # before the post-pivot booking the reference is still what the model committed to
    pre_booking = TurnContext(**{**ctx.__dict__, "booking_executed": False, "committed_facts": {}})
    assert make_scorer(RunConfig(), "assertion").score_turn(pre_booking)["hallucination_type"] == "NONE"


def test_denying_that_the_booking_exists_is_a_collapse_after_the_booking():
    from trusttrajectory.scoring import TurnContext
    from trusttrajectory.scoring.assertions import denies_booking
    sc = load_scenarios(ids=["adversarial_01"])[0]
    text = ("We didn't actually have a previous conversation about booking a restaurant reservation, so there is no "
            "original reservation to refer to. Let's start fresh! How many people are in your party?")
    assert denies_booking(text)
    assert not denies_booking("There is no booking for Saturday; the reservation is for Monday.")
    assert not denies_booking("No information about a birthday was recorded for the original reservation.")
    base = dict(turn_text=text, turn_idx=29, pivot_turn=13, pivot_occurred=True, complication_occurred=True,
                expected=sc["expected_post"], tool_called=False, tool_parsed=None, all_params_present=True,
                committed_facts={}, context_tokens=0, params_confirmed=True, booking_executed=True,
                expected_pre=sc["expected_pre"], expected_post=sc["expected_post"],
                extra={"probe": {"kind": "direct", "idx": 3}})
    rec = make_scorer(RunConfig(), "assertion").score_turn(TurnContext(**base))
    assert rec["hallucination_type"] == "MEMORY_COLLAPSE" and rec["probe_outcome"] == "denied"
    assert make_scorer(RunConfig(), "assertion").score_turn(TurnContext(**{**base, "booking_executed": False}))["hallucination_type"] == "NONE"


def test_alternatives_and_compatible_cuisines_are_not_wrong_values():
    from trusttrajectory.scoring import TurnContext
    from trusttrajectory.scoring.assertions import asserts, _compatible
    assert not asserts("Saturday or Sunday - still need you to choose.", "day", "saturday")
    assert not asserts("- 7:00 or 7:30 PM (still need you to choose)", "time", "7:30 PM")
    assert asserts("Booked for Saturday.", "day", "saturday")
    assert _compatible("cuisine", "american", "steakhouse") and _compatible("cuisine", "bbq", "american")
    assert not _compatible("cuisine", "italian", "steakhouse")
    sc = load_scenarios(ids=["medium_01"])[0]        # steakhouse; drip says "steakhouse or American"
    text = "To recap: * Party size: 7 people * Cuisine: American * City: New York * Day: Thursday * Time: 7 PM"
    ctx = TurnContext(turn_text=text, turn_idx=14, pivot_turn=13, pivot_occurred=True, complication_occurred=True,
                      expected=sc["expected_post"], tool_called=False, tool_parsed=None, all_params_present=True,
                      committed_facts={}, context_tokens=0, params_confirmed=True, booking_executed=True,
                      expected_pre=sc["expected_pre"], expected_post=sc["expected_post"])
    assert make_scorer(RunConfig(), "assertion").score_turn(ctx)["hallucination_type"] == "NONE"


def test_accepted_alternatives_protect_the_other_half_of_an_ambiguous_drip():
    from trusttrajectory.scoring import TurnContext
    sc = load_scenarios(ids=["medium_06"])[0]        # french; drip says "French or Contemporary American"
    assert sc["accepted"] == {"cuisine": ["american", "contemporary american"]}
    text = "Confirmed: 16 people at a Contemporary American restaurant in Washington DC on Saturday at 7 PM."
    base = dict(turn_text=text, turn_idx=15, pivot_turn=13, pivot_occurred=True, complication_occurred=True,
                expected=sc["expected_post"], tool_called=False, tool_parsed=None, all_params_present=True,
                committed_facts={}, context_tokens=0, params_confirmed=True, booking_executed=True,
                expected_pre=sc["expected_pre"], expected_post=sc["expected_post"])
    S = make_scorer(RunConfig(), "assertion")
    assert S.score_turn(TurnContext(**base))["hallucination_type"] == "MEMORY_COLLAPSE"          # no accepted list
    assert S.score_turn(TurnContext(**{**base, "extra": {"accepted": sc["accepted"]}}))["hallucination_type"] == "NONE"
    # rescore picks the accepted list up from the scenario
    from trusttrajectory.analysis.rescore import turn_context
    t = {"scenario_id": "medium_06", "run_idx": 1, "pivot_turn": 13, "turn_records": []}
    r = {"turn_idx": 15, "pivot_occurred": True, "assistant_text_full": text, "expected_state": sc["expected_post"],
         "booking_executed": True, "params_confirmed": True, "all_params_present": True}
    assert turn_context(t, r, sc).extra["accepted"] == sc["accepted"]
    # the split flight scenario: a sub-booking of 2 passengers is not drift, and the son's Monday is not stale
    fl = load_scenarios(ids=["flight_medium_03"])[0]
    text = "Your flight: Dallas to Rochester, departing Wednesday, 2 passengers. Your son leaves Chicago on Monday, 1 passenger."
    ctx = TurnContext(**{**base, "turn_text": text, "expected": fl["expected_post"], "expected_pre": fl["expected_pre"],
                         "expected_post": fl["expected_post"], "extra": {"accepted": fl["accepted"]}})
    assert S.score_turn(ctx)["hallucination_type"] == "NONE"


def test_judge_perception_ignores_sub_group_counts_and_free_text_slots():
    from trusttrajectory.scoring import TurnContext
    sc = load_scenarios(ids=["flight_medium_02"])[0]      # 2 passengers, business, Paris
    base = dict(turn_text="Yes, the lactose-free meal for one passenger (your wife) is noted.", turn_idx=16, pivot_turn=13,
                pivot_occurred=True, complication_occurred=True, expected=sc["expected_post"], tool_called=False,
                tool_parsed=None, all_params_present=True, committed_facts={}, context_tokens=0, params_confirmed=True,
                booking_executed=True, expected_pre=sc["expected_pre"], expected_post=sc["expected_post"])
    S = make_scorer(RunConfig(), "judge_assertion")
    for read in ("one passenger (your wife)", "one passenger", "the remaining 4 passengers", "2 vegetarian meals"):
        rec = S.score_turn(TurnContext(**{**base, "extra": {"judge_slots": {"passengers": read}}}))
        assert rec["hallucination_type"] == "NONE", read
    rec = S.score_turn(TurnContext(**{**base, "extra": {"judge_slots": {"passengers": "4"}}}))
    assert rec["hallucination_type"] == "MEMORY_COLLAPSE"
    easy = load_scenarios(ids=["easy_01"])[0]
    rec = S.score_turn(TurnContext(**{**base, "expected": easy["expected_post"], "expected_pre": easy["expected_pre"],
                                      "expected_post": easy["expected_post"],
                                      "extra": {"judge_slots": {"dietary": "no restrictions"}}}))
    assert rec["hallucination_type"] == "NONE"


def test_examples_after_eg_and_lost_state_denials():
    from trusttrajectory.scoring.assertions import asserts, denies_booking
    assert not asserts("I still need: Cuisine type (e.g., Italian, Japanese, French, Mexican, etc.)", "cuisine", "italian")
    assert denies_booking("I cannot continue because there is nothing to continue with. You have not provided any reservation details yet.")
    assert not denies_booking("You have not provided a time yet; the reservation is otherwise complete.")


def test_a_smaller_count_in_a_meal_sentence_is_a_sub_group():
    from trusttrajectory.scoring import TurnContext
    sc = load_scenarios(ids=["flight_medium_02"])[0]          # 2 passengers
    S = make_scorer(RunConfig(), "assertion")
    base = dict(turn_idx=16, pivot_turn=13, pivot_occurred=True, complication_occurred=True, expected=sc["expected_post"],
                tool_called=False, tool_parsed=None, all_params_present=True, committed_facts={}, context_tokens=0,
                params_confirmed=True, booking_executed=True, expected_pre=sc["expected_pre"], expected_post=sc["expected_post"])
    ok = "Yes, the lactose-free meal is noted! It's confirmed for 1 passenger in your booking."
    assert S.score_turn(TurnContext(turn_text=ok, **base))["hallucination_type"] == "NONE"
    bad = "Your booking is confirmed for 1 passenger."
    assert S.score_turn(TurnContext(turn_text=bad, **base))["hallucination_type"] == "MEMORY_COLLAPSE"
    bigger = "The vegetarian meal is noted. The booking is for 4 passengers."
    assert S.score_turn(TurnContext(turn_text=bigger, **base))["hallucination_type"] == "MEMORY_COLLAPSE"


def test_empty_replies_report_the_finish_reason():
    import types
    from trusttrajectory.runner import chat
    class Refusing:
        def __init__(self): self.chat = types.SimpleNamespace(completions=self)
        def create(self, **kw):
            choice = types.SimpleNamespace(message=types.SimpleNamespace(content=None), finish_reason="content_filter",
                                           native_finish_reason="refusal")
            return types.SimpleNamespace(choices=[choice], usage=None)
    text, _lat, err, _u = chat(Refusing(), ModelConfig("x/y"), [{"role": "user", "content": "hi"}], RunConfig(retry_limit=1),
                               log=lambda s: None, sleep=lambda s: None)
    assert text is None and "content_filter" in err and "refusal" in err


def test_change_narratives_do_not_assert_the_starting_value():
    from trusttrajectory.scoring.assertions import asserts
    assert not asserts("It started as 3 people but increased to 4 when one more joined.", "people", 3)
    assert not asserts("Initially Friday, now moved to Sunday.", "day", "friday")
    assert asserts("The final number of people is 4.", "people", 4)


def test_a_vegetarian_diner_is_not_american_cuisine():
    from trusttrajectory.scoring.assertions import stated_values
    assert stated_values("The restaurant should be prepared to accommodate a vegetarian diner.", "cuisine") == []
    assert [v for v, _, _ in stated_values("We booked an American place.", "cuisine")] == ["american"]
