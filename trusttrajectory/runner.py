"""Run trajectories: one scenario at a time, with checkpoints for long runs."""
from __future__ import annotations

import json
import os
import random
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import RunConfig
from .extraction import strip_think_blocks
from .models import ModelConfig
from .prompts import SYSTEM_PROMPTS
from .scoring import Scorer, TurnContext, make_scorer, missing_params_in_context, update_committed_facts
from .memory import RollingSummaryMemory, summary_request
from .simulator import (
    CONTINUE_PROMPT, FORCE_BOOK_PROMPT, POST_PROBES_CONFIRM_PROMPT,
    PRE_PIVOT_CONFIRM_PROMPT, DistractorTask, UserSimulator, choose_distractors,
)
from .tools import classify_tool_signal, count_tokens_approx, mock_book

Logger = Callable[[str], None]
Sleeper = Callable[[float], None]

_PARAMS_CONFIRMED = re.compile(
    r"\b(confirmed|noted|have everything|all set|got it|to summarize|to confirm|so to recap)\b",
    re.IGNORECASE,
)


def _quiet(_: str) -> None:
    pass


def _backoff_seconds(attempt: int, err: Exception) -> float:
    if "429" in str(err):
        return 30 * (2 ** attempt) + random.uniform(0, 5)
    return float(2 ** attempt)


def window_messages(messages: List[Dict[str, str]], limit: Optional[int]) -> List[Dict[str, str]]:
    """Keep the system prompt plus the most recent ``limit - 1`` messages."""
    if limit is None or len(messages) <= limit:
        return messages
    return [messages[0]] + messages[-(limit - 1):]


def _usage_dict(resp: Any) -> Optional[Dict[str, int]]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None
    out = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        val = getattr(usage, key, None)
        if isinstance(val, int):
            out[key] = val
    return out or None


# Slots that must be in context before a booking call is not premature.  Optional
# features (occasion, dietary notes, music, room...) never make a call premature.
MANDATORY_SLOTS: Dict[str, Tuple[str, ...]] = {
    "restaurant": ("people", "cuisine", "city", "day", "time"),
    "flight": ("origin", "destination", "date_out", "passengers"),
}


def required_slots(domain: str, expected: Dict[str, Any], mode: str = "mandatory") -> Dict[str, Any]:
    """The expected slots a booking call must have seen (``mode`` "mandatory" or "all")."""
    if mode == "all":
        return dict(expected)
    keys = MANDATORY_SLOTS.get(domain, tuple(expected))
    return {k: v for k, v in expected.items() if k in keys}


def chat(
    client: Any,
    model_cfg: ModelConfig,
    messages: List[Dict[str, str]],
    cfg: RunConfig,
    log: Logger = print,
    sleep: Sleeper = time.sleep,
) -> Tuple[Optional[str], Optional[float], Optional[str], Optional[Dict[str, int]]]:
    """One chat completion with retries.  Returns ``(text, latency_s, error, usage)``."""
    for attempt in range(cfg.retry_limit):
        try:
            t0 = time.time()
            kwargs: Dict[str, Any] = dict(
                model=model_cfg.name,
                messages=window_messages(messages, cfg.context_window_messages),
                temperature=model_cfg.temperature,
                max_tokens=cfg.max_tokens,
                extra_headers=model_cfg.extra_headers(),
            )
            extra_body = model_cfg.extra_body()
            if extra_body:
                kwargs["extra_body"] = extra_body
            resp = client.chat.completions.create(**kwargs)
            if not getattr(resp, "choices", None):
                # OpenRouter reports upstream failures as a 200 with ``choices: null`` and an ``error`` body.
                err = getattr(resp, "error", None)
                if err is None and hasattr(resp, "model_dump"):
                    err = resp.model_dump().get("error")
                raise RuntimeError(f"no choices in response: {json.dumps(err, default=str)[:200] if err else 'unknown upstream error'}")
            choice = resp.choices[0]
            text = choice.message.content or ""
            latency = round(time.time() - t0, 3)
            if model_cfg.strip_think_tags:
                text = strip_think_blocks(text)
            if not text.strip():
                # e.g. Anthropic refusals arrive as finish_reason=content_filter / native_finish_reason=refusal
                fr = getattr(choice, "finish_reason", None)
                native = getattr(choice, "native_finish_reason", None)
                if native is None and hasattr(choice, "model_dump"):
                    native = choice.model_dump().get("native_finish_reason")
                return None, latency, f"empty response (finish_reason={fr}, native_finish_reason={native})", _usage_dict(resp)
            return text, latency, None, _usage_dict(resp)
        except Exception as exc:  # noqa: BLE001 - any transport/API error is retried
            wait = _backoff_seconds(attempt, exc)
            log(f"      [API error attempt {attempt + 1}/{cfg.retry_limit}, "
                f"retrying in {wait:.1f}s]: {str(exc)[:300]}")
            if attempt < cfg.retry_limit - 1:
                sleep(wait)
            else:
                return None, None, str(exc), None
    return None, None, "no attempts", None


_POOL: Optional[List[Dict[str, Any]]] = None


def scenario_pool() -> List[Dict[str, Any]]:
    """The full suite, loaded once: distractor bookings are drawn from it."""
    global _POOL
    if _POOL is None:
        from .scenarios import load_scenarios
        _POOL = load_scenarios()
    return _POOL


def run_scenario(
    client: Any,
    model_cfg: ModelConfig,
    scenario: Dict[str, Any],
    run_idx: int,
    cfg: RunConfig,
    scorer: Optional[Scorer] = None,
    rng: Optional[random.Random] = None,
    log: Logger = print,
    sleep: Sleeper = time.sleep,
    pool: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Drive one trajectory and return its record (turn records included).

    ``pool`` is the scenario list distractor bookings are drawn from when
    ``cfg.interference_tasks`` is set (default: the whole suite).
    """
    scorer = scorer or make_scorer(cfg)
    sid = scenario["id"]
    domain = scenario.get("domain", "restaurant")
    pivot_turn = cfg.resolve_pivot_turn(rng)
    sim = UserSimulator(scenario, cfg, pivot_turn, rng=random.Random(f"{cfg.seed}:{scenario['id']}:{run_idx}:filler"))
    memory = (RollingSummaryMemory(cfg.memory_keep_messages, cfg.memory_summary_every)
              if cfg.memory_mode == "summary" else None)
    distractors: List[DistractorTask] = []
    if cfg.interference_tasks:
        chosen = choose_distractors(scenario, pool if pool is not None else scenario_pool(), cfg.interference_tasks,
                                    random.Random(f"{cfg.seed}:{sid}:{run_idx}:interference"))
        distractors = [DistractorTask(s, cfg) for s in chosen]
    active: Optional[DistractorTask] = None
    d_idx = 0

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPTS[domain]},
        {"role": "user", "content": scenario["initial"]},
    ]
    turn_records: List[Dict[str, Any]] = []
    latencies: List[float] = []
    committed_facts: Dict[str, str] = {}
    pivot_injected = complication_given = booking_done = post_booking_phase = False
    early_booking_turn: Optional[int] = None
    probe_idx = 0
    actual_turn = 0
    params_confirmed = False
    api_error: Optional[str] = None
    exp_pre, exp_post = scenario["expected_pre"], scenario["expected_post"]
    facts = scenario.get("facts_to_track", [])
    gate_turn = cfg.effective_min_booking_turn

    log(f"\n  ── [{sid}] run {run_idx} | model={model_cfg.label()} "
        f"| domain={domain} | difficulty={scenario['difficulty']} | pivot=T{pivot_turn}"
        + (f" | memory={cfg.memory_mode}" if memory else "")
        + (f" | distractors={[d.scenario['id'] for d in distractors]}" if distractors else "")
        + (f" | probes={cfg.probe_style}" if cfg.probe_style != "direct" else ""))

    for turn_idx in range(1, cfg.effective_max_turns + 1):
        actual_turn = turn_idx
        probe_now: Optional[Dict[str, Any]] = None

        # ── user side: scripted events first ───────────────────────────
        if not complication_given and turn_idx == sim.complication_turn:
            complication_given = True
            messages.append({"role": "user", "content": sim.complication_message()})
            log(f"    [COMPLICATION T{turn_idx}]")
        elif not pivot_injected and turn_idx >= pivot_turn:
            pivot_injected = True
            messages.append({"role": "user", "content": sim.pivot_message()})
            log(f"    [PIVOT T{turn_idx}]")
        elif post_booking_phase and active is None and d_idx < len(distractors):
            active = distractors[d_idx]
            active.start_message_idx = len(messages)
            messages.append({"role": "user", "content": active.intro()})
            log(f"    [INTERFERENCE {d_idx + 1}/{len(distractors)} T{turn_idx}: {active.scenario['id']}]")
        elif post_booking_phase and active is None and probe_idx < sim.n_probes:
            probe_now = sim.probe_spec(probe_idx)
            messages.append({"role": "user", "content": probe_now["text"]})
            probe_idx += 1
        elif (cfg.confirm_prompts_after_booking and post_booking_phase
              and probe_idx >= sim.n_probes and not pivot_injected):
            messages.append({"role": "user", "content": POST_PROBES_CONFIRM_PROMPT})
        elif (cfg.confirm_prompts_after_booking and booking_done
              and not post_booking_phase and not pivot_injected):
            messages.append({"role": "user", "content": PRE_PIVOT_CONFIRM_PROMPT})

        if cfg.continue_if_assistant_last and messages[-1]["role"] == "assistant":
            messages.append({"role": "user", "content": CONTINUE_PROMPT})

        if cfg.turn_delay_s:
            sleep(cfg.turn_delay_s)

        # ── memory: fold older messages into the model's own summary ───
        summary_new: Optional[str] = None
        summary_usage: Optional[Dict[str, int]] = None
        if memory is not None and memory.needs_update(messages):
            stext, _slat, serr, summary_usage = chat(
                client, model_cfg, summary_request(memory.summary, memory.pending(messages)), cfg, log=log, sleep=sleep)
            if serr:
                api_error = serr
                log(f"    [ERROR T{turn_idx} summary]: {serr[:80]}")
                break
            memory.fold(messages, stext or "")
            summary_new = stext or ""
            log(f"    [SUMMARY #{memory.n_summaries} T{turn_idx}: {len(summary_new)} chars, keeps {len(messages) - memory.folded} msgs]")
        view = memory.view(messages) if memory is not None else messages

        # ── model turn ─────────────────────────────────────────────────
        text, latency, err, usage = chat(client, model_cfg, view, cfg, log=log, sleep=sleep)
        if err:
            api_error = err
            log(f"    [ERROR T{turn_idx}]: {err[:80]}")
            break
        if not text:
            api_error = "empty response"
            log(f"    [ERROR T{turn_idx}]: empty response")
            break
        latencies.append(latency)
        messages.append({"role": "assistant", "content": text})

        if active is not None:
            booking_executed = active.booked
        else:
            booking_executed = post_booking_phase if pivot_injected else booking_done
        tool_called, tool_parsed, detect_reason = classify_tool_signal(text, booking_executed)
        is_implied = bool(tool_parsed and tool_parsed.get("implied"))
        is_real_tool_call = tool_called and not is_implied and turn_idx >= gate_turn
        context_tokens = count_tokens_approx(view)

        if active is not None:
            # Distractor segment: scored against the distractor's own state (premature calls, confab),
            # never against the original booking, and it leaves the original's committed facts alone.
            d_scenario = active.scenario
            d_exp = d_scenario["expected_pre"]
            d_required = required_slots(d_scenario.get("domain", domain), d_exp, cfg.premature_commit_slots)
            d_missing = missing_params_in_context(messages[active.start_message_idx:], d_required)
            record = scorer.score_turn(TurnContext(
                turn_text=text, turn_idx=turn_idx, pivot_turn=pivot_turn,
                pivot_occurred=True, complication_occurred=False,
                expected=d_exp, tool_called=tool_called, tool_parsed=tool_parsed,
                all_params_present=not d_missing, committed_facts={},
                context_tokens=context_tokens, params_confirmed=False,
                booking_executed=active.booked, expected_pre=d_exp, expected_post=d_exp,
                extra={"segment": "interference", "accepted": d_scenario.get("accepted")},
            ))
            record["segment"] = "interference"
            record["distractor_id"] = d_scenario["id"]
            record["committed_facts_before"] = {}
            record["expected_state"] = d_exp
            record["missing_params"] = d_missing
        else:
            expected_now = exp_post if pivot_injected else exp_pre
            required_now = required_slots(domain, expected_now, cfg.premature_commit_slots)
            missing_now = missing_params_in_context(messages, required_now)
            all_params = not missing_now
            if not params_confirmed and all_params and _PARAMS_CONFIRMED.search(text):
                params_confirmed = True

            committed_before = dict(committed_facts)
            record = scorer.score_turn(TurnContext(
                turn_text=text, turn_idx=turn_idx, pivot_turn=pivot_turn,
                pivot_occurred=pivot_injected, complication_occurred=complication_given,
                expected=expected_now, tool_called=tool_called, tool_parsed=tool_parsed,
                all_params_present=all_params, committed_facts=committed_before,
                context_tokens=context_tokens, params_confirmed=params_confirmed,
                booking_executed=booking_executed, expected_pre=exp_pre, expected_post=exp_post,
                extra={"segment": "main", "probe": probe_now, "accepted": scenario.get("accepted")},
            ))
            record["segment"] = "main"
            record["committed_facts_before"] = committed_before
            record["expected_state"] = expected_now
            record["missing_params"] = missing_now
            if probe_now:
                record["probe"] = probe_now
        record["latency_s"] = latency
        record["assistant_text"] = text[:400]
        record["assistant_text_full"] = text          # needed to re-score offline
        record["booking_executed"] = booking_executed
        record["detect_reason"] = detect_reason
        record["usage"] = usage
        if memory is not None:
            record["memory_summary"] = summary_new     # the summary written right before this turn, if any
            record["summary_usage"] = summary_usage
            record["memory_folded_messages"] = memory.folded - 1
        turn_records.append(record)

        if not booking_done and cfg.max_stall_turns and turn_idx >= cfg.max_stall_turns:
            log(f"    [EARLY EXIT: no booking after {cfg.max_stall_turns} turns]")
            break

        log(f"    T{turn_idx:2d} | {record['hallucination_type']:20s} "
            f"| acc={record['param_accuracy']:.2f} | credit={record['tool_credit']:.1f} "
            f"| pivot={pivot_injected} | detect={detect_reason}"
            + (f" | seg=interference" if active is not None else "")
            + (f" | probe={probe_now.get('kind')}:{probe_now.get('slot', probe_now.get('idx'))}"
               f"->{record.get('probe_outcome')}" if probe_now else ""))

        if active is None:
            harvest = getattr(scorer, "harvest", None)
            committed_facts = (harvest(text, committed_facts) if harvest
                               else update_committed_facts(text, committed_facts, facts))

        # ── environment response ───────────────────────────────────────
        if active is not None:
            if is_real_tool_call:
                tool_resp = mock_book(tool_parsed.get("params", {}))
                messages.append({"role": "user", "content": "TOOL_RESPONSE: " + json.dumps(tool_resp)})
                active.booked = True
                log(f"    [Distractor booking T{turn_idx}: {active.scenario['id']}]")
                active, d_idx = None, d_idx + 1
            elif is_implied:
                log(f"    [Implied confab T{turn_idx} — state NOT advanced]")
            else:
                reply = active.reply(text)
                if reply is None:
                    tool_resp = mock_book(active.scenario["expected_pre"])
                    messages.append({"role": "user", "content": "TOOL_RESPONSE: " + json.dumps(tool_resp)})
                    active.booked = True
                    log(f"    [Distractor AUTO-BOOK T{turn_idx}: {active.scenario['id']}]")
                    active, d_idx = None, d_idx + 1
                else:
                    messages.append({"role": "user", "content": reply})
        elif is_real_tool_call and (not booking_done or pivot_injected):
            tool_resp = mock_book(tool_parsed.get("params", {}))
            messages.append({"role": "user", "content": "TOOL_RESPONSE: " + json.dumps(tool_resp)})
            booking_done = True
            if not pivot_injected:
                early_booking_turn = turn_idx
                log(f"    [Pre-pivot booking T{turn_idx} — pivot still pending]")
            else:
                post_booking_phase = True
                log(f"    [Post-pivot booking T{turn_idx}]")
        elif is_implied:
            log(f"    [Implied confab T{turn_idx} — state NOT advanced]")
        elif not post_booking_phase and not pivot_injected:
            if cfg.auto_book_on_drip_exhausted and not booking_done and sim.drip_exhausted():
                tool_resp = mock_book(exp_pre)
                messages.append({"role": "user", "content": "TOOL_RESPONSE: " + json.dumps(tool_resp)})
                booking_done = True
                early_booking_turn = turn_idx
                log(f"    [AUTO-BOOK T{turn_idx}]")
            elif cfg.force_book_before_pivot and not booking_done and turn_idx >= pivot_turn - 1:
                messages.append({"role": "user", "content": FORCE_BOOK_PROMPT})
            else:
                reply = sim.drip_reply(text, pivot_injected, complication_given)
                if reply is not None:
                    messages.append({"role": "user", "content": reply})
        elif cfg.post_pivot_nudge and pivot_injected and not post_booking_phase and "?" in text:
            messages.append({"role": "user", "content": sim.post_pivot_nudge()})

        if post_booking_phase and active is None and d_idx >= len(distractors) and probe_idx >= sim.n_probes:
            log(f"    [All probes done T{turn_idx}]")
            break

    hall_turns = [r for r in turn_records if r["hallucination_type"] != "NONE" and r.get("segment", "main") == "main"]
    fht = hall_turns[0]["turn_idx"] if hall_turns else None
    log(f"    ► turns={actual_turn} | booked={booking_done} | FHT={fht} "
        f"| hall={len(hall_turns)} | lat={round(sum(latencies), 1)}s")

    return {
        "scenario_id": sid,
        "model": model_cfg.label(),
        "run_idx": run_idx,
        "difficulty": scenario["difficulty"],
        "domain": domain,
        "pivot_turn": pivot_turn,
        "turns_used": actual_turn,
        "pivot_injected": pivot_injected,
        "complication_given": complication_given,
        "booking_done": booking_done,
        "early_booking_turn": early_booking_turn,
        "first_hall_turn": fht,
        "total_hallucinations": len(hall_turns),
        "turn_records": turn_records,
        "latency_mean": round(statistics.mean(latencies), 3) if latencies else None,
        "latency_total": round(sum(latencies), 3) if latencies else None,
        "final_committed_facts": json.dumps(committed_facts),
        "scorer": getattr(scorer, "name", "unknown"),
        "config": cfg.summary(),
        "model_config": model_cfg.summary(),
        "distractor_ids": [d.scenario["id"] for d in distractors],
        "distractors_booked": sum(1 for d in distractors if d.booked),
        "n_summaries": memory.n_summaries if memory is not None else 0,
        "probes_asked": probe_idx,
        "error": api_error,
    }


def make_tag(model_cfg: ModelConfig, mode: str, when: Optional[datetime] = None) -> str:
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M")
    safe = re.sub(r"[^a-z0-9.-]+", "-", model_cfg.label().lower())
    return f"{safe}_{mode}_{stamp}"


def _trajectory_rng(seed: int, scenario_id: str, run_idx: int) -> random.Random:
    return random.Random(f"{seed}:{scenario_id}:{run_idx}")


def run_benchmark(
    client: Any,
    model_cfg: ModelConfig,
    scenarios: List[Dict[str, Any]],
    cfg: RunConfig,
    n_runs: int = 1,
    out_dir: str = "results",
    tag: Optional[str] = None,
    scorer: Optional[Scorer] = None,
    resume: bool = True,
    workers: int = 1,
    max_trajectory_retries: int = 2,
    retry_pause_s: float = 60.0,
    log: Logger = print,
    sleep: Sleeper = time.sleep,
) -> List[Dict[str, Any]]:
    """Run every (scenario, run) pair, checkpointing after each trajectory.

    A trajectory cut short by an API error is written to ``failed_<tag>.json``
    and retried (``max_trajectory_retries`` extra passes) instead of being
    kept as data; anything still failing is left out and reported.

    With ``resume=True`` an existing ``checkpoint_<tag>.json`` in ``out_dir`` is
    loaded and completed pairs are skipped, so an interrupted run can be
    restarted with the same tag.  ``workers > 1`` runs trajectories
    concurrently; each trajectory's log is buffered and printed whole so the
    output stays readable, and results are returned in scenario order.
    """
    scorer = scorer or make_scorer(cfg)
    tag = tag or make_tag(model_cfg, "full")
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, f"checkpoint_{tag}.json")

    trajectories: List[Dict[str, Any]] = []
    if resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as f:
            trajectories = json.load(f)
        log(f"  Resuming from {checkpoint_path}: {len(trajectories)} trajectories done")
    done = {(t["scenario_id"], t["run_idx"]) for t in trajectories}
    jobs = [(scenario, run_idx) for scenario in scenarios for run_idx in range(1, n_runs + 1)
            if (scenario["id"], run_idx) not in done]
    lock = threading.Lock()

    def _checkpoint() -> None:
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(trajectories, f, default=str)

    failed: List[Dict[str, Any]] = []
    failed_path = os.path.join(out_dir, f"failed_{tag}.json")

    def _job(scenario: Dict[str, Any], run_idx: int) -> None:
        rng = _trajectory_rng(cfg.seed, scenario["id"], run_idx)
        buffer: List[str] = []
        sink = log if workers <= 1 else buffer.append
        result = run_scenario(client, model_cfg, scenario, run_idx, cfg, scorer, rng, sink, sleep)
        with lock:
            if result.get("error"):
                # keep the partial trajectory for diagnostics, but never as data
                failed.append(result)
                with open(failed_path, "w", encoding="utf-8") as f:
                    json.dump(failed, f, default=str)
            else:
                trajectories.append(result)
                _checkpoint()
            if buffer:
                log("\n".join(buffer))
        if cfg.scenario_delay_s:
            sleep(cfg.scenario_delay_s)

    def _run_jobs(pending: List[Tuple[Dict[str, Any], int]]) -> None:
        if workers <= 1:
            for scenario, run_idx in pending:
                _job(scenario, run_idx)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(lambda j: _job(*j), pending))

    _run_jobs(jobs)
    for attempt in range(max_trajectory_retries):
        completed = {(t["scenario_id"], t["run_idx"]) for t in trajectories}
        retry = [(s, r) for s, r in jobs if (s["id"], r) not in completed]
        if not retry:
            break
        log(f"  Retrying {len(retry)} trajectories that hit API errors (pass {attempt + 1}/{max_trajectory_retries})")
        sleep(retry_pause_s)
        _run_jobs(retry)
    completed = {(t["scenario_id"], t["run_idx"]) for t in trajectories}
    missing = [(s["id"], r) for s, r in jobs if (s["id"], r) not in completed]
    if missing:
        log(f"  [warning] {len(missing)} trajectories still failed after retries and are NOT in the raw file: {missing[:10]}"
            f"{' …' if len(missing) > 10 else ''}. Re-run with the same --tag to retry them.")

    order = {s["id"]: i for i, s in enumerate(scenarios)}
    trajectories.sort(key=lambda t: (order.get(t["scenario_id"], len(order)), t["run_idx"]))
    _checkpoint()
    raw_path = os.path.join(out_dir, f"raw_{tag}.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(trajectories, f, indent=2, default=str)
    log(f"  Saved: {raw_path}")
    return trajectories
