"""Replay stored trajectories through the runner.

Every user-side message in a trajectory is a deterministic function of the
scenario, the run configuration and the assistant's turns (the simulator
answers the questions the assistant asks).  Feeding the stored assistant
texts back through :func:`trusttrajectory.runner.run_scenario` therefore
rebuilds the whole conversation offline, so harness-side quantities that were
not stored (which slots were in context before a booking call, committed
facts, the pivot bookkeeping) can be recomputed with current code, and any
scorer can be applied, without new API calls.
"""
from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

from .config import RunConfig
from .models import ModelConfig
from .runner import run_scenario
from .scenarios import load_scenarios
from .scoring import Scorer

_CARRY_TURN = ("latency_s", "usage", "summary_usage", "judge_slots", "judge_claims_booking_complete",
               "judge_contradicted_facts", "judge_parsed")


class ReplayExhausted(RuntimeError):
    """The runner asked for more turns than the stored trajectory has."""


class ReplayClient:
    """OpenAI-compatible stand-in that returns stored assistant turns in order."""

    def __init__(self, texts: Iterable[str]):
        self.texts = list(texts)
        self.i = 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: Any) -> Any:
        if self.i >= len(self.texts):
            raise ReplayExhausted(f"stored trajectory has only {len(self.texts)} turns")
        text = self.texts[self.i]
        self.i += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))], usage=None)


def config_for(t: Dict[str, Any], base: RunConfig) -> RunConfig:
    """``base`` with the per-run settings stored in the trajectory applied (pivot turn, controls)."""
    stored = t.get("config") or {}
    fields = {f.name for f in dataclasses.fields(RunConfig)}
    over = {k: v for k, v in stored.items() if k in fields and k not in ("thresholds", "pivot_turn")}
    over["pivot_turn"] = int(t.get("pivot_turn") or base.resolve_pivot_turn(None))
    return dataclasses.replace(base, **over)


def model_config_for(t: Dict[str, Any]) -> ModelConfig:
    mc = t.get("model_config") or {}
    return ModelConfig(name=mc.get("name") or t.get("model", "replay"), display_name=t.get("model"))


def stored_assistant_texts(t: Dict[str, Any]) -> List[str]:
    return [r["assistant_text_full"] if r.get("assistant_text_full") is not None else r.get("assistant_text", "")
            for r in t["turn_records"]]


def stored_texts(t: Dict[str, Any]) -> List[str]:
    """Every model output in call order: a turn's memory summary (if one was written) precedes its reply."""
    out: List[str] = []
    for r, text in zip(t["turn_records"], stored_assistant_texts(t)):
        if r.get("memory_summary") is not None:
            out.append(r["memory_summary"])
        out.append(text)
    return out


def replay_trajectory(t: Dict[str, Any], scenario: Dict[str, Any], scorer: Scorer, base_cfg: RunConfig) -> Dict[str, Any]:
    """Rebuild one trajectory from its stored assistant turns and score it with ``scorer``."""
    texts = stored_assistant_texts(t)
    cfg = config_for(t, base_cfg)
    new = run_scenario(ReplayClient(stored_texts(t)), model_config_for(t), scenario, int(t.get("run_idx", 1)), cfg,
                       scorer, log=lambda s: None, sleep=lambda s: None)
    recs = new["turn_records"]
    mismatch = (len(recs) != len(texts) or any(r.get("assistant_text_full") != x for r, x in zip(recs, texts))
                or bool(new.get("error") and not t.get("error")))   # the replay ran out of stored turns
    for old, rec in zip(t["turn_records"], recs):
        for key in _CARRY_TURN:
            if key in old:
                rec[key] = old[key]
    new.update({
        "latency_mean": t.get("latency_mean"),
        "latency_total": t.get("latency_total"),
        "model_config": t.get("model_config"),
        "error": t.get("error") or (new.get("error") if mismatch else None),
        "replayed_from": t.get("scorer"),
        "replay_mismatch": mismatch,
    })
    return new


def replay_trajectories(
    trajectories: List[Dict[str, Any]],
    scorer: Scorer,
    base_cfg: RunConfig,
    scenarios: Optional[Iterable[Dict[str, Any]]] = None,
    workers: int = 1,
) -> List[Dict[str, Any]]:
    by_id = {s["id"]: s for s in (scenarios if scenarios is not None else load_scenarios())}

    def one(t: Dict[str, Any]) -> Dict[str, Any]:
        scenario = by_id.get(t["scenario_id"])
        if scenario is None:
            raise KeyError(f"unknown scenario {t['scenario_id']!r}")
        return replay_trajectory(t, scenario, scorer, base_cfg)

    if workers <= 1:
        out = [one(t) for t in trajectories]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            out = list(pool.map(one, trajectories))
    n_bad = sum(1 for t in out if t.get("replay_mismatch"))
    if n_bad:
        print(f"  [warning] {n_bad} trajectories did not replay turn-for-turn (marked replay_mismatch)", flush=True)
    return out
