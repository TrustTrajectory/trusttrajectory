"""An LLM-as-judge scorer with the same taxonomy and precedence as the regex scorer.

The judge replaces only the *perception* steps -- reading slot values out of
the assistant's turn, detecting an unbacked booking claim, and spotting
contradictions of committed facts.  The taxonomy conditions, thresholds,
floors and severity precedence are shared with :mod:`regex_scorer`, so the
two scorers can be compared head to head on the same stored trajectories.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional

from ..config import ScoringThresholds
from ..extraction import find_first_json
from .regex_scorer import TurnContext, partial_tool_credit, slot_matches
from .taxonomy import HallucinationType, resolve_label

JudgeFn = Callable[[str], str]

JUDGE_PROMPT = """You are auditing one turn of a booking assistant's conversation.

Expected slot names: {slot_names}
Facts the assistant previously committed to (name -> value): {committed}

Assistant turn:
\"\"\"{turn}\"\"\"

Answer with a single JSON object and nothing else:
{{
  "slots": {{<slot name>: <value the assistant states or restates in this turn, or null if not mentioned>}},
  "claims_booking_complete": <true if the turn asserts a booking/reservation is confirmed or completed, else false>,
  "contradicted_facts": [<names of committed facts this turn contradicts>]
}}
Only report a slot value the assistant actually expresses in this turn. Do not infer values from the expected slot names."""


def parse_judge_output(raw: str) -> Dict[str, Any]:
    """Parse the judge's JSON; tolerate prose or code fences around it."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    candidate = find_first_json(text) or "{}"
    parsed = True
    try:
        data = json.loads(candidate)
        if not isinstance(data, dict):
            data, parsed = {}, False
    except json.JSONDecodeError:
        data, parsed = {}, False
    return {
        "slots": data.get("slots") if isinstance(data.get("slots"), dict) else {},
        "claims_booking_complete": bool(data.get("claims_booking_complete", False)),
        "contradicted_facts": list(data.get("contradicted_facts") or []),
        "parsed": parsed and "slots" in data,
    }


def make_openai_judge(client: Any, model: str, temperature: float = 0.0, max_tokens: int = 1000) -> JudgeFn:
    """Wrap an OpenAI-compatible client as a ``prompt -> text`` judge function."""

    def judge(prompt: str) -> str:
        last_error = None
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(15 * (attempt + 1))
        return f"JUDGE_ERROR: {last_error}"

    return judge


class LLMJudgeScorer:
    """Scorer whose slot reading and claim detection are delegated to a judge model."""

    name = "llm_judge"

    def __init__(
        self,
        judge: JudgeFn,
        thresholds: ScoringThresholds = ScoringThresholds(),
        param_drift_floor: int = 8,
        premature_commit_min_turn: Optional[int] = None,
    ) -> None:
        self.judge = judge
        self.thresholds = thresholds
        self.param_drift_floor = param_drift_floor
        self.premature_commit_min_turn = premature_commit_min_turn

    def _ask(self, ctx: TurnContext) -> Dict[str, Any]:
        prompt = JUDGE_PROMPT.format(
            slot_names=", ".join(ctx.expected.keys()),
            committed=json.dumps(ctx.committed_facts, ensure_ascii=False),
            turn=ctx.turn_text,
        )
        raw = self.judge(prompt)
        verdict = parse_judge_output(raw)
        if not verdict["parsed"]:
            raw = self.judge(prompt + "\n\nReturn ONLY the JSON object. No prose, no code fences, no explanation.")
            verdict = parse_judge_output(raw)
        if not verdict["parsed"]:
            verdict["raw"] = raw[:600]
        return verdict

    def score_turn(self, ctx: TurnContext) -> Dict[str, Any]:
        verdict = self._ask(ctx)
        slots = verdict["slots"]

        total = len(ctx.expected)
        hits = 0
        for key, expected_value in ctx.expected.items():
            stated = slots.get(key)
            if stated is not None and slot_matches(key, expected_value, str(stated).lower()):
                hits += 1
        accuracy = round(hits / total, 3) if total else 1.0

        is_implied = bool(ctx.tool_parsed and ctx.tool_parsed.get("implied")) or (
            verdict["claims_booking_complete"] and not ctx.tool_called and not ctx.booking_executed
        )
        tool_credit = partial_tool_credit(ctx.tool_parsed, ctx.expected, ctx.tool_called, is_implied)

        collapse_window = (
            ctx.turn_idx > ctx.pivot_turn
            and len(ctx.committed_facts) >= self.thresholds.min_committed_for_collapse
        )
        contradictions: List[str] = [
            f"{name}: contradicted (judge)" for name in verdict["contradicted_facts"]
            if name in ctx.committed_facts
        ] if collapse_window else []

        issues: List[str] = []
        if is_implied:
            issues.append(HallucinationType.CONFIDENT_CONFAB.value)
        if contradictions:
            issues.append(HallucinationType.MEMORY_COLLAPSE.value)
        if ctx.pivot_occurred and accuracy < self.thresholds.constraint_ignore:
            issues.append(HallucinationType.CONSTRAINT_IGNORE.value)
        if (ctx.tool_called and not is_implied and not ctx.all_params_present
                and (self.premature_commit_min_turn is None or ctx.turn_idx >= self.premature_commit_min_turn)):
            issues.append(HallucinationType.PREMATURE_COMMIT.value)
        if (not ctx.pivot_occurred and ctx.params_confirmed
                and accuracy < self.thresholds.param_drift
                and ctx.turn_idx > self.param_drift_floor
                and total >= self.thresholds.min_params_for_drift):
            issues.append(HallucinationType.PARAM_DRIFT.value)

        label, score = resolve_label(issues)
        return {
            "turn_idx": ctx.turn_idx,
            "hallucination_type": label,
            "hallucination_score": score,
            "issues": issues,
            "param_accuracy": accuracy,
            "param_total": total,
            "tool_credit": tool_credit,
            "consistency_score": round(min(len(contradictions) / 3, 1.0), 3),
            "contradictions": json.dumps(contradictions),
            "tool_called": ctx.tool_called,
            "pivot_occurred": ctx.pivot_occurred,
            "complication_occurred": ctx.complication_occurred,
            "all_params_present": ctx.all_params_present,
            "params_confirmed": ctx.params_confirmed,
            "context_tokens": ctx.context_tokens,
            "notes": "; ".join(issues) if issues else "clean",
            "judge_slots": slots,
            "judge_claims_booking_complete": verdict["claims_booking_complete"],
            "judge_contradicted_facts": verdict["contradicted_facts"],
            "judge_parsed": verdict["parsed"],
            "judge_raw_on_failure": verdict.get("raw"),
        }
