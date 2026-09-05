"""Per-turn scorers.  Both implement ``score_turn(TurnContext) -> dict``."""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from ..config import RunConfig
from .assertions import AssertionScorer, make_assertion_scorer, make_judge_assertion_scorer
from .llm_judge import JudgeFn, LLMJudgeScorer, make_openai_judge
from .regex_scorer import (
    RegexScorer, TurnContext, check_consistency, missing_params_in_context, param_accuracy,
    params_present_in_context, partial_tool_credit, score_turn,
    update_committed_facts,
)
from .taxonomy import PRECEDENCE, SEVERITY, TAXONOMY_ORDER, HallucinationType, resolve_label


class Scorer(Protocol):
    name: str

    def score_turn(self, ctx: TurnContext) -> Dict[str, Any]: ...


def make_scorer(cfg: RunConfig, kind: str = "regex", judge: Optional[JudgeFn] = None) -> Scorer:
    """Build a scorer whose floors and thresholds come from ``cfg``."""
    premature_floor = cfg.effective_min_booking_turn if cfg.premature_commit_respects_gate else None
    if kind == "regex":
        return RegexScorer(cfg.thresholds, cfg.effective_param_drift_floor, premature_floor)
    if kind == "llm_judge":
        if judge is None:
            raise ValueError("llm_judge scorer needs a judge function (see make_openai_judge)")
        return LLMJudgeScorer(judge, cfg.thresholds, cfg.effective_param_drift_floor, premature_floor)
    if kind == "assertion":
        return make_assertion_scorer(cfg.thresholds.min_committed_for_collapse, premature_floor)
    if kind == "judge_assertion":
        return make_judge_assertion_scorer(cfg.thresholds.min_committed_for_collapse, premature_floor)
    raise ValueError(f"unknown scorer {kind!r}; expected one of {sorted(SCORER_KINDS)}")


SCORER_KINDS = ("regex", "llm_judge", "assertion", "judge_assertion")


__all__ = [
    "Scorer", "TurnContext", "RegexScorer", "LLMJudgeScorer", "AssertionScorer", "make_scorer", "SCORER_KINDS",
    "make_openai_judge", "score_turn", "check_consistency", "param_accuracy",
    "partial_tool_credit", "update_committed_facts", "params_present_in_context", "missing_params_in_context",
    "HallucinationType", "SEVERITY", "PRECEDENCE", "TAXONOMY_ORDER", "resolve_label",
]
