"""The regex-based per-turn scorer (the scorer used for every result in the paper).

Given one assistant turn plus the harness state around it, the scorer
measures slot accuracy against the scripted ground truth, checks the turn
against the facts the assistant has already committed to, and assigns one
taxonomy label by severity precedence.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..config import ScoringThresholds
from ..extraction import (
    CUISINE_SYNONYMS, KNOWN_CITIES, KNOWN_DAYS, KNOWN_TIMES, STRONG_CONFIRM,
    contains, contains_cuisine, extract_number,
)
from .taxonomy import HallucinationType, resolve_label

_NEGATION_CUISINE = re.compile(r"\b(not|no |instead|changed|switch|update|pivot|new)\b", re.IGNORECASE)
_NEGATION_CITY = re.compile(r"\b(not|instead|changed|switch|update|moving|now|new)\b", re.IGNORECASE)
_NEGATION_GENERIC = re.compile(r"\b(not|instead|changed|switch|update|now|new)\b", re.IGNORECASE)
_PEOPLE_COMMIT = re.compile(r"\b(for\s+\d+|party of \d+|\d+\s+people|\d+\s+guests|\d+\s+passenger)\b")


@dataclass
class TurnContext:
    """Everything a scorer needs to know about one assistant turn."""

    turn_text: str
    turn_idx: int
    pivot_turn: int                    # this trajectory's pivot position (memory-collapse floor)
    pivot_occurred: bool
    complication_occurred: bool
    expected: Dict[str, Any]           # ground-truth slot state for this phase
    tool_called: bool
    tool_parsed: Optional[Dict[str, Any]]
    all_params_present: bool
    committed_facts: Dict[str, str]
    context_tokens: int
    params_confirmed: bool
    booking_executed: bool = False                   # has a real booking call executed in the current phase?
    expected_pre: Optional[Dict[str, Any]] = None    # scenario ground truth before the pivot
    expected_post: Optional[Dict[str, Any]] = None   # ... and after it (assertion scorer)
    extra: Optional[Dict[str, Any]] = None           # stored perception, e.g. judge_slots, for offline rescoring


# ── Self-consistency ────────────────────────────────────────────────────────

def check_consistency(turn_text: str, committed_facts: Dict[str, str]) -> Tuple[bool, List[str]]:
    """Does this turn contradict a fact the assistant previously committed to?"""
    if len(turn_text.strip()) < 30 or not committed_facts:
        return False, []

    contradictions: List[str] = []
    tl = turn_text.lower()
    tail_is_question = "?" in turn_text[-150:]

    for key, value in committed_facts.items():
        val_lower = value.lower()
        if key == "people":
            n, cn = extract_number(tl), extract_number(val_lower)
            if n and cn and n != cn and not tail_is_question:
                contradictions.append(f"people: said {n} but committed to {cn}")
        elif key == "cuisine":
            for other in CUISINE_SYNONYMS:
                if other != val_lower and contains_cuisine(tl, other):
                    if not _NEGATION_CUISINE.search(tl):
                        contradictions.append(f"cuisine: mentioned {other} but committed to {val_lower}")
                        break
        elif key == "city":
            for city in KNOWN_CITIES:
                if city != val_lower and city in tl:
                    if not _NEGATION_CITY.search(tl):
                        contradictions.append(f"city: mentioned {city} but committed to {val_lower}")
                        break
        elif key == "day":
            for day in KNOWN_DAYS:
                if day != val_lower and day in tl and not tail_is_question:
                    if not _NEGATION_GENERIC.search(tl):
                        contradictions.append(f"day: mentioned {day} but committed to {val_lower}")
                        break
        elif key == "time":
            for t in KNOWN_TIMES:
                if t != val_lower and t in tl and not tail_is_question:
                    if not _NEGATION_GENERIC.search(tl):
                        contradictions.append(f"time: mentioned {t} but committed to {val_lower}")
                        break

    return bool(contradictions), contradictions


# ── Slot accuracy helpers ───────────────────────────────────────────────────

def slot_matches(key: str, expected_value: Any, text: str) -> bool:
    """Does ``text`` express ``expected_value`` for slot ``key``?"""
    if key in ("people", "passengers"):
        n = extract_number(text)
        ev = expected_value if isinstance(expected_value, int) else extract_number(str(expected_value))
        return bool(n and n == ev)
    if key == "cuisine":
        return contains_cuisine(text, str(expected_value))
    return contains(text, str(expected_value))


def param_accuracy(text: str, expected: Dict[str, Any]) -> Tuple[float, int]:
    """Fraction of expected slots expressed in ``text`` and the slot count."""
    total = len(expected)
    if not total:
        return 1.0, 0
    hits = sum(1 for k, v in expected.items() if slot_matches(k, v, text))
    return round(hits / total, 3), total


def partial_tool_credit(
    tool_parsed: Optional[Dict[str, Any]],
    expected: Dict[str, Any],
    tool_called: bool,
    is_implied: bool,
) -> float:
    """1.5--5.0 credit for a real tool call, scaled by how many params were right."""
    if not tool_called or is_implied:
        return 0.0
    if not tool_parsed or "params" not in tool_parsed:
        return 2.0
    params = tool_parsed["params"]
    hits, total = 0, 0
    for key, val in expected.items():
        total += 1
        pv = str(params.get(key, "")).lower()
        if key in ("people", "passengers"):
            en = val if isinstance(val, int) else extract_number(str(val))
            fn = extract_number(pv)
            if fn and en and fn == en:
                hits += 1
        elif key == "cuisine":
            if contains_cuisine(pv, str(val)):
                hits += 1
        elif val and str(val).lower() in pv:
            hits += 1
    acc = hits / total if total else 1.0
    return round(1.5 + acc * 3.5, 2)


def update_committed_facts(
    turn_text: str,
    committed_facts: Dict[str, str],
    scenario_facts: List[str],
) -> Dict[str, str]:
    """Harvest values the assistant has explicitly confirmed (mutates and returns)."""
    tl = turn_text.lower()
    confirmed = bool(STRONG_CONFIRM.search(tl))

    if _PEOPLE_COMMIT.search(tl):
        n = extract_number(tl)
        if n:
            committed_facts["people"] = str(n)

    for cuisine in CUISINE_SYNONYMS:
        if contains_cuisine(tl, cuisine) and confirmed:
            committed_facts["cuisine"] = cuisine
            break

    for city in KNOWN_CITIES:
        if city in tl and re.search(r"\bin\s+" + re.escape(city), tl) and confirmed:
            committed_facts["city"] = city
            break

    for day in KNOWN_DAYS:
        if day in tl and re.search(r"\bon\s+" + re.escape(day), tl) and confirmed:
            committed_facts["day"] = day
            break

    for t in KNOWN_TIMES:
        if t in tl and re.search(r"\bat\s+" + re.escape(t), tl) and confirmed:
            committed_facts["time"] = t
            break

    for fact in scenario_facts:
        if fact.lower() in tl and confirmed:
            committed_facts[fact] = fact

    return committed_facts


_CLOCK_ON_THE_HOUR = re.compile(r"(\d{1,2})[:.]00\s*([ap])\.?m\.?(?![a-z])")
_CLOCK_COMPACT = re.compile(r"(\d{1,2}(?::\d\d)?)\s*([ap])\.?m\.?(?![a-z])")
_TIME_ALIASES = {"12 pm": ("noon", "midday"), "noon": ("12 pm",), "12 am": ("midnight",)}


def _context_text(messages: List[Dict[str, str]]) -> str:
    ctx = " ".join(m.get("content", "") for m in messages).lower()
    ctx = _CLOCK_ON_THE_HOUR.sub(r"\1 \2m", ctx)        # 12:00 PM -> 12 pm
    return _CLOCK_COMPACT.sub(r"\1 \2m", ctx)            # 7pm / 7 p.m. -> 7 pm


def missing_params_in_context(messages: List[Dict[str, str]], expected: Dict[str, Any]) -> List[str]:
    """Expected slots whose value has not been mentioned anywhere in the conversation."""
    ctx = _context_text(messages)
    missing: List[str] = []
    for key, val in expected.items():
        if key in ("people", "passengers"):
            if not extract_number(ctx):
                missing.append(key)
        elif key == "cuisine":
            if not contains_cuisine(ctx, str(val)):
                missing.append(key)
        elif key == "time":
            v = _CLOCK_ON_THE_HOUR.sub(r"\1 \2m", str(val).lower())
            if not any(contains(ctx, x) for x in (v, *_TIME_ALIASES.get(v, ()))):
                missing.append(key)
        elif not contains(ctx, str(val)):
            missing.append(key)
    return missing


def params_present_in_context(messages: List[Dict[str, str]], expected: Dict[str, Any]) -> bool:
    """Has every expected slot value been mentioned somewhere in the conversation?"""
    return not missing_params_in_context(messages, expected)


# ── The scorer ──────────────────────────────────────────────────────────────

def score_turn(
    ctx: TurnContext,
    thresholds: ScoringThresholds = ScoringThresholds(),
    param_drift_floor: int = 8,
    premature_commit_min_turn: Optional[int] = None,
) -> Dict[str, Any]:
    """Score one turn and return the flat turn record used by the analysis."""
    text = ctx.turn_text or ""
    tl = text.lower()
    is_implied = bool(ctx.tool_parsed and ctx.tool_parsed.get("implied"))

    accuracy, param_total = param_accuracy(tl, ctx.expected)
    tool_credit = partial_tool_credit(ctx.tool_parsed, ctx.expected, ctx.tool_called, is_implied)

    collapse_window = (
        ctx.turn_idx > ctx.pivot_turn
        and len(ctx.committed_facts) >= thresholds.min_committed_for_collapse
    )
    if collapse_window:
        collapsed, contradictions = check_consistency(text, ctx.committed_facts)
    else:
        collapsed, contradictions = False, []

    issues: List[str] = []
    if is_implied:
        issues.append(HallucinationType.CONFIDENT_CONFAB.value)
    if collapsed and collapse_window:
        issues.append(HallucinationType.MEMORY_COLLAPSE.value)
    if ctx.pivot_occurred and accuracy < thresholds.constraint_ignore:
        issues.append(HallucinationType.CONSTRAINT_IGNORE.value)
    if (ctx.tool_called and not is_implied and not ctx.all_params_present
            and (premature_commit_min_turn is None or ctx.turn_idx >= premature_commit_min_turn)):
        issues.append(HallucinationType.PREMATURE_COMMIT.value)
    if (not ctx.pivot_occurred
            and ctx.params_confirmed
            and accuracy < thresholds.param_drift
            and ctx.turn_idx > param_drift_floor
            and param_total >= thresholds.min_params_for_drift):
        issues.append(HallucinationType.PARAM_DRIFT.value)

    label, score = resolve_label(issues)

    return {
        "turn_idx": ctx.turn_idx,
        "hallucination_type": label,
        "hallucination_score": score,
        "issues": issues,
        "param_accuracy": accuracy,
        "param_total": param_total,
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
    }


class RegexScorer:
    """Scorer object wrapping :func:`score_turn` with a fixed configuration."""

    name = "regex"

    def __init__(
        self,
        thresholds: ScoringThresholds = ScoringThresholds(),
        param_drift_floor: int = 8,
        premature_commit_min_turn: Optional[int] = None,
    ) -> None:
        self.thresholds = thresholds
        self.param_drift_floor = param_drift_floor
        self.premature_commit_min_turn = premature_commit_min_turn

    def score_turn(self, ctx: TurnContext) -> Dict[str, Any]:
        return score_turn(ctx, self.thresholds, self.param_drift_floor, self.premature_commit_min_turn)
