"""The 46-scenario suite: loading, validation and filtering.

Scenarios live as JSON in ``trusttrajectory/data/scenarios/<tier>.json`` so the
benchmark can be consumed without Python.  Each scenario has:

``id``, ``difficulty``, ``domain``, ``initial`` (the opening user message),
``clarification_drip`` (ordered regex -> answer map: one constraint revealed
per matching question), ``complication`` + ``complication_turn`` (the mid-task
complication), ``pivot`` (the hard constraint pivot), ``expected_pre`` /
``expected_post`` (ground-truth slot state before and after the pivot),
``post_booking_probes`` (questions asked after the post-pivot booking) and
``facts_to_track`` (strings monitored by the self-consistency checker).  An
optional ``accepted`` map lists, per slot, alternative values the assertion
scorer never counts as wrong or stale: the other half of an "A or B" drip, or
the sub-totals of a scenario that legitimately splits into two bookings.
"""
from __future__ import annotations

import json
import re
from importlib import resources
from typing import Dict, Iterable, List, Optional

TIERS: List[str] = ["easy", "medium", "hard", "conflicting", "adversarial"]
DOMAINS: List[str] = ["restaurant", "flight"]

REQUIRED_KEYS = {
    "id", "difficulty", "domain", "initial", "clarification_drip",
    "complication", "complication_turn", "pivot",
    "expected_pre", "expected_post", "post_booking_probes", "facts_to_track",
}

# One flight scenario per tier: a quick end-to-end check before a full run.
SMOKE_IDS = [
    "flight_easy_01", "flight_medium_01", "flight_hard_01",
    "flight_conflict_01", "flight_adversarial_01",
]

Scenario = Dict[str, object]


def validate_scenario(s: Scenario) -> None:
    missing = REQUIRED_KEYS - set(s)
    if missing:
        raise ValueError(f"scenario {s.get('id')!r} missing keys: {sorted(missing)}")
    if s["difficulty"] not in TIERS:
        raise ValueError(f"scenario {s['id']!r}: unknown tier {s['difficulty']!r}")
    if s["domain"] not in DOMAINS:
        raise ValueError(f"scenario {s['id']!r}: unknown domain {s['domain']!r}")
    if not isinstance(s["complication_turn"], int) or s["complication_turn"] < 1:
        raise ValueError(f"scenario {s['id']!r}: complication_turn must be a positive int")
    for pattern in s["clarification_drip"]:
        re.compile(pattern)
    for key in ("expected_pre", "expected_post"):
        if not isinstance(s[key], dict) or not s[key]:
            raise ValueError(f"scenario {s['id']!r}: {key} must be a non-empty mapping")
    accepted = s.get("accepted", {})
    if not isinstance(accepted, dict) or any(not isinstance(v, list) for v in accepted.values()):
        raise ValueError(f"scenario {s['id']!r}: accepted must map slots to lists of values")


def _data_files() -> Iterable:
    return resources.files("trusttrajectory.data.scenarios").iterdir()


def load_scenarios(
    tiers: Optional[Iterable[str]] = None,
    domains: Optional[Iterable[str]] = None,
    ids: Optional[Iterable[str]] = None,
) -> List[Scenario]:
    """Load scenarios in canonical tier order, optionally filtered."""
    tier_set = set(tiers) if tiers else None
    domain_set = set(domains) if domains else None
    id_set = set(ids) if ids else None

    out: List[Scenario] = []
    for tier in TIERS:
        if tier_set and tier not in tier_set:
            continue
        path = resources.files("trusttrajectory.data.scenarios").joinpath(f"{tier}.json")
        items = json.loads(path.read_text(encoding="utf-8"))
        for s in items:
            validate_scenario(s)
            if s["difficulty"] != tier:
                raise ValueError(f"{path.name}: scenario {s['id']} has tier {s['difficulty']}")
            if domain_set and s["domain"] not in domain_set:
                continue
            if id_set and s["id"] not in id_set:
                continue
            out.append(s)
    seen = set()
    for s in out:
        if s["id"] in seen:
            raise ValueError(f"duplicate scenario id {s['id']!r}")
        seen.add(s["id"])
    return out


def smoke_scenarios(scenarios: Optional[List[Scenario]] = None) -> List[Scenario]:
    scenarios = scenarios if scenarios is not None else load_scenarios()
    return [s for s in scenarios if s["id"] in SMOKE_IDS]


def tier_counts(scenarios: List[Scenario]) -> Dict[str, int]:
    return {t: sum(1 for s in scenarios if s["difficulty"] == t) for t in TIERS}
