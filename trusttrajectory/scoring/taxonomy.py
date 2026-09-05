"""The five-class agentic hallucination taxonomy.

Each turn receives exactly one label.  When several conditions fire on the
same turn, the higher-severity class wins (``PRECEDENCE``); the severity
weights govern only that precedence rule and are not used to weight any
aggregate statistic.
"""
from __future__ import annotations

from enum import Enum
from typing import Iterable, List, Tuple


class HallucinationType(str, Enum):
    NONE = "NONE"
    PARAM_DRIFT = "PARAM_DRIFT"
    PREMATURE_COMMIT = "PREMATURE_COMMIT"
    CONSTRAINT_IGNORE = "CONSTRAINT_IGNORE"
    CONFIDENT_CONFAB = "CONFIDENT_CONFAB"
    MEMORY_COLLAPSE = "MEMORY_COLLAPSE"

    def __str__(self) -> str:  # so f-strings print the bare name
        return self.value


SEVERITY = {
    HallucinationType.CONFIDENT_CONFAB: 0.95,
    HallucinationType.MEMORY_COLLAPSE: 0.90,
    HallucinationType.CONSTRAINT_IGNORE: 0.85,
    HallucinationType.PREMATURE_COMMIT: 0.80,
    HallucinationType.PARAM_DRIFT: 0.60,
}

# Highest severity first.
PRECEDENCE: List[HallucinationType] = [
    HallucinationType.CONFIDENT_CONFAB,
    HallucinationType.MEMORY_COLLAPSE,
    HallucinationType.CONSTRAINT_IGNORE,
    HallucinationType.PREMATURE_COMMIT,
    HallucinationType.PARAM_DRIFT,
]

# Display order used by the taxonomy figure and tables.
TAXONOMY_ORDER: List[str] = [
    "PARAM_DRIFT", "PREMATURE_COMMIT", "CONSTRAINT_IGNORE",
    "CONFIDENT_CONFAB", "MEMORY_COLLAPSE",
]


def resolve_label(issues: Iterable[str]) -> Tuple[str, float]:
    """Collapse the set of fired conditions to one label and its severity.

    Returns ``("NONE", 0.0)`` when nothing fired.  The score is the maximum
    severity among the fired conditions, matching the original scorer.
    """
    fired = {HallucinationType(i) for i in issues}
    if not fired:
        return HallucinationType.NONE.value, 0.0
    label = next(t for t in PRECEDENCE if t in fired)
    score = max(SEVERITY[t] for t in fired)
    return label.value, round(score, 3)
