"""Run configuration: turn budget, pivot position, execution gate, thresholds.

The paper's original runs used three harness configurations that differed in a
handful of knobs.  Those knobs are all fields of :class:`RunConfig` and the
three configurations are preserved as :data:`PRESETS` so any published run can
be reproduced exactly.  New controls requested by reviewers --
randomised pivot position and a no-gate variant -- are exposed on the same
object (``pivot_turn=(lo, hi)`` and ``gate_mode="none"``).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple, Union

PivotSpec = Union[int, Tuple[int, int]]

MEMORY_MODES = ("full", "summary")
PROBE_STYLES = ("direct", "presupposition", "both")

DEFAULT_PIVOT_SUFFIX = (
    "\n\nPlease update all parameters and call the booking tool "
    "again with the revised details."
)


@dataclass(frozen=True)
class ScoringThresholds:
    """Operational thresholds of the taxonomy (Section 3.3 of the paper)."""

    constraint_ignore: float = 0.40   # post-pivot param accuracy below this
    param_drift: float = 0.50         # pre-pivot param accuracy below this
    min_committed_for_collapse: int = 3
    min_params_for_drift: int = 3


@dataclass(frozen=True)
class RunConfig:
    # ── trajectory shape ────────────────────────────────────────────────
    max_turns: int = 22
    pivot_turn: PivotSpec = 13          # int, or an inclusive (lo, hi) range sampled per trajectory
    max_stall_turns: Optional[int] = 25  # abandon a trajectory with no booking by this turn

    # ── execution gate ──────────────────────────────────────────────────
    gate_mode: str = "hard"             # "hard": block tool execution before min_booking_turn; "none": never block
    min_booking_turn: int = 4
    premature_commit_respects_gate: bool = False  # only flag PREMATURE_COMMIT at/after min_booking_turn
    premature_commit_slots: str = "mandatory"     # "mandatory": party size, place, date/time (and cuisine) must be in
                                                  # context before a call; "all": every expected slot (the original rule)

    # ── user-simulator behaviour ────────────────────────────────────────
    pivot_instruction_suffix: Optional[str] = DEFAULT_PIVOT_SUFFIX
    drip_once_per_pattern: bool = True          # each drip answer is given at most once
    drip_requires_question: bool = False        # only answer when the assistant asked something
    drip_fallback_reply: Optional[str] = "Please continue."
    complication_topic_reply: Optional[str] = None
    conflict_fallback_reply: Optional[str] = None
    continue_if_assistant_last: bool = True     # never send two assistant turns back to back
    confirm_prompts_after_booking: bool = True
    auto_book_on_drip_exhausted: bool = True    # pre-pivot administrative advancement
    force_book_before_pivot: bool = True        # nudge to book at pivot-1 if still unbooked
    post_pivot_nudge: bool = False              # assisted execution after the pivot
    # ── experimental controls / interventions ───────────────────────────
    filler_tokens_per_turn: int = 0             # length control: neutral filler appended to pre-pivot user turns
    state_card_at_pivot: bool = False           # intervention: restate the confirmed state right before the pivot

    # ── harder conditions (post-review extension) ───────────────────────
    # memory: "full" sends the (windowed) history; "summary" keeps a rolling summary that the model under
    # test writes itself plus the last ``memory_keep_messages`` messages, re-summarising once
    # ``memory_summary_every`` further messages have accumulated (ConversationSummaryBuffer-style).
    memory_mode: str = "full"
    memory_keep_messages: int = 6
    memory_summary_every: int = 4
    # interference: after the post-pivot booking, the user runs this many unrelated bookings (other
    # scenarios of the same domain with non-colliding values) before the probes about the original booking.
    interference_tasks: int = 0
    interference_max_turns: int = 8             # per distractor booking, before the user forces the call
    # probes: "direct" asks the scenario's questions; "presupposition" has the user assert the pre-pivot
    # value of each changed slot ("we're still at 7 PM, right?"); "both" asks presupposition probes first.
    probe_style: str = "direct"

    # ── scoring floors ──────────────────────────────────────────────────
    param_drift_floor: Optional[int] = None     # default: min_booking_turn + 4
    thresholds: ScoringThresholds = field(default_factory=ScoringThresholds)

    # ── API / decoding ──────────────────────────────────────────────────
    temperature: float = 0.2
    max_tokens: int = 1024
    retry_limit: int = 4
    context_window_messages: Optional[int] = 20  # keep system + last (n-1) messages; None = full history
    turn_delay_s: float = 0.0
    scenario_delay_s: float = 3.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.gate_mode not in ("hard", "none"):
            raise ValueError(f"gate_mode must be 'hard' or 'none', got {self.gate_mode!r}")
        if isinstance(self.pivot_turn, tuple):
            lo, hi = self.pivot_turn
            if not (1 <= lo <= hi <= self.max_turns):
                raise ValueError(f"pivot range {self.pivot_turn} must satisfy 1 <= lo <= hi <= max_turns")
        elif not (1 <= self.pivot_turn <= self.max_turns):
            raise ValueError(f"pivot_turn {self.pivot_turn} must be within 1..max_turns")
        if self.filler_tokens_per_turn < 0:
            raise ValueError("filler_tokens_per_turn must be >= 0")
        if self.memory_mode not in MEMORY_MODES:
            raise ValueError(f"memory_mode must be one of {MEMORY_MODES}, got {self.memory_mode!r}")
        if self.memory_keep_messages < 1 or self.memory_summary_every < 1:
            raise ValueError("memory_keep_messages and memory_summary_every must be >= 1")
        if self.interference_tasks < 0 or self.interference_max_turns < 1:
            raise ValueError("interference_tasks must be >= 0 and interference_max_turns >= 1")
        if self.probe_style not in PROBE_STYLES:
            raise ValueError(f"probe_style must be one of {PROBE_STYLES}, got {self.probe_style!r}")

    # ── derived values ──────────────────────────────────────────────────
    def resolve_pivot_turn(self, rng: Optional[random.Random] = None) -> int:
        """Fixed pivot, or one sampled from the configured range."""
        if isinstance(self.pivot_turn, tuple):
            lo, hi = self.pivot_turn
            return (rng or random).randint(lo, hi)
        return self.pivot_turn

    @property
    def effective_param_drift_floor(self) -> int:
        return self.min_booking_turn + 4 if self.param_drift_floor is None else self.param_drift_floor

    @property
    def effective_min_booking_turn(self) -> int:
        return 1 if self.gate_mode == "none" else self.min_booking_turn

    @property
    def effective_max_turns(self) -> int:
        """The turn budget, extended for the distractor bookings of the interference condition."""
        return self.max_turns + self.interference_tasks * (self.interference_max_turns + 2)

    def with_overrides(self, **kwargs) -> "RunConfig":
        return replace(self, **kwargs)

    def summary(self) -> Dict[str, object]:
        return {
            "max_turns": self.max_turns,
            "pivot_turn": self.pivot_turn,
            "gate_mode": self.gate_mode,
            "min_booking_turn": self.min_booking_turn,
            "param_drift_floor": self.effective_param_drift_floor,
            "context_window_messages": self.context_window_messages,
            "auto_book_on_drip_exhausted": self.auto_book_on_drip_exhausted,
            "force_book_before_pivot": self.force_book_before_pivot,
            "post_pivot_nudge": self.post_pivot_nudge,
            "premature_commit_slots": self.premature_commit_slots,
            "filler_tokens_per_turn": self.filler_tokens_per_turn,
            "state_card_at_pivot": self.state_card_at_pivot,
            "memory_mode": self.memory_mode,
            "memory_keep_messages": self.memory_keep_messages,
            "memory_summary_every": self.memory_summary_every,
            "interference_tasks": self.interference_tasks,
            "interference_max_turns": self.interference_max_turns,
            "probe_style": self.probe_style,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "thresholds": {
                "constraint_ignore": self.thresholds.constraint_ignore,
                "param_drift": self.thresholds.param_drift,
            },
        }


# Presets reproduce the three harness configurations the paper's runs came from.
PRESETS: Dict[str, RunConfig] = {
    # The v4-OpenRouter harness (pivot at T13); the default configuration.
    "v4or_t13": RunConfig(),
    # Same turn budget, with post-pivot assisted execution
    # (nudges), no pre-pivot auto-book / force-book, and a shorter output cap.
    "v4or_t13_nudge": RunConfig(
        max_turns=22, pivot_turn=13, min_booking_turn=4, max_stall_turns=None,
        premature_commit_respects_gate=True,
        pivot_instruction_suffix=None,
        drip_once_per_pattern=False, drip_requires_question=True,
        drip_fallback_reply="Please go ahead and proceed.",
        conflict_fallback_reply=(
            "Please flag the conflict in your notes and proceed "
            "with the booking as stated."
        ),
        continue_if_assistant_last=False,
        auto_book_on_drip_exhausted=False, force_book_before_pivot=False,
        post_pivot_nudge=True,
        max_tokens=700, retry_limit=3, context_window_messages=None,
        turn_delay_s=2.0,
    ),
    # The 30-turn variant with the pivot at T15.
    "v4or_t15": RunConfig(
        max_turns=30, pivot_turn=15, min_booking_turn=8, max_stall_turns=None,
        pivot_instruction_suffix=None,
        drip_once_per_pattern=False, drip_requires_question=True,
        drip_fallback_reply=None,
        complication_topic_reply=(
            "Please note the requirement and proceed with the best available option."
        ),
        continue_if_assistant_last=False, confirm_prompts_after_booking=False,
        auto_book_on_drip_exhausted=False, force_book_before_pivot=False,
        max_tokens=700, retry_limit=3, context_window_messages=None,
    ),
}


def get_preset(name: str) -> RunConfig:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}") from exc
