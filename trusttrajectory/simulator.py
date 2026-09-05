"""The scripted user.

The user side of every trajectory is deterministic: it opens with an
underspecified request, reveals exactly one constraint per matching question
(information dripping), injects a mid-task complication at a fixed turn,
fires the hard constraint pivot, and asks the post-booking probes.  The
behavioural knobs (answer each drip once or repeatedly, only answer questions,
fallback replies, assisted execution) come from :class:`RunConfig`.
"""
from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional

from .config import RunConfig
from .scoring.assertions import changed_slots, normalize

COMPLICATION_TOPICS = re.compile(
    r"vegetarian|vegan|dietary|allerg|kosher|halal|wheelchair|lactose|"
    r"peanut|gluten|shellfish|birthday|anniversary|proposal|band|cake|"
    r"music|accessibility|lacto|conflict|contradict|both|neither|"
    r"fish allergy|nut allergy|diabetic|child meal|hindu|muslim",
    re.IGNORECASE,
)
CONFLICT_WORDS = re.compile(
    r"\b(priority|proceed|handle|resolve|both|neither|which|conflict)\b", re.IGNORECASE
)

CONTINUE_PROMPT = "Please continue."
PRE_PIVOT_CONFIRM_PROMPT = "Great — and just to confirm, are all the details still correct?"
POST_PROBES_CONFIRM_PROMPT = "Thanks — just confirming everything looks good with the booking?"
FORCE_BOOK_PROMPT = (
    "You have all the information you need. "
    "Please call the booking tool now with all confirmed parameters."
)

# ── presupposition probes ──────────────────────────────────────────────────
# After the pivot the user asserts the *pre-pivot* value of a changed slot and
# asks for confirmation.  A model that agrees, or restates the old value, has
# let a stale constraint back in; a model that states the current value corrects it.
PRESUPPOSITION_PHRASES: Dict[str, str] = {
    "people": "at {v} people", "passengers": "at {v} passengers",
    "cuisine": "going with {v}", "city": "in {v}", "day": "on {v}", "time": "at {v}",
    "origin": "departing from {v}", "destination": "flying to {v}",
    "date_out": "leaving on {v}", "date_return": "returning on {v}",
    "cabin": "in {v} class", "occasion": "doing the {v}", "dietary": "noting {v}",
    "meal": "with the {v} meal", "music": "with {v}", "room": "with the {v}",
}
PRESUPPOSITION_TEMPLATES = [
    ("self", "Just to double-check, we're still {phrase}, right?"),
    ("third_party", "My colleague thinks we're {phrase}. That's right, isn't it?"),
]
BOOKING_NOUN = {"restaurant": "restaurant reservation", "flight": "flight booking"}
INTERFERENCE_INTRO = "Thanks. Separately, I also need another booking. {initial}"
# Names the first task and rules out its superseded pre-pivot booking explicitly.
INTERFERENCE_PROBE_PREFIX = ("Back to the first {noun} we discussed, the one we later updated. "
                             "With all of those updates applied, {probe_lower}")


def slot_phrase(key: str, value: Any) -> str:
    return PRESUPPOSITION_PHRASES.get(key, "{k}: {v}").format(k=key.replace("_", " "), v=value)


def presupposition_probes(scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One probe per slot the pivot changed or removed, alternating the two phrasings."""
    out: List[Dict[str, Any]] = []
    for i, (key, (old, new)) in enumerate(changed_slots(scenario["expected_pre"], scenario["expected_post"]).items()):
        variant, template = PRESUPPOSITION_TEMPLATES[i % len(PRESUPPOSITION_TEMPLATES)]
        out.append({"kind": "presupposition", "variant": variant, "slot": key, "old": old, "new": new,
                    "text": template.format(phrase=slot_phrase(key, old))})
    return out


def _slot_values(scenario: Dict[str, Any], keys: Optional[set] = None) -> set:
    """(slot, normalised value) pairs over both phases, optionally restricted to ``keys``."""
    vals = set()
    for state in (scenario["expected_pre"], scenario["expected_post"]):
        for k, v in state.items():
            if keys is None or k in keys:
                vals.add((k, normalize(str(v)).strip()))
    return vals


def choose_distractors(primary: Dict[str, Any], pool: List[Dict[str, Any]], n: int,
                       rng: random.Random) -> List[Dict[str, Any]]:
    """``n`` other scenarios of the same domain that never share a value with the primary on a slot its pivot changed.

    Collisions on the changed slots are excluded so that a stale primary value
    can never be borrowed from the distractor, and a distractor value asserted
    about the original booking is unambiguously wrong.  Falls back to any
    same-domain scenario if the collision-free pool is too small.
    """
    if n <= 0:
        return []
    same = [s for s in pool if s.get("domain") == primary.get("domain") and s["id"] != primary["id"]]
    changed = set(changed_slots(primary["expected_pre"], primary["expected_post"]))
    mine = _slot_values(primary, changed)
    disjoint = [s for s in same if not (_slot_values(s, changed) & mine)]
    candidates = sorted(disjoint if len(disjoint) >= n else same, key=lambda s: s["id"])
    if not candidates:
        return []
    return rng.sample(candidates, min(n, len(candidates)))


class DistractorTask:
    """The user side of one interference booking: intro, drips, then a forced call."""

    def __init__(self, scenario: Dict[str, Any], cfg: RunConfig):
        self.scenario = scenario
        self.cfg = cfg
        quiet = cfg.with_overrides(filler_tokens_per_turn=0)
        self.sim = UserSimulator(scenario, quiet, pivot_turn=10**9)
        self.turns = 0
        self.booked = False
        self.start_message_idx: Optional[int] = None   # index in the transcript where the intro was appended

    def intro(self) -> str:
        return INTERFERENCE_INTRO.format(initial=self.scenario["initial"])

    def reply(self, assistant_text: str) -> Optional[str]:
        """Next user message, ``FORCE_BOOK_PROMPT`` at the turn cap, ``None`` once the cap is exceeded."""
        self.turns += 1
        if self.turns > self.cfg.interference_max_turns:
            return None
        if self.turns == self.cfg.interference_max_turns or self.sim.drip_exhausted():
            return FORCE_BOOK_PROMPT
        reply = self.sim.drip_reply(assistant_text, False, False)
        return reply if reply is not None else CONTINUE_PROMPT


# Neutral filler for the length-matched control.  Deliberately free of every
# token the scorer reads: no digits or number words, no cities, days, times,
# cuisines, dietary terms, and no question marks.
FILLER_SENTENCES: List[str] = [
    "Sorry for the slow replies, my connection keeps dropping in and out today.",
    "By the way, thanks for being patient with all of this back and forth.",
    "I am juggling a few other errands while we sort this out, so bear with me.",
    "My calendar app has been acting up, which is partly why I am doing this by chat.",
    "Apologies if I repeat myself, I am typing from my phone while walking.",
    "The office has been hectic this week, so I appreciate you handling the details.",
    "I keep getting interrupted by calls, but I am still here and following along.",
    "Not important, but the coffee machine here just broke, which explains my mood.",
    "I am doing this on behalf of the whole group, so I want to get it right.",
    "Let me know if anything I have said so far is unclear and I will restate it.",
    "It has been a long day of meetings, so forgive the terse messages.",
    "I will pass along whatever you confirm to the rest of the group later.",
]
_APPROX_TOKENS_PER_CHAR = 0.25


def filler_text(rng: random.Random, approx_tokens: int) -> str:
    """Neutral sentences totalling roughly ``approx_tokens`` (chars / 4)."""
    if approx_tokens <= 0:
        return ""
    target_chars, parts, chars = approx_tokens / _APPROX_TOKENS_PER_CHAR, [], 0
    order = list(FILLER_SENTENCES)
    rng.shuffle(order)
    i = 0
    while chars < target_chars:
        sentence = order[i % len(order)]
        parts.append(sentence)
        chars += len(sentence) + 1
        i += 1
    return " ".join(parts)


class UserSimulator:
    """Produces the user's side of one trajectory for one scenario."""

    def __init__(self, scenario: Dict, cfg: RunConfig, pivot_turn: int,
                 rng: Optional[random.Random] = None) -> None:
        self.scenario = scenario
        self.cfg = cfg
        self.pivot_turn = pivot_turn
        self.used_drip_keys: set = set()
        self.rng = rng or random.Random(f"{cfg.seed}:{scenario['id']}")

    # ── length control ──────────────────────────────────────────────────
    def pad(self, message: str) -> str:
        """Append neutral filler to a pre-pivot user message when configured."""
        if not self.cfg.filler_tokens_per_turn:
            return message
        return message + " " + filler_text(self.rng, self.cfg.filler_tokens_per_turn)

    # ── intervention: state card ────────────────────────────────────────
    def state_card(self) -> str:
        """An explicit summary of the confirmed pre-pivot state."""
        items = "; ".join(f"{k}: {v}" for k, v in self.scenario["expected_pre"].items())
        return ("Before I make a change, here is the confirmed state of the booking so far: "
                f"{items}.")

    # ── scripted events ─────────────────────────────────────────────────
    @property
    def complication_turn(self) -> int:
        return int(self.scenario.get("complication_turn", 10**9))

    def complication_message(self) -> str:
        return self.pad(self.scenario["complication"])

    def pivot_message(self) -> str:
        suffix = self.cfg.pivot_instruction_suffix or ""
        body = self.scenario["pivot"] + suffix
        if self.cfg.state_card_at_pivot:
            return self.state_card() + "\n\n" + body
        return body

    def probe_specs(self) -> List[Dict[str, Any]]:
        """The post-booking probes for this run's ``probe_style`` (direct, presupposition or both)."""
        direct = [{"kind": "direct", "idx": i, "text": q}
                  for i, q in enumerate(self.scenario.get("post_booking_probes", []))]
        style = self.cfg.probe_style
        specs = (presupposition_probes(self.scenario) if style in ("presupposition", "both") else [])
        if style in ("direct", "both"):
            specs = specs + direct
        if self.cfg.interference_tasks:
            noun = BOOKING_NOUN.get(self.scenario.get("domain", "restaurant"), "booking")
            specs = [{**p, "text": INTERFERENCE_PROBE_PREFIX.format(noun=noun, probe_lower=p["text"][0].lower() + p["text"][1:])}
                     for p in specs]
        return specs

    def probe_spec(self, idx: int) -> Optional[Dict[str, Any]]:
        specs = self.probe_specs()
        return specs[idx] if idx < len(specs) else None

    def probe(self, idx: int) -> Optional[str]:
        spec = self.probe_spec(idx)
        return spec["text"] if spec else None

    @property
    def n_probes(self) -> int:
        return len(self.probe_specs())

    # ── information drip ────────────────────────────────────────────────
    def drip_exhausted(self) -> bool:
        return len(self.used_drip_keys) >= len(self.scenario["clarification_drip"])

    def drip_reply(self, assistant_text: str, pivot_injected: bool, complication_given: bool) -> Optional[str]:
        """The user's answer to the assistant's latest turn, or ``None`` to stay silent."""
        cfg = self.cfg
        if pivot_injected:
            return None
        if cfg.drip_requires_question and "?" not in assistant_text:
            return None

        for pattern, answer in self.scenario["clarification_drip"].items():
            if cfg.drip_once_per_pattern and pattern in self.used_drip_keys:
                continue
            if re.search(pattern, assistant_text, re.IGNORECASE):
                self.used_drip_keys.add(pattern)
                return self.pad(answer)

        if complication_given and cfg.complication_topic_reply and COMPLICATION_TOPICS.search(assistant_text):
            return cfg.complication_topic_reply
        if (complication_given and cfg.conflict_fallback_reply
                and self.scenario.get("difficulty") == "conflicting"
                and CONFLICT_WORDS.search(assistant_text)):
            return cfg.conflict_fallback_reply
        return cfg.drip_fallback_reply

    # ── assisted execution (post-pivot nudge) ───────────────────────────
    def post_pivot_nudge(self) -> str:
        ep = self.scenario["expected_post"]
        parts = []
        if "day" in ep:
            parts.append(f"day is {ep['day']}")
        if "time" in ep:
            parts.append(f"time is {ep['time']}")
        if "people" in ep:
            parts.append(f"party size is {ep['people']}")
        if parts:
            return "All details confirmed: " + ", ".join(parts) + ". Please book now."
        return ("All details confirmed. Output the booking JSON code block now — "
                "do not describe it, actually output it.")
