"""Regex slot extraction and synonym normalisation.

These helpers implement the "Parameter Extraction" rules of the paper
(Section 3.3): party sizes are read as digits or number words, cuisines are
matched through a synonym map, and cities / days / times are matched against
the closed vocabularies used by the scenario suite.  Partial matches count as
misses.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

NUMBER_WORDS: Dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "twenty-five": 25, "twenty five": 25,
}

# Insertion order matters: several callers stop at the first cuisine that
# matches, so keep this in the order the benchmark has always used.
CUISINE_SYNONYMS: Dict[str, List[str]] = {
    "thai":          ["thai food", "pad thai"],
    "italian":       ["pasta", "pizza", "trattoria"],
    "japanese":      ["sushi", "ramen", "izakaya", "omakase"],
    "korean":        ["korean bbq", "kbbq", "bibimbap"],
    "mediterranean": ["mezze", "hummus", "levantine"],
    "french":        ["bistro", "brasserie"],
    "chinese":       ["dim sum", "cantonese"],
    "ethiopian":     ["injera"],
    "spanish":       ["tapas", "paella"],
    "bbq":           ["barbecue", "barbeque", "smokehouse"],
    "brunch":        ["brunch spot", "eggs benedict"],
    "indian":        ["curry", "tandoori", "naan"],
    "mexican":       ["tacos", "enchiladas", "guacamole"],
    "steakhouse":    ["steak house", "steakhouse", "chop house"],
    "american":      ["burger", "diner"],
}

KNOWN_CITIES: List[str] = [
    "new york", "san francisco", "chicago", "boston", "miami", "seattle",
    "portland", "los angeles", "san jose", "austin", "dallas", "denver",
    "las vegas", "washington dc", "houston", "san antonio", "phoenix",
    "cape town", "nairobi", "singapore", "tokyo", "paris", "amsterdam",
    "london", "rochester", "zurich", "geneva", "dubai", "costa rica",
    "cancun", "dar es salaam",
]

KNOWN_DAYS: List[str] = [
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "tomorrow",
]

KNOWN_TIMES: List[str] = [
    "6 pm", "6:30 pm", "7 pm", "7:30 pm", "8 pm", "8:30 pm",
    "9 pm", "11 am", "noon", "12 pm", "1 pm", "10:30 am", "12:30 pm",
]

# Phrases that signal the assistant is committing to a value rather than
# merely mentioning it.  Used when harvesting "committed facts".
STRONG_CONFIRM = re.compile(
    r"\b(noted|confirmed|have it down|booking for|will book|"
    r"great|perfect|understood|all set|i have|got it|recorded)\b"
)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_number(text: str) -> Optional[int]:
    """Return the first 1--99 integer in ``text`` (digits first, then words)."""
    m = re.search(r"\b([1-9][0-9]?)\b", text)
    if m:
        return int(m.group(1))
    tl = text.lower()
    for word, value in sorted(NUMBER_WORDS.items(), key=lambda kv: -len(kv[0])):
        if re.search(r"\b" + re.escape(word) + r"\b", tl):
            return value
    return None


def contains(text: str, target: str) -> bool:
    """Case-insensitive substring test; an empty target never matches."""
    return bool(target) and target.lower() in text.lower()


def contains_cuisine(text: str, cuisine: str) -> bool:
    """True if ``text`` mentions ``cuisine`` or one of its synonyms."""
    if not cuisine:
        return False
    tl = text.lower()
    if cuisine.lower() in tl:
        return True
    return any(syn in tl for syn in CUISINE_SYNONYMS.get(cuisine.lower(), []))


def find_first_json(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` substring of ``text``, if any."""
    start, depth = None, 0
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}" and start is not None:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def strip_think_blocks(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks emitted by some models."""
    return _THINK_BLOCK.sub("", text).strip()
