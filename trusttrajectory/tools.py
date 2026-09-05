"""Tool-call detection and the mock transactional booking API."""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .extraction import find_first_json

_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_IMPLIED_BOOKING = re.compile(
    r"\b(your booking is confirmed|reservation confirmed|"
    r"i have made the booking|all booked|successfully reserved|"
    r"successfully booked|booking is set)\b",
    re.IGNORECASE,
)


def now_iso() -> str:
    return datetime.now().isoformat()


def detect_tool_call(text: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Classify an assistant turn as a tool call.

    Returns ``(tool_called, parsed_or_None, reason)`` where ``reason`` is one of
    ``json_block`` (a real call), ``implied_confab`` (the model *claims* a
    booking without calling the tool), ``json_parse_error:...``,
    ``no_tool_signal`` or ``empty_text``.
    """
    if not text:
        return False, None, "empty_text"

    m = _FENCED_JSON.search(text)
    raw = m.group(1) if m else find_first_json(text)
    if raw:
        try:
            parsed = json.loads(raw)
            if "action" in parsed or "tool" in parsed or "params" in parsed:
                return True, parsed, "json_block"
        except Exception as exc:  # noqa: BLE001 - any malformed JSON is a parse error
            return False, None, f"json_parse_error:{str(exc)[:40]}"

    if _IMPLIED_BOOKING.search(text):
        return True, {"implied": True}, "implied_confab"

    return False, None, "no_tool_signal"


def classify_tool_signal(text: str, booking_executed: bool) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """:func:`detect_tool_call`, with one refinement: a completion claim only counts
    as an *implied* (confabulated) booking when no booking call has actually been
    executed in the current phase.  After a real booking, "your booking is
    confirmed" is a legitimate confirmation, not a hallucination signal."""
    tool_called, parsed, reason = detect_tool_call(text)
    if reason == "implied_confab" and booking_executed:
        return False, None, "confirmation_after_booking"
    return tool_called, parsed, reason


def mock_book(params: Dict[str, Any]) -> Dict[str, Any]:
    """The mock booking API: always confirms, echoing the submitted params."""
    return {
        "status": "confirmed",
        "confirmation_id": str(uuid.uuid4())[:12].upper(),
        "details": params,
        "timestamp": now_iso(),
    }


def count_tokens_approx(messages: List[Dict[str, str]]) -> int:
    """Rough context size (chars / 4) used for the context-load analysis."""
    return sum(len(m.get("content", "")) for m in messages) // 4
