import hashlib
import json
import random
import types

import pytest

from trusttrajectory.config import PRESETS

CITIES = ["new york", "boston", "chicago", "los angeles"]
DAYS = ["friday", "saturday", "thursday"]


class ScriptedClient:
    """OpenAI-compatible stand-in whose reply is a deterministic function of the
    conversation so far (plus a salt), so runs are reproducible in tests."""

    def __init__(self, salt: str = "t", tool_turn: int = 6, confab: bool = False):
        self.salt = salt
        self.tool_turn = tool_turn
        self.confab = confab
        self.calls = []
        self.chat = types.SimpleNamespace(completions=self)

    def _reply(self, messages):
        self.calls.append([dict(m) for m in messages])  # snapshot: the runner mutates its list
        n_assistant = sum(1 for m in messages if m["role"] == "assistant") + 1
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        # hash the conversation without tool-response payloads (they carry fresh ids/timestamps)
        stable = [m["content"] if not m["content"].startswith("TOOL_RESPONSE") else "TOOL_RESPONSE" for m in messages]
        rng = random.Random(hashlib.sha256((self.salt + json.dumps(stable)).encode()).hexdigest())
        if last_user.startswith("TOOL_RESPONSE"):
            return "Confirmed — your table for 2 people in new york on friday at 8 PM, italian, is booked. Vegetarian noted."
        if "Please update all parameters" in last_user or "call the booking tool" in last_user.lower():
            return ('```json\n{"action": "call_tool", "tool": "book_table", "params": '
                    '{"people": 3, "cuisine": "italian", "city": "new york", "day": "saturday", "time": "7 PM"}}\n```')
        if self.confab and n_assistant == 4:
            return "Your booking is confirmed for friday."
        if n_assistant >= self.tool_turn and n_assistant % 3 == 0:
            return ('```json\n{"action": "call_tool", "tool": "book_table", "params": '
                    '{"people": 2, "cuisine": "italian", "city": "new york", "day": "friday", "time": "8 PM"}}\n```')
        questions = ["How many people?", "What cuisine?", "Which city?", "What time?", "Which day?",
                     "Any dietary restrictions?", "What is your budget?", "Indoor or outdoor seating?",
                     "Any specific preference?", "Is that correct?"]
        q = questions[(n_assistant - 1) % len(questions)]
        return f"Noted, I have a table for 2 people in {rng.choice(CITIES)} on {rng.choice(DAYS)}. {q}"

    def create(self, model, messages, temperature, max_tokens, extra_headers=None, **kwargs):
        text = self._reply(messages)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))])


class FailingClient:
    def __init__(self, error="Error code: 429 - rate limited"):
        self.error = error
        self.attempts = 0
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.attempts += 1
        raise RuntimeError(self.error)


@pytest.fixture
def cfg():
    return PRESETS["v4or_t13"]


@pytest.fixture
def scripted_client():
    return ScriptedClient()


@pytest.fixture
def no_sleep():
    return lambda _s: None


@pytest.fixture
def quiet():
    return lambda _s: None
