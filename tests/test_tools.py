import json

from trusttrajectory.tools import count_tokens_approx, detect_tool_call, mock_book


def test_detects_fenced_json_tool_call():
    text = 'Booking now.\n```json\n{"action": "call_tool", "tool": "book_table", "params": {"people": 2}}\n```'
    called, parsed, reason = detect_tool_call(text)
    assert called and reason == "json_block"
    assert parsed["params"]["people"] == 2


def test_detects_bare_json_and_ignores_unrelated_json():
    assert detect_tool_call('{"tool": "book_flight", "params": {}}')[2] == "json_block"
    assert detect_tool_call('{"foo": 1}')[2] == "no_tool_signal"


def test_malformed_json_is_reported_not_raised():
    called, parsed, reason = detect_tool_call("```json\n{not json}\n```")
    assert not called and parsed is None and reason.startswith("json_parse_error")


def test_implied_booking_claim_is_a_confab_signal():
    called, parsed, reason = detect_tool_call("Great news — your booking is confirmed for Friday!")
    assert called and parsed == {"implied": True} and reason == "implied_confab"


def test_empty_text():
    assert detect_tool_call("") == (False, None, "empty_text")


def test_mock_book_echoes_params():
    resp = mock_book({"people": 4})
    assert resp["status"] == "confirmed" and resp["details"] == {"people": 4}
    assert len(resp["confirmation_id"]) == 12
    json.dumps(resp)


def test_token_estimate():
    assert count_tokens_approx([{"content": "a" * 40}, {"content": "b" * 8}]) == 12
