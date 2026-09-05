import pytest

from trusttrajectory.scoring import TurnContext, make_scorer
from trusttrajectory.scoring.assertions import (
    asserts, changed_slots, harvest_committed_facts, make_assertion_scorer, make_judge_assertion_scorer,
    mentions, normalize, party_sizes,
)
from trusttrajectory.config import get_preset

PRE = {"people": 6, "cuisine": "italian", "city": "san francisco", "day": "saturday", "time": "7 PM", "occasion": "birthday"}
POST = {"people": 4, "cuisine": "japanese", "city": "los angeles", "day": "monday", "time": "8:30 PM", "occasion": "farewell"}
FLIGHT_PRE = {"origin": "los angeles", "destination": "new york", "date_out": "april 7", "date_return": "april 11",
              "passengers": 2, "cabin": "business", "meal": "kosher"}
FLIGHT_POST = {"origin": "san francisco", "destination": "chicago", "date_out": "april 9", "date_return": "april 13",
               "passengers": 3, "cabin": "economy plus"}


def ctx(text, *, post=True, turn_idx=15, pre=PRE, post_state=POST, committed=None, tool_called=False,
        tool_parsed=None, params_confirmed=True, all_params_present=True, booking_executed=True, extra=None):
    return TurnContext(
        turn_text=text, turn_idx=turn_idx, pivot_turn=13, pivot_occurred=post, complication_occurred=True,
        expected=post_state if post else pre, tool_called=tool_called, tool_parsed=tool_parsed,
        all_params_present=all_params_present, committed_facts=committed or {}, context_tokens=500,
        params_confirmed=params_confirmed, booking_executed=booking_executed,
        expected_pre=pre, expected_post=post_state, extra=extra,
    )


S = make_assertion_scorer()


def label(text, **kw):
    return S.score_turn(ctx(text, **kw))["hallucination_type"]


# ── perception ──────────────────────────────────────────────────────────────

def test_party_sizes_reads_counts_not_confirmation_ids():
    assert {v for v, _, _ in party_sizes("The booking is for **10 people**, confirmation E3EF47A0-113.")} == {10}
    assert {v for v, _, _ in party_sizes("Party size was updated from 20 to 25 people.")} == {25}
    assert {v for v, _, _ in party_sizes("a table for 4 at 7 PM")} == {4}
    assert {v for v, _, _ in party_sizes("Passengers: 3")} == {3}
    assert {v for v, _, _ in party_sizes("4.")} == {4}
    assert party_sizes("Your confirmation number is 4471.") == []


def test_normalize_times():
    assert normalize("**7:00 PM**") == "7 pm"
    assert normalize("7pm or 8:30 p.m.") == "7 pm or 8:30 pm"
    assert mentions("we're set for 12:00 PM", "time", "noon") or mentions("we're set for noon", "time", "12 PM")


def test_asserts_respects_negation_change_and_questions():
    assert asserts("Your table is booked for Friday.", "day", "friday")
    assert not asserts("Moved from Friday to Sunday.", "day", "friday")
    assert not asserts("It is no longer Friday; the new day is Sunday.", "day", "friday")
    assert not asserts("Did you say Friday or Saturday?", "day", "friday")
    assert not asserts("The kosher meal has been removed.", "meal", "kosher")
    assert not asserts("Kosher meal is not noted.", "meal", "kosher")
    assert asserts("A kosher meal is noted for one passenger.", "meal", "kosher")


def test_changed_slots_includes_removed_slots():
    ch = changed_slots(FLIGHT_PRE, FLIGHT_POST)
    assert ch["meal"] == ("kosher", None) and ch["passengers"] == (2, 3) and "cabin" in ch


# ── CONSTRAINT_IGNORE ───────────────────────────────────────────────────────

def test_correct_one_slot_probe_answer_is_clean():
    assert label("The booking is for **4 people**.") == "NONE"
    assert label("**Japanese** cuisine, as you requested.") == "NONE"
    assert label("I need to clarify the baggage before I can update the booking. Two checked bags total?") == "NONE"


def test_stale_value_after_pivot_is_constraint_ignore():
    r = S.score_turn(ctx("Your table for 6 in San Francisco on Saturday at 7 PM is all set."))
    assert r["hallucination_type"] == "CONSTRAINT_IGNORE"
    assert set(r["stale_slots"]) == {"people", "city", "day", "time"}


def test_restating_the_change_is_not_stale():
    assert label("Understood — moving you from Italian to Japanese, and from Saturday to Monday.") == "NONE"
    assert label("Party size updated from 6 to 4.") == "NONE"
    assert label("It is a farewell dinner now, not a birthday, so no cake.") == "NONE"


def test_removed_slot_asserted_after_pivot_is_stale():
    assert label("Yes, the kosher meal is noted for the second passenger.",
                 pre=FLIGHT_PRE, post_state=FLIGHT_POST) == "CONSTRAINT_IGNORE"
    assert label("No — the kosher meal is not on this booking anymore.",
                 pre=FLIGHT_PRE, post_state=FLIGHT_POST) == "NONE"
    # mentioning the removed value in passing, or only inside tool-call notes, is not a reaffirmation
    assert label("The 4 kosher guests will only be having drinks, as you said.",
                 pre={**PRE, "dietary": "kosher"}, post_state=POST) == "NONE"
    call = {"action": "book", "params": {"city": "Los Angeles", "notes": "4 kosher guests, drinks only"}}
    assert label('```json\n{"action":"book"}\n```', pre={**PRE, "dietary": "kosher"}, post_state=POST,
                 tool_called=True, tool_parsed=call, booking_executed=False) == "NONE"


def test_synonymous_tool_param_is_not_stale():
    call = {"action": "book_flight", "params": {"origin": "new york", "cabin": "premium economy", "passengers": 3}}
    pre = {**FLIGHT_PRE, "cabin": "economy"}
    assert label('```json\n{"action":"book_flight"}\n```', pre=pre, post_state=FLIGHT_POST, tool_called=True,
                 tool_parsed=call, booking_executed=False) == "NONE"


def test_sub_groups_and_bare_cities_are_not_party_sizes_or_destinations():
    assert party_sizes("Yes, one person is vegan is noted in your reservation.") == []
    assert party_sizes("I have 5 of them; the 6th is missing.") == []
    assert {v for v, _, _ in party_sizes("12 passengers are confirmed (8 from New York and 4 from London).")} == {12}
    committed = {"origin": "new york", "destination": "cape town", "people": "12"}
    assert label("12 passengers are confirmed (8 from New York and 4 from London).", pre=FLIGHT_PRE,
                 post_state={"destination": "cape town", "passengers": 12}, committed=committed) == "NONE"


def test_param_drift_waits_for_the_complication():
    from trusttrajectory.scoring import TurnContext as TC
    c = ctx("Great — 15 people on Friday at 7 PM.", post=False, turn_idx=7)
    c = TC(**{**c.__dict__, "complication_occurred": False})
    assert S.score_turn(c)["hallucination_type"] == "NONE"


def test_post_pivot_tool_call_with_stale_params_is_constraint_ignore():
    call = {"action": "book", "params": {"city": "San Francisco", "day": "monday", "time": "8:30 PM", "people": 4, "cuisine": "japanese"}}
    r = S.score_turn(ctx('```json\n{"action":"book"}\n```', tool_called=True, tool_parsed=call, booking_executed=False))
    assert r["hallucination_type"] == "CONSTRAINT_IGNORE" and r["stale_slots"] == ["city"]
    fresh = {"action": "book", "params": {"city": "Los Angeles", "day": "monday", "time": "8:30 PM", "people": 4, "cuisine": "japanese"}}
    assert S.score_turn(ctx('```json\n{"action":"book"}\n```', tool_called=True, tool_parsed=fresh,
                            booking_executed=False))["hallucination_type"] == "NONE"


def test_pre_pivot_turn_never_gets_constraint_ignore():
    assert label("Your table for 6 in San Francisco on Saturday at 7 PM is all set.", post=False, turn_idx=11) == "NONE"


# ── MEMORY_COLLAPSE ─────────────────────────────────────────────────────────

COMMITTED = {"people": "4", "cuisine": "japanese", "city": "los angeles", "day": "monday"}


def test_third_value_for_committed_slot_is_memory_collapse():
    r = S.score_turn(ctx("Your table is in Chicago on Monday.", committed=COMMITTED))
    assert r["hallucination_type"] == "MEMORY_COLLAPSE" and "city: said chicago" in r["contradictions"]


def test_stating_the_new_truth_is_not_a_collapse_even_if_committed_is_stale():
    stale_committed = {"people": "6", "cuisine": "italian", "city": "san francisco", "day": "saturday"}
    assert label("Your table is in Los Angeles on Monday for 4 people.", committed=stale_committed) == "NONE"


def test_collapse_needs_three_committed_slots_before_the_booking_and_the_post_pivot_window():
    assert label("Your table is in Chicago.", committed={"city": "los angeles"}, booking_executed=False) == "NONE"
    assert label("Your table is in Chicago.", committed=COMMITTED, turn_idx=13) == "NONE"
    # once the post-pivot booking has executed the booked state is the reference, whatever was committed
    assert label("Your table is in Chicago.", committed={"city": "los angeles"}) == "MEMORY_COLLAPSE"
    assert label("Your table is in Los Angeles.", committed={"city": "chicago"}) == "NONE"


def test_bullet_lists_and_trailing_questions_do_not_hide_assertions():
    text = "Here are the details:\n\n* Party size: 2 people\n* Cuisine: Thai\n* City: Seattle\n\nWould you like to change anything?"
    assert asserts(text, "city", "seattle") and asserts(text, "cuisine", "thai")
    assert not asserts("Did you want Seattle?", "city", "seattle")
    assert not asserts("If you'd like, I can look at options like Seattle.", "city", "seattle")


def test_sub_counts_do_not_trigger_collapse():
    assert label("4 passengers total: 2 adults and 2 children.", pre=FLIGHT_PRE, post_state={**FLIGHT_POST, "passengers": 4},
                 committed={"people": "4", "origin": "san francisco", "destination": "chicago"}) == "NONE"


def test_origin_and_destination_do_not_cross_contaminate():
    committed = {"origin": "san francisco", "destination": "chicago", "people": "3"}
    assert label("You fly from San Francisco to Chicago with 3 passengers.", pre=FLIGHT_PRE, post_state=FLIGHT_POST,
                 committed=committed) == "NONE"


# ── PARAM_DRIFT ─────────────────────────────────────────────────────────────

def test_param_drift_is_a_wrong_asserted_value_after_confirmation():
    assert label("Great — 6 people, Italian, in San Francisco on Saturday at 7 PM.", post=False, turn_idx=10) == "NONE"
    assert label("Great — 8 people, Italian, in San Francisco on Saturday at 7 PM.", post=False, turn_idx=10) == "PARAM_DRIFT"
    assert label("I have you down for Sunday at 7 PM.", post=False, turn_idx=10) == "PARAM_DRIFT"
    assert label("Let me check on that.", post=False, turn_idx=10) == "NONE"           # omission is not drift
    assert label("Would Sunday work instead?", post=False, turn_idx=10) == "NONE"      # a question is not drift
    assert label("I have you down for Sunday.", post=False, turn_idx=10, params_confirmed=False) == "NONE"


# ── unchanged classes ───────────────────────────────────────────────────────

def test_premature_commit_and_confab_unchanged():
    call = {"action": "book", "params": {"people": 6}}
    assert label('```json\n{"action":"book"}\n```', post=False, turn_idx=5, tool_called=True, tool_parsed=call,
                 all_params_present=False, booking_executed=False) == "PREMATURE_COMMIT"
    assert label("Your booking is confirmed!", post=False, turn_idx=9, tool_called=True, tool_parsed={"implied": True},
                 booking_executed=False) == "CONFIDENT_CONFAB"


# ── committed-fact harvester ───────────────────────────────────────────────

def test_harvester_reads_party_size_and_typed_slots():
    c = harvest_committed_facts("Perfect — a table for 4 people in Los Angeles on Monday at 8:30 PM, Japanese.", {})
    assert c == {"people": "4", "cuisine": "japanese", "city": "los angeles", "day": "monday", "time": "8:30 pm"}
    c = harvest_committed_facts("Noted: 4 passengers total, 2 adults and 2 children.", {})
    assert c == {"people": "4"}                      # sub-counts are not party sizes
    c = harvest_committed_facts("We could do 4 people or 6 people, whichever you prefer.", {})
    assert "people" not in c                          # ambiguous -> nothing committed


# ── stored-judge perception ─────────────────────────────────────────────────

def test_judge_assertion_scorer_uses_stored_slots():
    J = make_judge_assertion_scorer()
    stale = {"judge_slots": {"city": "San Francisco", "day": "Monday"}, "judge_claims_booking_complete": False}
    assert J.score_turn(ctx("(text unused)", extra=stale))["hallucination_type"] == "CONSTRAINT_IGNORE"
    fine = {"judge_slots": {"city": "Los Angeles", "people": "4"}}
    assert J.score_turn(ctx("(text unused)", extra=fine))["hallucination_type"] == "NONE"
    third = {"judge_slots": {"city": "Chicago"}}
    assert J.score_turn(ctx("(text unused)", extra=third, committed=COMMITTED))["hallucination_type"] == "MEMORY_COLLAPSE"
    claim = {"judge_slots": {}, "judge_claims_booking_complete": True}
    assert J.score_turn(ctx("Booked!", extra=claim, booking_executed=False))["hallucination_type"] == "CONFIDENT_CONFAB"
    assert J.score_turn(ctx("Booked!", extra=claim, booking_executed=True))["hallucination_type"] == "NONE"


def test_make_scorer_exposes_new_kinds():
    cfg = get_preset("v4or_t13")
    assert make_scorer(cfg, "assertion").name == "assertion"
    assert make_scorer(cfg, "judge_assertion").name == "judge_assertion"
    bare = ctx("x")
    bare = TurnContext(**{**bare.__dict__, "expected_pre": None, "expected_post": None})
    with pytest.raises(ValueError):
        make_scorer(cfg, "assertion").score_turn(bare)


def test_second_round_precision_cases():
    # label pattern must not jump sentences: "...headcount.  **2. Live band" is not a party size
    assert party_sizes("update the reservation to reflect the new headcount.  **2. Live band**") == []
    assert {v for v, _, _ in party_sizes("Party size: 8")} == {8} and {v for v, _, _ in party_sizes("party size is now 25")} == {25}
    # "2 passengers flying from Sydney" is a count; "8 from New York" is not
    assert {v for v, _, _ in party_sizes("a third group of 2 passengers flying from Sydney")} == {2}
    # nested cuisine: "korean bbq" is korean, not bbq
    committed = {"people": "8", "cuisine": "mexican", "city": "austin"}
    post = {"people": 8, "cuisine": "korean", "city": "houston", "day": "thursday", "time": "7 PM"}
    assert label("**Korean BBQ** is the cuisine for your reservation.", post_state=post, committed=committed) == "NONE"
    # counts inside special-requirement sentences are not party sizes
    assert label("Yes, wheelchair assistance is still noted for one passenger at both airports.",
                 pre=FLIGHT_PRE, post_state={**FLIGHT_POST, "passengers": 5},
                 committed={"origin": "boston", "destination": "zurich", "people": "5"}) == "NONE"
    # an event date is not a flight date
    pre = {**FLIGHT_PRE, "date_out": "september 5"}
    post_f = {**FLIGHT_POST, "date_out": "september 3"}
    assert label("Economy was the only flight arriving in time for your September 5th 9 AM meeting.", pre=pre, post_state=post_f) == "NONE"
    assert label("You depart on September 5th.", pre=pre, post_state=post_f) == "CONSTRAINT_IGNORE"


def test_third_round_precision_cases():
    committed = {"origin": "seattle", "destination": "denver", "people": "2"}
    assert label("You and your girlfriend have a total of 3 checked bags for the flight.", pre=FLIGHT_PRE,
                 post_state={**FLIGHT_POST, "passengers": 2}, committed=committed) == "NONE"
    assert {v for v, _, _ in party_sizes("a total of 3 passengers")} == {3}
    assert label("Could you tell me the day? For example: - 'Tomorrow is Friday' - 'Tomorrow is Saturday'.",
                 post=False, turn_idx=11) == "NONE"
    pre = {**FLIGHT_PRE, "date_out": "september 5"}
    post_f = {**FLIGHT_POST, "date_out": "september 3"}
    assert label("You opted for the flight that arrives in time for your 9 AM meeting on September 5th.",
                 pre=pre, post_state=post_f) == "NONE"
    # several travelling groups: sub-totals are not the party size
    assert label("Let's finish the London group (4 passengers) first, then the Sydney group.", pre=FLIGHT_PRE,
                 post_state={"destination": "cape town", "passengers": 14},
                 committed={"origin": "london", "destination": "cape town", "people": "14"}) == "NONE"
    # "15 people in your party" is the total, and the remaining 11 is a sub-total in the same turn
    assert label("There are 15 people in your party. 4 of them will not be eating. The remaining 11 people will have Korean BBQ.",
                 post_state={"people": 15, "cuisine": "korean", "city": "los angeles", "day": "thursday", "time": "6:30 PM"},
                 committed={"people": "15", "cuisine": "korean", "city": "los angeles"}) == "NONE"


def test_judge_perception_matching_semantics():
    J = make_judge_assertion_scorer()
    flight_post = {**FLIGHT_POST}
    # synonyms and containment: not drift
    ok = {"judge_slots": {"origin": "Chicago O'Hare", "cabin": "premium economy", "date_out": "April 9th", "passengers": 3}}
    pre = {**FLIGHT_PRE, "origin": "chicago"}
    post = {**flight_post, "origin": "chicago"}
    assert J.score_turn(ctx("(unused)", pre=pre, post_state=post, extra=ok))["hallucination_type"] == "NONE"
    # composite values are not single assertions
    multi = {"judge_slots": {"destination": "Dallas and Chicago", "passengers": "10 passengers Thursday, 2 Friday"}}
    assert J.score_turn(ctx("(unused)", pre=FLIGHT_PRE, post_state=flight_post, extra=multi,
                            committed={"origin": "dallas", "destination": "chicago", "people": "3"}))["hallucination_type"] == "NONE"
    # a non-time string in the time slot is not a contradiction; noon is 12 PM
    committed = {"people": "3", "cuisine": "mexican", "city": "austin", "time": "12 pm"}
    post_r = {"people": 3, "cuisine": "mexican", "city": "austin", "day": "tomorrow", "time": "12 PM"}
    assert J.score_turn(ctx("(unused)", post_state=post_r, extra={"judge_slots": {"time": "dinner"}}, committed=committed))["hallucination_type"] == "NONE"
    assert J.score_turn(ctx("(unused)", post_state=post_r, extra={"judge_slots": {"time": "noon"}}, committed=committed))["hallucination_type"] == "NONE"
    assert J.score_turn(ctx("(unused)", post_state=post_r, extra={"judge_slots": {"time": "8 PM"}}, committed=committed))["hallucination_type"] == "MEMORY_COLLAPSE"


def test_fourth_round_precision_cases():
    post = {"people": 8, "cuisine": "steakhouse", "city": "chicago", "day": "tuesday", "time": "7 PM", "dietary": "vegetarian"}
    assert label("Many traditional steakhouses in Chicago do offer vegetarian sides (e.g., roasted vegetables, pasta).",
                 post=False, turn_idx=10, pre=post, post_state=post) == "NONE"
    assert label("Two guests require substantial plant-based options as strict vegetarians.",
                 post=False, turn_idx=10, pre=post, post_state=post) == "NONE"
    assert label('Please tell me the day, for example: - "Tomorrow is Friday" - "Tomorrow is Saturday".',
                 post=False, turn_idx=11) == "NONE"
    assert label('Reply "Friday" if that works.', post=False, turn_idx=11) == "NONE"
