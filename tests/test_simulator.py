from trusttrajectory.config import PRESETS
from trusttrajectory.scenarios import load_scenarios
from trusttrajectory.simulator import UserSimulator


def scenario(id_="easy_01"):
    return load_scenarios(ids=[id_])[0]


def test_canonical_drip_answers_each_pattern_once_then_falls_back():
    sim = UserSimulator(scenario(), PRESETS["v4or_t13"], pivot_turn=13)
    assert sim.drip_reply("How many people?", False, False) == "just 2 of us"
    assert sim.drip_reply("How many people again?", False, False) == "Please continue."  # already used
    assert sim.drip_reply("What cuisine do you like?", False, False) == "Italian"
    assert not sim.drip_exhausted()
    assert sim.drip_reply("anything", True, False) is None  # silent after the pivot


def test_pivot_message_carries_the_instruction_suffix_only_when_configured():
    s = scenario()
    assert UserSimulator(s, PRESETS["v4or_t13"], 13).pivot_message().endswith("with the revised details.")
    assert UserSimulator(s, PRESETS["v4or_t15"], 15).pivot_message() == s["pivot"]


def test_question_only_and_repeatable_drip_for_the_t15_preset():
    sim = UserSimulator(scenario(), PRESETS["v4or_t15"], 15)
    assert sim.drip_reply("I have noted your party size.", False, False) is None  # no question mark
    assert sim.drip_reply("How many people?", False, False) == "just 2 of us"
    assert sim.drip_reply("How many people?", False, False) == "just 2 of us"  # repeatable
    # a drip pattern always wins over the complication reply ...
    assert sim.drip_reply("Is the vegetarian requirement strict?", False, True) == "no restrictions"
    # ... but a complication-topic question that no drip pattern covers gets the fixed reply
    assert sim.drip_reply("Should I note the wheelchair access?", False, True) == \
        "Please note the requirement and proceed with the best available option."
    assert sim.drip_reply("Shall I proceed?", False, True) is None


def test_conflict_fallback_for_the_nudge_preset():
    sim = UserSimulator(scenario("conflict_01"), PRESETS["v4or_t13_nudge"], 13)
    assert sim.drip_reply("Which requirement takes priority?", False, True).startswith("Please flag the conflict")
    assert sim.drip_reply("Anything else?", False, False) == "Please go ahead and proceed."


def test_post_pivot_nudge_uses_expected_post_state():
    sim = UserSimulator(scenario(), PRESETS["v4or_t13_nudge"], 13)
    assert sim.post_pivot_nudge() == "All details confirmed: day is saturday, time is 7 PM, party size is 3. Please book now."
    flight = UserSimulator(scenario("flight_medium_04"), PRESETS["v4or_t13_nudge"], 13)
    assert flight.post_pivot_nudge().startswith("All details confirmed. Output the booking JSON")
