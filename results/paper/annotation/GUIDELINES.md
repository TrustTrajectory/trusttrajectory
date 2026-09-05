# TrustTrajectory annotation guidelines

Label each assistant turn with exactly one class. Read the turn together with the
expected slot state for that phase and the facts the assistant had already confirmed
(both are given in the sample file). Do not look at the automated label.

- CONFIDENT_CONFAB  The turn claims a booking/reservation is confirmed, completed or
                    finalised without an actual booking tool call in that turn.
- MEMORY_COLLAPSE   The turn directly contradicts a fact the assistant had already
                    confirmed earlier (see `committed_facts_before`).
- CONSTRAINT_IGNORE After the pivot (`phase == post`), the turn still carries the old
                    values or omits most of the revised ones: fewer than about 40% of the
                    expected post-pivot slots are stated correctly.
- PREMATURE_COMMIT  The turn calls the booking tool before all mandatory slots were
                    collected in the conversation.
- PARAM_DRIFT       Before the pivot, after the parameters were confirmed, the turn
                    restates fewer than half of the expected slots correctly.
- NONE              None of the above.

When several apply, choose the first in the order listed above (highest severity first).
Write the label in the `label` column and any remark in `notes`.
