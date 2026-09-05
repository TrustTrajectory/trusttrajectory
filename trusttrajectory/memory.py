"""Rolling-summary memory for the model under test (the ``memory_mode="summary"`` condition).

Production agents rarely resend the whole transcript.  A common design keeps
the most recent messages verbatim and folds older ones into a running summary
that the model writes itself (LangChain's ``ConversationSummaryBufferMemory``,
and the "compaction" step of most coding agents).  Under this condition the
model under test sees its system prompt, its own latest summary and the last
``keep`` messages; whenever ``every`` further messages have piled up beyond the
kept window, the model is asked to fold them into a new summary.  The summary
prompt is deliberately generic: it asks for "everything needed to continue the
task", not for the current value of each slot.
"""
from __future__ import annotations

from typing import Dict, List

SUMMARY_SYSTEM_PROMPT = (
    "You maintain the working memory of an assistant that is helping a user with a booking. "
    "You will be given the assistant's previous summary and the conversation that happened since then. "
    "Write an updated summary that preserves everything needed to continue the task accurately. "
    "Reply with the summary text only."
)
SUMMARY_HEADER = "\n\nSummary of the conversation so far (your own notes):\n"


def summary_request(previous: str, evicted: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """The chat request that asks the model to fold ``evicted`` into ``previous``."""
    transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in evicted)
    body = (f"Previous summary:\n{previous.strip() or '(none yet)'}\n\n"
            f"Conversation since then:\n{transcript}\n\nUpdated summary:")
    return [{"role": "system", "content": SUMMARY_SYSTEM_PROMPT}, {"role": "user", "content": body}]


class RollingSummaryMemory:
    """Tracks which prefix of the transcript has been folded into the summary."""

    def __init__(self, keep: int, every: int):
        if keep < 1 or every < 1:
            raise ValueError("keep and every must be >= 1")
        self.keep, self.every = keep, every
        self.summary = ""
        self.folded = 1          # messages[1:folded] are represented by the summary (index 0 = system prompt)
        self.n_summaries = 0

    def needs_update(self, messages: List[Dict[str, str]]) -> bool:
        return len(messages) - self.folded > self.keep + self.every

    def pending(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """The messages the next fold would move into the summary."""
        return messages[self.folded: len(messages) - self.keep]

    def fold(self, messages: List[Dict[str, str]], new_summary: str) -> None:
        self.summary = (new_summary or "").strip()
        self.folded = len(messages) - self.keep
        self.n_summaries += 1

    def view(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """What the model is sent: system prompt (+ summary) and the unfolded tail."""
        if self.folded <= 1 and not self.summary:
            return list(messages)
        head = {"role": messages[0]["role"], "content": messages[0]["content"] + SUMMARY_HEADER + self.summary}
        return [head] + list(messages[self.folded:])
