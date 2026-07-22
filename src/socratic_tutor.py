"""
socratic_tutor.py
------------------
The core Socratic reasoning engine. Wraps the Claude API with:
  - retrieved context grounding (RAG)
  - per-question hint-count tracking
  - a "leak checker" pass that stops the model from accidentally
    blurting out the direct answer
  - a reveal mechanism after N unsuccessful hints
"""

import os
import json
from dataclasses import dataclass, field
from anthropic import Anthropic
from .retriever import Retriever

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
MAX_HINTS = int(os.getenv("MAX_HINTS_BEFORE_REVEAL", "4"))

SOCRATIC_SYSTEM_PROMPT = """You are a patient, encouraging Socratic tutor.

You will be given:
1. Context retrieved from the course material (ground truth - never contradict it,
   never invent facts not supported by it).
2. The conversation so far between you and the student.

Your job is NOT to give the direct answer. Instead:
- Ask exactly ONE guiding question that helps the student take the next step
  themselves.
- Base every hint strictly on the retrieved context.
- If the student's latest message is fully correct, say so clearly, briefly
  explain why using the retrieved context, and cite which part of the material
  supports it.
- If the student is partially correct, acknowledge the correct part specifically,
  then ask a follow-up question targeting only the part they're missing.
- If the student seems completely lost, make your next question simpler and more
  concrete (e.g. a smaller example) rather than repeating the same question.
- Never give away the final formula, numeric answer, or conclusion directly -
  guide them to state it themselves.
- Keep responses short: 2-4 sentences plus one question, never a lecture.

Respond ONLY with a JSON object, no markdown fences, no preamble:
{
  "response": "<your reply to the student>",
  "student_status": "correct" | "partial" | "stuck" | "unclear",
  "is_final_answer_revealed": true | false
}
"""

LEAK_CHECK_PROMPT = """You are a strict reviewer. Given the retrieved ground-truth
context and a proposed tutor response, determine if the tutor response directly
states the final answer, formula, or conclusion instead of guiding the student
toward it themselves.

Respond ONLY with JSON: {"leaks_answer": true | false, "reason": "<short reason>"}

Context:
{context}

Proposed tutor response:
{response}
"""


@dataclass
class TutorSession:
    """Holds the running state for one student question/topic."""
    history: list[dict] = field(default_factory=list)  # [{role, content}]
    hint_count: int = 0
    resolved: bool = False


class SocraticTutor:
    def __init__(self, api_key: str | None = None):
        self.client = Anthropic(api_key=api_key) if api_key else Anthropic()
        self.retriever = Retriever()

    def _call_claude(self, system: str, user_content: str) -> dict:
        resp = self.client.messages.create(
            model=MODEL,
            max_tokens=500,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )
        raw = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fall back gracefully if the model doesn't return clean JSON
            return {"response": raw, "student_status": "unclear", "is_final_answer_revealed": False}

    def _check_leak(self, context: str, response: str) -> bool:
        prompt = LEAK_CHECK_PROMPT.format(context=context, response=response)
        result = self._call_claude(system="You are a strict JSON-only reviewer.", user_content=prompt)
        return bool(result.get("leaks_answer", False))

    def ask(self, session: TutorSession, student_message: str) -> dict:
        """
        Processes one turn of the conversation. Returns a dict with:
          - reply: the text to show the student
          - sources: list of source/title strings used for grounding
          - hint_count, resolved: updated session metadata
        """
        retrieved = self.retriever.retrieve(student_message, k=3)
        context_text = "\n\n---\n\n".join(
            f"[{c['source']} - {c['title']}]\n{c['text']}" for c in retrieved
        ) or "(No matching course material found for this question.)"

        session.history.append({"role": "student", "content": student_message})

        history_text = "\n".join(
            f"{turn['role'].upper()}: {turn['content']}" for turn in session.history
        )

        force_reveal = session.hint_count >= MAX_HINTS

        user_content = f"""RETRIEVED CONTEXT:
{context_text}

CONVERSATION SO FAR:
{history_text}

Hints given so far: {session.hint_count} (max before reveal: {MAX_HINTS})
{"The student has struggled long enough - you must now clearly explain the answer, citing the context, rather than asking another question." if force_reveal else ""}
"""

        result = self._call_claude(SOCRATIC_SYSTEM_PROMPT, user_content)
        reply = result.get("response", "").strip()
        status = result.get("student_status", "unclear")
        revealed = bool(result.get("is_final_answer_revealed", False)) or force_reveal

        # Leak check: only bother if the model claims it did NOT reveal the answer yet
        if not revealed and retrieved:
            if self._check_leak(context_text, reply):
                # Ask the model to try again, more conservatively
                retry_content = user_content + "\n\nYour previous draft revealed the answer directly. Rewrite it as a guiding question instead, without stating the final result."
                result = self._call_claude(SOCRATIC_SYSTEM_PROMPT, retry_content)
                reply = result.get("response", "").strip()
                status = result.get("student_status", status)
                revealed = bool(result.get("is_final_answer_revealed", False))

        session.history.append({"role": "tutor", "content": reply})

        if status == "correct" or revealed:
            session.resolved = True
        else:
            session.hint_count += 1

        return {
            "reply": reply,
            "sources": [f"{c['source']} — {c['title']}" for c in retrieved],
            "hint_count": session.hint_count,
            "resolved": session.resolved,
            "status": status,
        }
