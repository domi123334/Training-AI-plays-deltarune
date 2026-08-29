"""
groq_dodge_advisor.py

Occasionally asks Groq which direction to dodge, given the SOUL's position
relative to a detected threat. This is NOT meant to replace the fast
geometric heuristic in heuristic_dodge.py - an API call takes hundreds of
milliseconds at best, which is far too slow to be the primary dodge
decision-maker in a real-time ~30-decisions/sec loop. See
heuristic_llm_agent.py for how this is used at a low, configurable
probability alongside (not instead of) the heuristic.

SETUP
-----
    pip install groq
    export GROQ_API_KEY="your-key-here"

Silently disables itself if the key isn't set or `groq` isn't installed -
callers should always have a heuristic fallback ready regardless.
"""

import os
from typing import Optional


class DodgeAdvisor:
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        self.model = model
        self.enabled = False
        self.client = None

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return  # caller falls back to the heuristic - no need to print here,
                     # heuristic_llm_agent.py only consults this rarely anyway

        try:
            from groq import Groq
        except ImportError:
            return

        self.client = Groq(api_key=api_key)
        self.enabled = True

    def suggest_direction(self, dx: float, dy: float) -> Optional[str]:
        """
        dx, dy: the threat's position relative to the SOUL (threat_x - soul_x,
        threat_y - soul_y). Returns "UP"/"DOWN"/"LEFT"/"RIGHT", or None if
        disabled or the call fails/returns something unparseable - callers
        must fall back to the heuristic in that case.
        """
        if not self.enabled:
            return None

        prompt = (
            "A game character needs to dodge away from a threat. The threat's "
            f"position relative to the character is ({dx:.0f}, {dy:.0f}) in pixels, "
            "where positive x is to the right and positive y is downward. "
            "Which single direction should the character move to get away from "
            "it? Reply with exactly one word: UP, DOWN, LEFT, or RIGHT."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.3,
            )
            raw = response.choices[0].message.content.strip().upper()
        except Exception:
            return None

        for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
            if direction in raw:
                return direction
        return None
