"""
groq_menu_advisor.py

Uses Groq's API to pick which fight-menu option (FIGHT/ACT/ITEM/SPARE/etc.)
to choose, given the labels currently visible on screen. This is the
"knowledge" half of hybrid_play.py: the trained RL model has no semantic
understanding of what these verbs actually do (it only learned "confirming
something here is rewarded"), whereas an LLM at least has some general
knowledge of how these RPG-style menus typically work.

PLAYSTYLE: tuned to win fights efficiently - no attempt to track a
pacifist/neutral/genocide route. In practice this means it should default
to FIGHT when available (the direct, reliable way to deal damage), and only
prefer something else if a visible label is obviously a stronger option
(e.g. a boosted attack). It has no access to HP, enemy identity, or turn
history, so treat its picks as a reasonable heuristic, not optimal play.

SETUP
-----
    pip install groq
    export GROQ_API_KEY="your-key-here"

Silently disables itself (falls back to picking FIGHT if present, else the
first available label) if the key isn't set or `groq` isn't installed -
hybrid_play.py still runs without Groq, just without the "knowledge" layer.
"""

import os
from typing import Optional


class MenuAdvisor:
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        self.model = model
        self.enabled = False
        self.client = None

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("[MenuAdvisor] GROQ_API_KEY not set - falling back to a simple "
                  "'prefer FIGHT' heuristic instead of LLM-advised picks.")
            return

        try:
            from groq import Groq
        except ImportError:
            print("[MenuAdvisor] `groq` package not installed - falling back to a "
                  "simple 'prefer FIGHT' heuristic. Install with: pip install groq")
            return

        self.client = Groq(api_key=api_key)
        self.enabled = True

    def _fallback_choice(self, labels: list[str]) -> str:
        for label in labels:
            if "FIGHT" in label.upper():
                return label
        return labels[0]

    def choose_action(self, labels: list[str]) -> str:
        """
        labels: the OCR'd, non-empty label texts currently visible (already
        filtered of None entries by the caller). Returns one of the given
        labels verbatim - never invents an option that wasn't in the list.
        """
        if not labels:
            raise ValueError("choose_action called with no visible labels")

        if not self.enabled:
            return self._fallback_choice(labels)

        options_str = ", ".join(labels)
        prompt = (
            "You are playing a turn in Deltarune's battle system. The goal is "
            "simply to win this fight as efficiently as possible - do not try to "
            "spare enemies or worry about a pacifist/genocide route, just pick "
            "whatever action best helps win quickly.\n\n"
            f"Available actions this turn: {options_str}\n\n"
            "Reply with ONLY the exact text of one option from that list, nothing else."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.3,
            )
            raw_choice = response.choices[0].message.content.strip().upper()
        except Exception as exc:
            print(f"[MenuAdvisor] Groq API call failed, using fallback: {exc}")
            return self._fallback_choice(labels)

        # Match the model's reply back to one of the actual visible labels
        # (case-insensitive, substring-tolerant) - never trust it blindly as
        # an exact match, since OCR text and LLM output can both be noisy.
        for label in labels:
            if label.upper() in raw_choice or raw_choice in label.upper():
                return label

        print(f"[MenuAdvisor] Couldn't match Groq's reply ('{raw_choice}') to a "
              f"visible option, using fallback.")
        return self._fallback_choice(labels)
