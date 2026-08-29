"""
groq_teacher.py

An episode-end "teacher" that uses Groq's API to review how each training
episode went and give a short, specific coaching tip - printed to the
console so you can read it as training progresses.

IMPORTANT SCOPE NOTE: this is advisory, not a training signal. It doesn't
modify the numeric rewards PPO already used to update the model for that
episode - by the time an episode ends, PPO has already seen and learned
from those transitions. This is a coach watching from the sidelines and
telling you what it noticed, not something that edits the RL math.

WHY NOT PER-STEP FEEDBACK: the environment runs in real time at roughly
TARGET_FPS decisions/second (see deltarune_env.py). An API call - even a
fast one on Groq - takes hundreds of milliseconds at best, which would
wreck the frame timing the environment relies on. Reviewing once per
episode (which naturally takes several seconds to complete) fits without
disrupting anything.

SETUP
-----
    pip install groq
    export GROQ_API_KEY="your-key-here"

If the key isn't set or the `groq` package isn't installed, this silently
disables itself - training still works fine, you just won't get coaching
output. Never hardcode your API key into this file; always use the
environment variable.
"""

import os
from typing import Optional


class EpisodeTeacher:
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        self.model = model
        self.enabled = False
        self.client = None

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("[EpisodeTeacher] GROQ_API_KEY not set - coaching disabled. "
                  "Set it with: export GROQ_API_KEY=\"your-key-here\"")
            return

        try:
            from groq import Groq
        except ImportError:
            print("[EpisodeTeacher] `groq` package not installed - coaching disabled. "
                  "Install with: pip install groq")
            return

        self.client = Groq(api_key=api_key)
        self.enabled = True

    def review_episode(self, stats: dict) -> Optional[str]:
        """
        stats should contain:
          total_reward: float
          turn_count: int
          steps: int
          died: bool  - True if the episode ended via a SOUL break, False
                        if it ended via truncation (ran out of steps/timeout)
        Returns a short coaching tip string, or None if the teacher is
        disabled or the API call fails.
        """
        if not self.enabled:
            return None

        outcome = "died (SOUL broke)" if stats.get("died") else "timed out / truncated"
        prompt = (
            "You are coaching a reinforcement learning agent learning to play "
            "Deltarune's bullet-hell dodge fights via screen-capture and simulated "
            "key presses. Here's how its last episode went:\n\n"
            f"- Total reward: {stats.get('total_reward', 0):.2f}\n"
            f"- Turns completed (menu actions confirmed): {stats.get('turn_count', 0)}\n"
            f"- Episode length: {stats.get('steps', 0)} steps\n"
            f"- Outcome: {outcome}\n\n"
            "Give ONE short, specific, actionable coaching tip (1-2 sentences) for "
            "what to try differently next episode. Be concrete, not generic "
            "encouragement - the agent can't act on 'try harder'."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"[EpisodeTeacher] Groq API call failed, skipping this episode's review: {exc}")
            return None
