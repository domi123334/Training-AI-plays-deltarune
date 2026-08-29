"""
llm_vision_agent.py

Plays Deltarune (any chapter, 1-5) by having an LLM (Claude, vision-enabled)
look at screenshots and choose a movement action, instead of training an RL
policy. Reuses your existing capture/key-injection plumbing:

    - CAPTURE_REGION / find_window_region()   from deltarune_env.py
    - key_down() / key_up()                   from linux_input.py

HOW IT WORKS
------------
Loop, forever (or until Ctrl+C):
  1. Screenshot the game window region.
  2. Send the image + a short prompt to Claude, asking for ONE action:
     forward/backward/left/right/none (mapped to w/s/a/d).
  3. Press that key, hold it briefly, release it.
  4. Repeat.

This is intentionally simple and *not* a from-scratch RL agent - the LLM is
just reacting frame-by-frame with no memory of past frames unless you add it
(see HISTORY_LEN below for a cheap way to give it some).

BEFORE YOU RUN THIS
--------------------
1. `pip install anthropic mss opencv-python pynput` (numpy comes with cv2).
2. Set your API key:  export ANTHROPIC_API_KEY=sk-ant-...
3. Calibrate CAPTURE_REGION below (or call find_window_region("DELTARUNE")
   if you're on Linux/X11 with xdotool installed - see deltarune_env.py).
4. Keep the game window focused, unobstructed, and not overlapped -
   this is screen-watching + synthetic key presses, so alt-tabbing breaks it.
5. Run:  python llm_vision_agent.py
   You'll get a countdown to switch focus to the game window before it
   starts sending key presses.

LATENCY NOTE
------------
Round-tripping a screenshot to an LLM and back takes real time (hundreds of
ms to a couple seconds depending on model/load). This loop is NOT fast
enough for frame-perfect bullet-hell dodging - it's better suited to overworld
navigation, slower attack patterns, or chapters/sections with more reaction
time. If you need faster reflexes, consider a hybrid: a cheap pixel-based
"emergency dodge" (like the flash-detection heuristic in deltarune_env.py)
running every frame locally, with the LLM only called periodically for
higher-level navigation/strategy.
"""

import os
import sys
import time
import base64
import json
import collections
from typing import Optional

import cv2
import mss
import numpy as np
from anthropic import Anthropic

from linux_input import key_down, key_up

# ---------------------------------------------------------------------------
# CONFIG - calibrate these to your setup
# ---------------------------------------------------------------------------

# Pixel region of your screen where the Deltarune window renders.
# (left, top, width, height) - update to match your monitor/window, or use
# find_window_region("DELTARUNE") from deltarune_env.py if on Linux/X11.
CAPTURE_REGION = {"left": 100, "top": 100, "width": 640, "height": 480}

# Model to use for vision decisions.
MODEL = "claude-sonnet-5"

# How often to ask the LLM for a new decision, in seconds. Lower = more
# responsive but more API calls/cost and more latency-induced lag.
DECISION_INTERVAL = 0.6

# How long to hold the chosen movement key per decision, in seconds.
# Usually close to DECISION_INTERVAL so movement is roughly continuous.
KEY_HOLD_SECONDS = 0.5

# How many past (thumbnail) frames + decisions to include as context, so the
# LLM has some sense of recent motion/trend instead of judging one still
# frame in isolation. 0 disables this (fastest/cheapest, least context).
HISTORY_LEN = 3

# Default movement keys (matches deltarune_env.py's convention).
KEY_FORWARD = "w"   # up
KEY_BACK = "s"      # down
KEY_LEFT = "a"
KEY_RIGHT = "d"

ACTION_TO_KEY = {
    "forward": KEY_FORWARD,
    "backward": KEY_BACK,
    "left": KEY_LEFT,
    "right": KEY_RIGHT,
    "none": None,
}

SYSTEM_PROMPT = """You are playing Deltarune, a top-down RPG with real-time \
bullet-hell dodging segments during battles. You will be shown a screenshot \
of the current game state. Your only job is to choose ONE movement action \
for the player-controlled SOUL (heart icon) or overworld character:

- "forward": move up
- "backward": move down
- "left": move left
- "right": move right
- "none": stay still (e.g. mid-dialogue, menu you don't want to navigate \
with movement, or the safest option right now)

If you're in a bullet-hell/dodging segment, prioritize moving the SOUL away \
from incoming projectiles/attacks and toward open space. If you're walking \
around the overworld, move toward the apparent objective (an NPC, a door, \
a visible path). If a text box or menu is on screen, prefer "none" unless \
movement is clearly the way to advance/select something.

Respond with ONLY a compact JSON object, no other text, no markdown fences:
{"action": "<forward|backward|left|right|none>", "reason": "<5 words or fewer>"}
"""


class LLMVisionAgent:
    def __init__(self, capture_region: dict = CAPTURE_REGION):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: set the ANTHROPIC_API_KEY environment variable.")
            sys.exit(1)

        self.client = Anthropic(api_key=api_key)
        self.capture_region = capture_region
        self.sct = mss.mss()
        self.history = collections.deque(maxlen=HISTORY_LEN)
        self._held_key: Optional[str] = None

    # -- capture ---------------------------------------------------------

    def _grab_frame_png_b64(self) -> str:
        """Capture the region and return it as a base64-encoded PNG string."""
        raw = np.array(self.sct.grab(self.capture_region))  # BGRA
        bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            raise RuntimeError("Failed to encode screenshot as PNG")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    # -- decision ----------------------------------------------------------

    def _build_messages(self, frame_b64: str) -> list:
        content = []

        # Include a few recent frames + the actions taken, for light context.
        for past_frame_b64, past_action in self.history:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": past_frame_b64,
                    },
                }
            )
            content.append(
                {"type": "text", "text": f"(previous frame - action taken: {past_action})"}
            )

        content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": frame_b64},
            }
        )
        content.append({"type": "text", "text": "(current frame - choose your action now)"})

        return [{"role": "user", "content": content}]

    def decide_action(self, frame_b64: str) -> str:
        messages = self._build_messages(frame_b64)
        try:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=100,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
        except Exception as e:
            print(f"[warn] API call failed ({e}); defaulting to 'none'")
            return "none"

        text = "".join(block.text for block in response.content if block.type == "text").strip()
        text = text.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(text)
            action = parsed.get("action", "none")
            reason = parsed.get("reason", "")
        except (json.JSONDecodeError, AttributeError):
            print(f"[warn] couldn't parse model output: {text!r}; defaulting to 'none'")
            return "none"

        if action not in ACTION_TO_KEY:
            print(f"[warn] unknown action {action!r}; defaulting to 'none'")
            action = "none"

        print(f"[decision] {action}" + (f" - {reason}" if reason else ""))
        return action

    # -- control -----------------------------------------------------------

    def _apply_action(self, action: str):
        key = ACTION_TO_KEY[action]

        if self._held_key is not None and self._held_key != key:
            key_up(self._held_key)
            self._held_key = None

        if key is not None:
            key_down(key)
            self._held_key = key

        time.sleep(KEY_HOLD_SECONDS)

        if self._held_key is not None:
            key_up(self._held_key)
            self._held_key = None

    def release_all_keys(self):
        if self._held_key is not None:
            key_up(self._held_key)
            self._held_key = None

    # -- main loop -----------------------------------------------------------

    def run(self, max_steps: Optional[int] = None):
        step = 0
        try:
            while max_steps is None or step < max_steps:
                loop_start = time.time()

                frame_b64 = self._grab_frame_png_b64()
                action = self.decide_action(frame_b64)

                if HISTORY_LEN > 0:
                    self.history.append((frame_b64, action))

                self._apply_action(action)

                elapsed = time.time() - loop_start
                remaining = DECISION_INTERVAL - elapsed
                if remaining > 0:
                    time.sleep(remaining)

                step += 1
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            self.release_all_keys()
            self.sct.close()


def countdown(seconds: int = 5):
    print(f"Starting in {seconds} seconds - switch to the Deltarune window now.")
    for i in range(seconds, 0, -1):
        print(f"  {i}...")
        time.sleep(1)


if __name__ == "__main__":
    countdown(5)
    agent = LLMVisionAgent()
    agent.run()
