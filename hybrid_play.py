"""
hybrid_play.py

Plays live using two different decision-makers for two different jobs:

- DODGING (bullet-hell phases): the trained RL model, deterministic
  (model.predict(obs, deterministic=True)) - this is the actual learned
  skill, reflexive pixel-to-movement mapping that RL is well-suited for.
- MENU DECISIONS (FIGHT/ACT/ITEM/SPARE/etc.): a small state machine advised
  by Groq (see groq_menu_advisor.py) - the RL model has no semantic
  understanding of what these verbs do, it only ever learned "confirming
  something here is rewarded." An LLM at least has some general knowledge
  of how these menus work. Playstyle is "win efficiently," not pacifist/
  genocide-route aware - see groq_menu_advisor.py for what that means in
  practice.

HOW THE MENU HANDOFF WORKS
---------------------------
1. Every loop iteration, capture a frame and check whether the SOUL is at
   the menu row (same detection deltarune_env.py's RewardEstimator uses for
   turn-progress reward).
2. The instant the menu is detected, OCR every visible verb label
   (MenuActionReader.read_all_visible_labels), ask MenuAdvisor which one to
   pick, and note its x-position as a target.
3. Take manual control: move the SOUL left/right until it's aligned with
   the target x-position (within CURSOR_ALIGN_TOLERANCE), then tap confirm.
   This bypasses the RL model's action for these frames entirely.
4. Once confirmed, hand control back to the RL model for the next dodge phase.

LIMITATIONS - read before relying on this
-------------------------------------------
- OCR on small pixel-art fonts is not perfectly reliable. A misread label
  can lead to a wrong pick or a failed match (falls back to "prefer FIGHT").
- There's no HP or enemy-identity reading, so "win efficiently" is a
  heuristic based only on the visible option text, not actual battle state.
- If MENU_SOUL_Y / MENU_VERB_X_CENTERS in deltarune_env.py aren't calibrated
  to your battle UI, menu detection and cursor alignment won't work right -
  calibrate those first (see deltarune_env.py's docstring).

USAGE
-----
    python hybrid_play.py                          # uses deltarune_ppo_final.zip
    python hybrid_play.py checkpoints/some_model.zip
"""

import sys
import time
from typing import Optional

from stable_baselines3 import PPO

from deltarune_env import (
    DeltaruneEnv,
    MenuActionReader,
    find_soul,
    MENU_SOUL_Y,
    MENU_Y_TOLERANCE,
    MENU_VERB_X_CENTERS,
)
from groq_menu_advisor import MenuAdvisor

COUNTDOWN_SECONDS = 5
DEFAULT_MODEL_PATH = "deltarune_ppo_final"
NUM_EPISODES = 5

# How close (in pixels) the SOUL needs to be to the target verb's x-center
# before we consider the cursor "aligned" and tap confirm.
CURSOR_ALIGN_TOLERANCE = 8

# Movement/button indices, matching deltarune_env.py's MOVE_KEYS/BUTTON_KEYS
# ordering: MOVE_KEYS = [None, UP, DOWN, LEFT, RIGHT], BUTTON_KEYS = [None, ENTER, CANCEL, SKIP]
MOVE_NONE, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT = range(5)
BUTTON_NONE, BUTTON_ENTER = 0, 1


def countdown(seconds: int):
    print(f"Starting in {seconds} seconds - switch to the Deltarune window now.")
    for i in range(seconds, 0, -1):
        print(f"  {i}...")
        time.sleep(1)


class HybridController:
    def __init__(self, model_path: str):
        print(f"Loading model from: {model_path}")
        self.model = PPO.load(model_path)
        self.env = DeltaruneEnv(max_episode_steps=1200)
        self.action_reader = MenuActionReader()
        self.advisor = MenuAdvisor()

        # Menu state-machine state
        self.in_menu_control = False
        self.target_x: Optional[int] = None
        self.target_label: Optional[str] = None

    def _detect_menu(self, frame_bgr) -> Optional[tuple[int, int]]:
        """Returns the SOUL's (x, y) if it's an intact SOUL at the menu row, else None."""
        fragment_count, soul_pos = find_soul(frame_bgr)
        if fragment_count != 1 or soul_pos is None:
            return None
        if abs(soul_pos[1] - MENU_SOUL_Y) > MENU_Y_TOLERANCE:
            return None
        return soul_pos

    def _begin_menu_control(self, frame_bgr):
        raw_labels = self.action_reader.read_all_visible_labels(frame_bgr)
        visible = [(label, x) for label, x in zip(raw_labels, MENU_VERB_X_CENTERS) if label]

        if not visible:
            # Couldn't read anything - don't get stuck, just let the RL
            # model keep acting normally until the menu either resolves
            # itself or OCR catches something on a later frame.
            return

        labels_only = [label for label, _ in visible]
        chosen_label = self.advisor.choose_action(labels_only)
        chosen_x = next(x for label, x in visible if label == chosen_label)

        print(f"[Hybrid] Menu detected. Options: {labels_only} -> choosing: {chosen_label}")
        self.in_menu_control = True
        self.target_x = chosen_x
        self.target_label = chosen_label

    def _menu_control_action(self, soul_x: int) -> tuple[int, int]:
        """Returns (move_idx, button_idx) to align with and confirm the target."""
        delta = self.target_x - soul_x
        if abs(delta) <= CURSOR_ALIGN_TOLERANCE:
            print(f"[Hybrid] Aligned on {self.target_label}, confirming.")
            self.in_menu_control = False
            self.target_x = None
            self.target_label = None
            return MOVE_NONE, BUTTON_ENTER
        move_idx = MOVE_RIGHT if delta > 0 else MOVE_LEFT
        return move_idx, BUTTON_NONE

    def run(self, num_episodes: int = NUM_EPISODES):
        try:
            for episode in range(1, num_episodes + 1):
                obs, _ = self.env.reset()
                self.in_menu_control = False
                self.target_x = None
                done = truncated = False
                episode_reward = 0.0

                while not (done or truncated):
                    # Check menu state using the same frame the env is about
                    # to capture internally - a small amount of redundant
                    # capture is the simplest way to keep this decoupled
                    # from DeltaruneEnv's internals.
                    frame = self.env._grab_frame()

                    if not self.in_menu_control:
                        menu_soul_pos = self._detect_menu(frame)
                        if menu_soul_pos is not None:
                            self._begin_menu_control(frame)

                    if self.in_menu_control:
                        fragment_count, soul_pos = find_soul(frame)
                        if fragment_count == 1 and soul_pos is not None:
                            action = self._menu_control_action(soul_pos[0])
                        else:
                            action = (MOVE_NONE, BUTTON_NONE)  # lost track - wait
                    else:
                        action, _ = self.model.predict(obs, deterministic=True)

                    obs, reward, done, truncated, info = self.env.step(action)
                    episode_reward += reward

                print(
                    f"Episode {episode}/{num_episodes} finished - "
                    f"reward: {episode_reward:.2f}, turns completed: {info.get('turn_count', 0)}"
                )
        except KeyboardInterrupt:
            print("Stopped.")
        finally:
            self.env.close()


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
    controller = HybridController(model_path)
    countdown(COUNTDOWN_SECONDS)
    controller.run()


if __name__ == "__main__":
    main()
