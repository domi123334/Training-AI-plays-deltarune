"""
heuristic_llm_agent.py

A zero-training "model" that acts as if it were heuristic_dodge.py, but
with its menu decisions handed to Groq's LLM, and a small, deliberately
low-probability chance of the LLM also weighing in on dodge direction.
Exposes the same predict(obs, deterministic=...) -> (action, state)
interface as an SB3 PPO model, so it's a drop-in swap for the trained
model in play.py / hybrid_play.py / a competition harness.

WHY THIS EXISTS: for a competition needing two comparable agents - one
actually trained (see train.py / train_jevil.py) and one that requires no
training time at all but still behaves sensibly from the moment it's
created.

DODGING: primarily the exact same white-pixel-avoidance geometric logic as
heuristic_dodge.py (reused via composition, not duplicated) - this is the
fast, reliable, safety-critical path and runs on essentially every
decision. At LLM_DODGE_INFLUENCE_PROBABILITY (default 10%), the decision is
instead handed to Groq (groq_dodge_advisor.py) for that one step. This
stays low on purpose: an API call takes hundreds of milliseconds, far too
slow to be the primary decision-maker in a real-time ~30-decisions/sec
loop - if the LLM call fails or Groq isn't configured, it falls straight
back to the heuristic rather than stalling.

MENU DECISIONS: handled by Groq every time (see groq_menu_advisor.py) -
menu decisions aren't time-critical the way dodging is (the SOUL isn't
about to get hit while you're choosing a verb), so there's no need to
restrict the LLM's involvement there the way dodging requires. EXCEPTION:
fixed party-management rules (priority_actions.py) override the advisor
when triggered - if a party member's HP number is showing yellow
(critical) and ITEM is available, go heal with the Dark Burger instead of
whatever the advisor would have picked; if TP is above 82%, prefer Ralsei's
Heal Prayer (party low on HP) or Susie's Rude Buster (otherwise) for a big
tactical swing. These rules also drive a second menu-navigation stage to
find and confirm the specific submenu option (see SUBMENU_X/
SUBMENU_Y_POSITIONS in deltarune_env.py - needs calibration).

IMPORTANT: this agent does its OWN screen capture inside predict() rather
than using the `obs` argument passed in - the Gym env's obs is a grayscale
downsampled frame stack (color is discarded), but the dodge/SOUL detection
this reuses from heuristic_dodge.py needs full-resolution color to work.
`obs` and `deterministic` are still accepted so the call signature matches
PPO's, but `obs` is otherwise unused. This agent never presses keys itself -
it only returns actions; DeltaruneEnv.step() is what actually injects them.

USAGE
-----
    from heuristic_llm_agent import HeuristicLLMAgent
    model = HeuristicLLMAgent()          # no .learn() call - ready immediately
    action, _ = model.predict(obs, deterministic=True)
"""

from typing import Optional

import random

import numpy as np

from heuristic_dodge import HeuristicDodger
from deltarune_env import (
    find_soul,
    MenuActionReader,
    MENU_SOUL_Y,
    MENU_Y_TOLERANCE,
    MENU_VERB_X_CENTERS,
    detect_low_hp_characters,
    read_tp_fraction,
    SUBMENU_X,
    SUBMENU_Y_POSITIONS,
)
from groq_menu_advisor import MenuAdvisor
from groq_dodge_advisor import DodgeAdvisor
from priority_actions import PriorityDecisionMaker

CURSOR_ALIGN_TOLERANCE = 8

# How often the LLM gets a say in a dodge decision, instead of the fast
# geometric heuristic. Kept LOW and intentional: an API call takes hundreds
# of milliseconds, far too slow to be the primary decision-maker in a
# real-time ~30-decisions/sec loop. The heuristic (instant, reliable) makes
# the vast majority of dodge calls; the LLM only occasionally weighs in.
LLM_DODGE_INFLUENCE_PROBABILITY = 0.10

# Matches deltarune_env.py's MOVE_KEYS / BUTTON_KEYS ordering:
# MOVE_KEYS = [None, UP, DOWN, LEFT, RIGHT], BUTTON_KEYS = [None, ENTER, CANCEL, SKIP]
MOVE_NONE, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT = range(5)
BUTTON_NONE, BUTTON_ENTER = 0, 1


class HeuristicLLMAgent:
    def __init__(self):
        # Composition, not duplication: reuse HeuristicDodger's detection
        # logic (_find_soul, _find_nearby_white_centroid, _choose_dodge_key)
        # without ever calling .run() or its key-pressing methods - we only
        # want its decisions, the env applies the actual key presses.
        self._dodger = HeuristicDodger()
        self._action_reader = MenuActionReader()
        self._advisor = MenuAdvisor()
        self._dodge_advisor = DodgeAdvisor()
        self._priority = PriorityDecisionMaker()

        # Top-level menu state-machine state (same approach as hybrid_play.py)
        self._in_menu_control = False
        self._target_x: Optional[int] = None
        self._target_label: Optional[str] = None

        # Submenu state - only entered when a PriorityDecisionMaker rule
        # forced the top-level pick (Dark Burger / Heal Prayer / Rude
        # Buster). Normal advisor picks don't have a specific known submenu
        # target, so they stop after the top-level confirm.
        self._pending_submenu_target_text: Optional[str] = None
        self._in_submenu_control = False
        self._submenu_target_y: Optional[int] = None
        self._submenu_target_label: Optional[str] = None

        from heuristic_dodge import KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT
        self._key_to_move_idx = {
            KEY_UP: MOVE_UP,
            KEY_DOWN: MOVE_DOWN,
            KEY_LEFT: MOVE_LEFT,
            KEY_RIGHT: MOVE_RIGHT,
        }

    def _detect_menu(self, frame_bgr) -> Optional[tuple[int, int]]:
        fragment_count, soul_pos = find_soul(frame_bgr)
        if fragment_count != 1 or soul_pos is None:
            return None
        if abs(soul_pos[1] - MENU_SOUL_Y) > MENU_Y_TOLERANCE:
            return None
        return soul_pos

    def _begin_menu_control(self, frame_bgr):
        raw_labels = self._action_reader.read_all_visible_labels(frame_bgr)
        visible = [(label, x) for label, x in zip(raw_labels, MENU_VERB_X_CENTERS) if label]
        if not visible:
            return  # couldn't read anything - stay in dodge mode this frame

        labels_only = [label for label, _ in visible]

        # Priority rules (low HP -> heal, high TP -> Heal Prayer/Rude Buster)
        # take precedence over the normal advisor pick. See priority_actions.py.
        low_hp_chars = detect_low_hp_characters(frame_bgr)
        tp_fraction = read_tp_fraction(frame_bgr)
        priority_pick = self._priority.decide_top_level(low_hp_chars, tp_fraction, labels_only)

        if priority_pick is not None:
            chosen_label = priority_pick
            self._pending_submenu_target_text = self._priority.target_text_for(chosen_label)
            print(
                f"[HeuristicLLMAgent] Priority rule triggered (low HP: {low_hp_chars}, "
                f"TP: {tp_fraction:.0%}) -> forcing: {chosen_label}"
            )
        else:
            chosen_label = self._advisor.choose_action(labels_only)
            self._pending_submenu_target_text = None
            print(f"[HeuristicLLMAgent] Menu detected. Options: {labels_only} -> choosing: {chosen_label}")

        chosen_x = next(x for label, x in visible if label == chosen_label)
        self._in_menu_control = True
        self._target_x = chosen_x
        self._target_label = chosen_label

    def _menu_control_action(self, soul_x: int) -> tuple[int, int]:
        delta = self._target_x - soul_x
        if abs(delta) <= CURSOR_ALIGN_TOLERANCE:
            print(f"[HeuristicLLMAgent] Aligned on {self._target_label}, confirming.")
            self._in_menu_control = False
            self._target_x = None
            self._target_label = None
            if self._pending_submenu_target_text is not None:
                # A priority rule picked this verb specifically to reach a
                # submenu option (e.g. ITEM -> Dark Burger) - proceed there
                # instead of returning to dodge mode.
                self._in_submenu_control = True
            return MOVE_NONE, BUTTON_ENTER
        move_idx = MOVE_RIGHT if delta > 0 else MOVE_LEFT
        return move_idx, BUTTON_NONE

    def _try_begin_submenu(self, frame_bgr) -> bool:
        """
        Attempts to locate the pending target (e.g. "DARK BURGER") among the
        submenu options at SUBMENU_Y_POSITIONS. Returns True once found and
        locked in as this submenu attempt's target.
        """
        target_text = self._pending_submenu_target_text
        for y in SUBMENU_Y_POSITIONS:
            label = self._action_reader.read_label_near(frame_bgr, SUBMENU_X, y)
            if label and (target_text in label or label in target_text):
                self._submenu_target_y = y
                self._submenu_target_label = label
                print(f"[HeuristicLLMAgent] Found '{label}' in submenu at y={y}.")
                return True
        return False

    def _submenu_control_action(self, soul_y: int) -> tuple[int, int]:
        delta = self._submenu_target_y - soul_y
        if abs(delta) <= CURSOR_ALIGN_TOLERANCE:
            print(f"[HeuristicLLMAgent] Aligned on {self._submenu_target_label}, confirming.")
            self._in_submenu_control = False
            self._submenu_target_y = None
            self._submenu_target_label = None
            self._pending_submenu_target_text = None
            return MOVE_NONE, BUTTON_ENTER
        move_idx = MOVE_DOWN if delta > 0 else MOVE_UP
        return move_idx, BUTTON_NONE

    def _dodge_action(self, frame_bgr) -> tuple[int, int]:
        """
        Primarily heuristic_dodge.py's geometric decision logic. At a low,
        configured probability (LLM_DODGE_INFLUENCE_PROBABILITY), the
        decision is instead handed to Groq - see the module docstring for
        why this stays rare rather than being the default path.
        """
        soul_pos = self._dodger._find_soul(frame_bgr)
        if soul_pos is None:
            return MOVE_NONE, BUTTON_NONE

        white = self._dodger._find_nearby_white_centroid(frame_bgr, soul_pos)
        if white is None:
            return MOVE_NONE, BUTTON_NONE  # nothing dangerous nearby - stay put

        if random.random() < LLM_DODGE_INFLUENCE_PROBABILITY:
            sx, sy = soul_pos
            wx, wy, _ = white
            suggestion = self._dodge_advisor.suggest_direction(wx - sx, wy - sy)
            if suggestion is not None:
                direction_to_move_idx = {
                    "UP": MOVE_UP, "DOWN": MOVE_DOWN,
                    "LEFT": MOVE_LEFT, "RIGHT": MOVE_RIGHT,
                }
                return direction_to_move_idx[suggestion], BUTTON_NONE
            # Groq disabled or the call failed/returned something unusable -
            # fall through to the heuristic below rather than stalling.

        dodge_key = self._dodger._choose_dodge_key(soul_pos, white)
        return self._key_to_move_idx[dodge_key], BUTTON_NONE

    def predict(self, obs=None, state=None, episode_start=None, deterministic: bool = True):
        """
        Matches SB3's PPO.predict() signature so this can be swapped in
        anywhere a trained model is expected. `obs` is accepted but unused -
        see the module docstring for why. Returns (action, state) like SB3.
        """
        frame = self._dodger._grab_frame_bgr()

        if self._in_submenu_control:
            fragment_count, soul_pos = find_soul(frame)
            if fragment_count != 1 or soul_pos is None:
                return np.array([MOVE_NONE, BUTTON_NONE]), None  # lost track - wait

            if self._submenu_target_y is None:
                if not self._try_begin_submenu(frame):
                    # Submenu hasn't rendered yet (or OCR hasn't caught the
                    # target this frame) - wait rather than guess.
                    return np.array([MOVE_NONE, BUTTON_NONE]), None

            move_idx, button_idx = self._submenu_control_action(soul_pos[1])
            return np.array([move_idx, button_idx]), None

        if not self._in_menu_control:
            menu_soul_pos = self._detect_menu(frame)
            if menu_soul_pos is not None:
                self._begin_menu_control(frame)

        if self._in_menu_control:
            fragment_count, soul_pos = find_soul(frame)
            if fragment_count == 1 and soul_pos is not None:
                move_idx, button_idx = self._menu_control_action(soul_pos[0])
            else:
                move_idx, button_idx = MOVE_NONE, BUTTON_NONE  # lost track - wait
        else:
            move_idx, button_idx = self._dodge_action(frame)

        action = np.array([move_idx, button_idx])
        return action, None
