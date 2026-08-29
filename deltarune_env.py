"""
deltarune_env.py

A Gymnasium environment that plays Deltarune (Chapter 1) via screen capture
and simulated keyboard input, with no memory reading / Cheat Engine required.

HOW IT WORKS
------------
- Observation: a downsampled grayscale crop of the game window, stacked over
  a few recent frames (so the agent can perceive bullet motion/velocity).
- Action: a MultiDiscrete pair (movement, button):
    movement - up/down/left/right/none, held continuously like a joystick
    button   - confirm/cancel/skip/none, tapped once per step, not held
  This mirrors how the game is actually controlled: you can move the SOUL
  and press confirm in the same instant, so they're independent axes rather
  than one flat list where only one thing can happen per step.
- Reward: primarily driven by detecting the SOUL breaking in half - the
  game's actual fatal-hit animation - via color-based blob detection, with
  a couple of pixel-heuristic fallbacks (dark screen, brightness flash) for
  transitions that might slip past it. See RewardEstimator below.
- Stepping: locked to a target frame rate (see TARGET_FPS) with drift
  correction, so the agent gets a fresh decision point on a consistent
  cadence instead of an arbitrary fixed sleep. Effectively "predicts" a
  move every frame budget rather than every few frames.

BEFORE YOU RUN THIS
--------------------
1. Launch Deltarune yourself and get to the battle you want to train on
   (e.g. the Susie/Lancer fight, or a later bullet-hell wave).
2. Note the pixel coordinates of the game window on your screen. Update
   CAPTURE_REGION below (or use `find_window_region()` on Linux).
3. Keep the game window focused and unobstructed while training - this
   approach is literally watching your screen and pressing keys, so alt-tabbing
   or window movement will break it.
4. This is inherently slower to train than direct memory access. To get a
   "fast learning" agent, keep episodes SHORT (single battle waves, a few
   seconds each) rather than trying to learn all of chapter 1 end-to-end.
5. TARGET_FPS defaults to 30, not 60 - screen capture + preprocessing +
   model inference all add real overhead, and asking for 60 decisions/sec
   when the loop can't sustain it just means constant drift correction
   with no slack. Raise it only if you've confirmed your loop keeps up
   (see the FPS warning printed during training).
6. Button taps release on a background timer rather than blocking the step
   loop - a blocking sleep during every button-press step used to eat most
   of a single frame's budget, which is the main thing that made gameplay
   feel stuttery. Movement stays synchronous since it's a state change
   (press once, hold), not a timed action.
7. Turn-progress reward and menu-action logging need MENU_SOUL_Y calibrated
   to your battle UI - pause on the action-select screen (FIGHT/ACT/ITEM/
   SPARE) and note the SOUL's y pixel position within CAPTURE_REGION. The
   optional OCR logging (MenuActionReader) additionally needs `pytesseract`
   and the `tesseract-ocr` system binary; training works fine without it,
   you just won't get the last_action_label info field populated.

This is intended as a starting point, not a finished product - you will need
to calibrate CAPTURE_REGION, tune RewardEstimator, and possibly replace the
heuristic reward with something more precise (e.g. OCR on the HP display)
once the basic loop is working.
"""

import time
import atexit
import threading
import collections
from typing import Optional

import numpy as np
import cv2
import mss
import gymnasium as gym
from gymnasium import spaces

from input_backend import key_down, key_up


# ---------------------------------------------------------------------------
# CONFIG - calibrate these to your setup
# ---------------------------------------------------------------------------

# Pixel region of your screen where the Deltarune window renders.
# (left, top, width, height) - update these to match your monitor/window.
CAPTURE_REGION = {"left": 100, "top": 100, "width": 640, "height": 480}

# Downsampled observation size fed to the neural net. Smaller = faster training.
OBS_WIDTH = 84
OBS_HEIGHT = 84

# How many recent frames to stack (lets the agent perceive motion/velocity).
FRAME_STACK = 4

# Deltarune's default battle-menu movement keys. Adjust if you've remapped.
KEY_UP = "w"
KEY_DOWN = "s"
KEY_LEFT = "a"
KEY_RIGHT = "d"

# Deltarune's default menu/confirm keys. Adjust if you've remapped.
KEY_ENTER = "z"     # confirm / advance dialogue / select FIGHT-ACT-ITEM-MERCY
KEY_CANCEL = "x"    # cancel / menu back
KEY_SKIP = "space"  # fast-forward / skip text, if you've bound it separately

# Movement axis: held continuously while selected, like a joystick direction.
MOVE_KEYS = [None, KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT]

# Button axis: tapped once per step (pressed briefly then released), never
# held across steps - holding these in a menu causes unwanted repeat/mash
# behavior that doesn't correspond to a single deliberate action.
BUTTON_KEYS = [None, KEY_ENTER, KEY_CANCEL, KEY_SKIP]

# How long a button tap is held down for, in seconds. Short enough to read
# as a single press, long enough for the game to reliably register it.
BUTTON_TAP_SECONDS = 0.03

# Downsampling interpolation. INTER_NEAREST is fastest (matters since this
# runs every single step); INTER_AREA looks slightly cleaner but costs more
# CPU time per frame. Switch to INTER_AREA only if you have FPS headroom.
DOWNSAMPLE_INTERPOLATION = cv2.INTER_NEAREST

# Target decision rate for internal game-frame capture (not the same as how
# often the agent re-decides - see FRAME_SKIP below).
TARGET_FPS = 30
FRAME_BUDGET_SECONDS = 1.0 / TARGET_FPS

# How many internal game frames a single agent decision spans. At
# TARGET_FPS=30, FRAME_SKIP=4 means the agent picks a new (movement, button)
# roughly 7.5 times/sec instead of 30 - fewer, more deliberate decisions
# instead of a fresh (possibly different) choice every single frame, which
# is what "frantic" key-pressing usually comes from. Raise for calmer/more
# deliberate play, lower if it feels sluggish to react to fast patterns.
FRAME_SKIP = 4

# Penalty applied when the agent's movement choice differs from its last
# decision (a direction change). Small enough not to prevent real dodges,
# but enough to make flip-flopping direction every decision cost something,
# so holding a direction is the "cheaper" default unless there's a reason
# to change it.
MOVEMENT_SWITCH_PENALTY = 0.05

# After tapping a button (confirm/cancel/skip), how many agent decisions
# must pass before another button tap is allowed. Prevents rapid menu-mash
# behavior where the agent spams confirm every decision.
BUTTON_COOLDOWN_DECISIONS = 3

# HSV bounds for detecting the SOUL (red heart). Two ranges are used because
# red wraps around hue 0/180 in OpenCV's HSV space. Same idea as
# heuristic_dodge.py - if detection isn't reliable, sample the SOUL's actual
# HSV value from a screenshot and narrow these.
RED_LOWER_1 = np.array([0, 120, 100])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([170, 120, 100])
RED_UPPER_2 = np.array([180, 255, 255])

# Minimum pixel area for a red blob to count as a real SOUL fragment, rather
# than compression noise or a stray red pixel elsewhere on screen.
MIN_SOUL_FRAGMENT_AREA = 8

# ---------------------------------------------------------------------------
# ACTION-MENU TRACKING - calibrate these to your fight's UI
# ---------------------------------------------------------------------------
# Deltarune's action menu (FIGHT / ACT / ITEM / SPARE, or ACT / DEFEND for
# Ralsei, etc.) has you move the SOUL next to a verb and press confirm. We
# can't read the text itself without OCR (see MenuActionReader below), but
# we CAN tell "a turn was just completed" from the SOUL's position: it sits
# at a fixed row while the menu is open, then leaves that row once a choice
# is confirmed and the turn proceeds. That transition is a clean, reliable
# signal that the agent should be rewarded for successfully acting - much
# more informative than survival-only reward, which never tells it "the
# menu exists and doing something there matters."
#
# Calibrate MENU_SOUL_Y by pausing on the action-select screen and noting
# the SOUL's y pixel position (full-res, within CAPTURE_REGION).
MENU_SOUL_Y = 400
MENU_Y_TOLERANCE = 15  # pixels - "in the menu" if within this of MENU_SOUL_Y

# Reward for successfully confirming a menu action (turn completed).
TURN_PROGRESS_REWARD = 2.0

# X-coordinates (full-res, within CAPTURE_REGION) of each verb's position,
# left to right. This varies by whose turn it is - Kris/Susie get
# FIGHT/ACT/ITEM/SPARE, Ralsei gets ACT/DEFEND/ITEM/SPARE (no FIGHT), etc.
# Edit this list to match the current turn's actual layout before training;
# it's used only for optional OCR logging (see MenuActionReader), not for
# the turn-progress reward itself, which is layout-agnostic.
MENU_VERB_X_CENTERS = [120, 250, 380, 510]

# Discrete action space is now (movement, button) - two independent choices
# per step rather than one flat list, so both can happen simultaneously.
ACTION_SPACE_NVEC = [len(MOVE_KEYS), len(BUTTON_KEYS)]

# ---------------------------------------------------------------------------
# CORNER HANDLING - mirrors heuristic_dodge.py's SPIKES_AT_CORNERS idea, but
# expressed as reward shaping instead of a hard-coded steering rule, since
# the RL agent has to learn the behavior rather than follow it directly.
# ---------------------------------------------------------------------------
# The rectangle the SOUL can move inside during a dodge phase (full-res
# pixel coordinates, within CAPTURE_REGION - same space as MENU_SOUL_Y).
# Calibrate by pausing during a bullet pattern and noting the box's border.
BATTLE_BOX = {"x0": 400, "y0": 300, "x1": 900, "y1": 650}

# Whether THIS fight's battle box actually has spike hazards sitting in its
# corners. If True, the SOUL sitting in a corner while the enemy is
# attacking (i.e. outside the action menu - see "in_menu" below) gets a
# small reward penalty, so the agent learns to avoid them. If False (the
# default), corners are just as safe as anywhere else in the box and no
# penalty applies - there's nothing there to teach the agent to avoid.
SPIKES_AT_CORNERS = False

# How close (pixels) to TWO walls at once counts as "in a corner" for this
# penalty - same idea as heuristic_dodge.py's CORNER_ESCAPE_MARGIN.
CORNER_MARGIN = 45

# Reward penalty applied per step the SOUL spends in a corner during an
# attack phase, when SPIKES_AT_CORNERS is True. Kept small relative to the
# SOUL-break penalty (-10.0) below - sitting in a corner isn't itself
# fatal, so this should discourage the habit without swamping the actual
# death signal in training.
CORNER_PENALTY = 0.3


class MenuActionReader:
    """
    Optional: reads the text label the SOUL is hovering over via OCR, purely
    for logging/visibility into what the agent is actually choosing during
    training - it does NOT feed into the reward. Pixel-art fonts are small
    and OCR misreads are common, so treat this as "probably right, useful
    for spotting patterns," not ground truth.

    Requires: pip install pytesseract, and the tesseract-ocr binary
    installed on your system (e.g. `sudo apt install tesseract-ocr`).
    If pytesseract isn't installed, this silently disables itself so the
    rest of training still works without it.
    """

    LABEL_WHITELIST = "FIGHTACTIEMSPARDEFNYRfightactitemspardefny "
    CROP_HALF_WIDTH = 45
    CROP_HALF_HEIGHT = 12

    def __init__(self):
        try:
            import pytesseract
            self._pytesseract = pytesseract
            self.enabled = True
        except ImportError:
            self._pytesseract = None
            self.enabled = False

    def read_label_near(self, frame_bgr: np.ndarray, soul_x: int, soul_y: int) -> Optional[str]:
        if not self.enabled:
            return None
        h, w = frame_bgr.shape[:2]
        x0 = max(soul_x - self.CROP_HALF_WIDTH, 0)
        x1 = min(soul_x + self.CROP_HALF_WIDTH, w)
        y0 = max(soul_y - self.CROP_HALF_HEIGHT, 0)
        y1 = min(soul_y + self.CROP_HALF_HEIGHT, h)
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Upscale small pixel-art text before OCR - tesseract does much
        # better with larger, smoother glyphs than tiny sharp-edged sprites.
        upscaled = cv2.resize(gray_crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        try:
            text = self._pytesseract.image_to_string(
                upscaled,
                config=f'--psm 7 -c tessedit_char_whitelist="{self.LABEL_WHITELIST}"',
            )
        except Exception:
            return None
        cleaned = text.strip().upper()
        return cleaned or None

    def read_all_visible_labels(self, frame_bgr: np.ndarray) -> list[Optional[str]]:
        """
        Reads every verb slot in MENU_VERB_X_CENTERS along MENU_SOUL_Y,
        returning a list the same length as MENU_VERB_X_CENTERS (None for
        slots that didn't OCR to anything - e.g. a character with fewer
        than 4 options that turn). Used by hybrid_play.py to see the full
        set of choices available before picking one, not just whichever one
        the SOUL happened to be on.
        """
        if not self.enabled:
            return [None] * len(MENU_VERB_X_CENTERS)
        return [
            self.read_label_near(frame_bgr, x_center, MENU_SOUL_Y)
            for x_center in MENU_VERB_X_CENTERS
        ]


def find_soul(frame_bgr: np.ndarray) -> tuple[int, Optional[tuple[int, int]]]:
    """
    Detect the SOUL (red heart) in a color frame via HSV thresholding.

    Returns (fragment_count, centroid). centroid is the largest red blob's
    (x, y) if fragment_count == 1 (an intact SOUL), else None - once it's
    broken into pieces, "where is the SOUL" is ambiguous (and usually the
    episode/turn is ending anyway). Shared by RewardEstimator (for SOUL-break
    and turn detection) and hybrid_play.py (for menu-cursor navigation).
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1) | cv2.inRange(
        hsv, RED_LOWER_2, RED_UPPER_2
    )
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    fragments = [
        i for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] >= MIN_SOUL_FRAGMENT_AREA
    ]
    if len(fragments) == 1:
        cx, cy = centroids[fragments[0]]
        return 1, (int(cx), int(cy))
    return len(fragments), None


class RewardEstimator:
    """
    Heuristic reward signal computed purely from pixel data, since we have
    no direct read on HP/game-state without memory access.

    Strategy:
      - Small positive reward per frame survived (encourages staying alive).
      - Reward for completing a turn: SOUL sits at the menu row, then leaves
        it - detected as a confirmed action rather than aimless movement.
        This is what actually teaches the agent that the menu exists and
        interacting with it matters, rather than only ever dodging.
      - Large negative reward + episode end when the SOUL is detected
        breaking in half - this is Deltarune's actual fatal-hit animation
        (the heart visually splits into two pieces), so it's a much more
        specific and reliable death signal than a generic screen flash.
      - Large negative reward if the frame looks like a game-over screen
        (very dark / mostly black frame, held for multiple steps) - kept as
        a fallback for transitions the SOUL-break check might miss.
      - Smaller negative reward on a generic screen "flash", which can catch
        non-fatal hits that don't break the SOUL but still cost HP.
      - Optional small negative reward for sitting in a battle-box corner
        while the enemy is attacking, but ONLY if SPIKES_AT_CORNERS is set
        for this fight - otherwise corners are unremarkable and untouched.

    HOW SOUL-BREAK DETECTION WORKS: the SOUL is a solid red blob. Under
    normal play it's a single connected red region. When it breaks, that
    region splits into two (or more) separate red blobs. We count red
    connected components each frame; a rise from one blob to two-or-more
    is treated as the break event.

    HOW TURN DETECTION WORKS: the SOUL sits at a known row (MENU_SOUL_Y)
    while the action-select menu is open, and leaves that row once you
    confirm a choice. We track a boolean "was the SOUL at the menu row last
    frame" and reward the frame it transitions from True to False - a
    completed, confirmed action. This works regardless of which specific
    verb was chosen (FIGHT/ACT/ITEM/SPARE/DEFEND), so it doesn't need to
    know the menu layout - only MenuActionReader (optional, for logging)
    cares about the actual verb positions.

    TUNE THIS. Pixel heuristics are fragile - if SOUL-break detection isn't
    firing reliably, sample the SOUL's actual on-screen HSV color and adjust
    RED_LOWER/UPPER, or the more robust upgrade path is OCR on the HP number
    via pytesseract, or eventually reading memory directly with pymem
    (skipping Cheat Engine by scripting the address search).
    """

    def __init__(self, action_reader: Optional["MenuActionReader"] = None):
        self.prev_frame: Optional[np.ndarray] = None
        self.dark_frame_streak = 0
        self.prev_soul_fragment_count = 1  # assume intact SOUL at episode start
        self.was_in_menu = False
        self.turn_count = 0
        self.last_action_label: Optional[str] = None
        self.action_reader = action_reader

    def reset(self):
        self.prev_frame = None
        self.dark_frame_streak = 0
        self.prev_soul_fragment_count = 1
        self.was_in_menu = False
        self.turn_count = 0
        self.last_action_label = None

    def estimate(self, frame_bgr: np.ndarray) -> tuple[float, bool]:
        """
        Returns (reward, done) for the current frame.
        frame_bgr: full-resolution BGR color capture (before downsampling).
        """
        reward = 0.01  # small survival bonus per step
        done = False

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean_brightness = gray.mean()

        # Heuristic 1: SOUL break - the primary "did I just die" signal.
        fragment_count, soul_pos = find_soul(frame_bgr)
        if fragment_count >= 2 and self.prev_soul_fragment_count < 2:
            reward -= 10.0
            done = True
        self.prev_soul_fragment_count = fragment_count

        # Heuristic 1b: turn completion - SOUL was at the menu row, now isn't.
        if soul_pos is not None and not done:
            in_menu = abs(soul_pos[1] - MENU_SOUL_Y) <= MENU_Y_TOLERANCE
            if self.was_in_menu and not in_menu:
                reward += TURN_PROGRESS_REWARD
                self.turn_count += 1
                if self.action_reader is not None and self.action_reader.enabled:
                    # Read the label from where the SOUL WAS (last frame's
                    # position is gone, so approximate using this frame's
                    # nearest verb slot - good enough for logging purposes).
                    self.last_action_label = self.action_reader.read_label_near(
                        frame_bgr, soul_pos[0], MENU_SOUL_Y
                    )
            self.was_in_menu = in_menu

            # Heuristic 1c: corner penalty, attack phase only. "Not in the
            # menu" is our proxy for "the enemy is attacking and the SOUL is
            # actually dodging" (the menu is only up between turns, when
            # nothing is flying at you). Only applies when this fight's box
            # actually has spike hazards in its corners - if SPIKES_AT_CORNERS
            # is False, corners are unremarkable open space and the agent is
            # free to pass through or sit in them like anywhere else in the box.
            if SPIKES_AT_CORNERS and not in_menu:
                sx, sy = soul_pos
                near_x_wall = (
                    (sx - BATTLE_BOX["x0"]) < CORNER_MARGIN
                    or (BATTLE_BOX["x1"] - sx) < CORNER_MARGIN
                )
                near_y_wall = (
                    (sy - BATTLE_BOX["y0"]) < CORNER_MARGIN
                    or (BATTLE_BOX["y1"] - sy) < CORNER_MARGIN
                )
                if near_x_wall and near_y_wall:
                    reward -= CORNER_PENALTY

        # Heuristic 2: game-over / battle-lost screens in Deltarune are
        # mostly black. If we see near-black frames for a sustained streak,
        # treat the episode as over. Kept as a fallback in case SOUL-break
        # detection misses a transition (e.g. instant wipes, cutscenes).
        if mean_brightness < 12:
            self.dark_frame_streak += 1
        else:
            self.dark_frame_streak = 0

        if self.dark_frame_streak > TARGET_FPS:  # ~1s of near-black frames
            reward -= 5.0
            done = True

        # Heuristic 3: sudden large brightness spike/flash often corresponds
        # to taking a (possibly non-fatal) hit in Deltarune's battle UI.
        # This is approximate - tune the threshold against your own recordings.
        if self.prev_frame is not None:
            diff = np.abs(gray.astype(np.int16) - self.prev_frame.astype(np.int16))
            flash_ratio = (diff > 80).mean()
            if flash_ratio > 0.35:
                reward -= 1.0

        self.prev_frame = gray
        return reward, done


class DeltaruneEnv(gym.Env):
    """Gymnasium environment for a single Deltarune battle/dodge segment."""

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 600, warn_on_fps_drop: bool = True):
        super().__init__()

        self.action_space = spaces.MultiDiscrete(ACTION_SPACE_NVEC)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(FRAME_STACK, OBS_HEIGHT, OBS_WIDTH),
            dtype=np.uint8,
        )

        self.sct = mss.mss()
        self.action_reader = MenuActionReader()
        self.reward_estimator = RewardEstimator(action_reader=self.action_reader)
        self.frames = collections.deque(maxlen=FRAME_STACK)
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self._held_move_key: Optional[str] = None
        self._pending_button_key: Optional[str] = None
        self._button_timer: Optional[threading.Timer] = None
        self.warn_on_fps_drop = warn_on_fps_drop
        self._drop_streak = 0
        self._prev_move_idx: Optional[int] = None  # for the direction-switch penalty
        self._button_cooldown_remaining = 0        # decisions left before another tap is allowed

        # Safety net: if training crashes or is interrupted mid-episode, a
        # held movement key would otherwise stay pressed in the game forever.
        # This guarantees keys get released even on an unhandled exception.
        atexit.register(self._release_all_keys)

    # -- capture / control helpers -----------------------------------------

    def _grab_frame(self) -> np.ndarray:
        """Capture the game window region and return a BGR color full-res frame.

        Color (not just grayscale) is needed here because RewardEstimator
        detects the SOUL breaking apart via its red color - the observation
        stack still uses grayscale, derived from this via _to_gray_obs.
        """
        raw = np.array(self.sct.grab(CAPTURE_REGION))
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

    def _to_gray_obs(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (OBS_WIDTH, OBS_HEIGHT), interpolation=DOWNSAMPLE_INTERPOLATION)

    def _apply_move(self, move_idx: int):
        """Update the held movement key only if the direction actually changed."""
        move_key = MOVE_KEYS[move_idx]
        if move_key == self._held_move_key:
            return  # already holding the right direction (or nothing) - no-op
        if self._held_move_key is not None:
            key_up(self._held_move_key)
        if move_key is not None:
            key_down(move_key)
        self._held_move_key = move_key

    def _apply_button(self, button_idx: int):
        """
        Tap a button key without blocking the step loop.

        Pressing then sleeping BUTTON_TAP_SECONDS inline used to eat most of
        a single frame's budget at TARGET_FPS - a 30ms blocking sleep against
        a ~33ms budget left almost no room for capture/inference, causing the
        exact stutter the FPS-drop warning flags. Instead, press immediately
        and schedule the release on a background timer so the step loop can
        move straight on to capturing the next frame.
        """
        button_key = BUTTON_KEYS[button_idx]
        if button_key is None:
            return

        # If a previous tap's release hasn't fired yet, finish it immediately
        # rather than letting two taps overlap or releasing the wrong key.
        if self._button_timer is not None:
            self._button_timer.cancel()
            if self._pending_button_key is not None:
                key_up(self._pending_button_key)

        key_down(button_key)
        self._pending_button_key = button_key
        self._button_timer = threading.Timer(
            BUTTON_TAP_SECONDS, self._release_button, args=(button_key,)
        )
        self._button_timer.daemon = True
        self._button_timer.start()

    def _release_button(self, button_key: str):
        try:
            key_up(button_key)
        except Exception:
            pass  # best-effort - env may be shutting down
        if self._pending_button_key == button_key:
            self._pending_button_key = None

    def _release_all_keys(self):
        if self._button_timer is not None:
            self._button_timer.cancel()
            self._button_timer = None
        if self._pending_button_key is not None:
            try:
                key_up(self._pending_button_key)
            except Exception:
                pass
            self._pending_button_key = None
        if self._held_move_key is not None:
            try:
                key_up(self._held_move_key)
            except Exception:
                pass  # best-effort on shutdown/crash paths
            self._held_move_key = None

    # -- gym API -------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._release_all_keys()
        self.reward_estimator.reset()
        self.current_step = 0
        self._drop_streak = 0
        self._prev_move_idx = None
        self._button_cooldown_remaining = 0

        frame = self._grab_frame()
        down = self._to_gray_obs(frame)
        self.frames.clear()
        for _ in range(FRAME_STACK):
            self.frames.append(down)

        obs = np.stack(self.frames, axis=0)
        return obs, {}

    def _run_one_internal_frame(self, move_idx: int, button_idx: int):
        """
        Apply one (movement, button) pair for a single internal game frame,
        capture the result, and score it. Returns (down_obs, reward, done)
        or raises on capture/input failure - caller handles exceptions.
        """
        self._apply_move(move_idx)
        self._apply_button(button_idx)
        frame = self._grab_frame()
        reward, done = self.reward_estimator.estimate(frame)
        down = self._to_gray_obs(frame)
        return down, reward, done

    def step(self, action):
        move_idx, button_idx = int(action[0]), int(action[1])

        # Direction-switch penalty: charged once per agent decision (not per
        # internal frame), so flip-flopping direction every decision costs
        # something and holding a direction is the cheaper default. Applied
        # up front so it's part of the total regardless of what happens
        # during the frame-skip window below.
        total_reward = 0.0
        if self._prev_move_idx is not None and move_idx != self._prev_move_idx:
            total_reward -= MOVEMENT_SWITCH_PENALTY
        self._prev_move_idx = move_idx

        # Button cooldown: while on cooldown, silently treat any requested
        # button press as "none" so the agent can't spam confirm/cancel/skip
        # every decision. The movement axis is unaffected.
        effective_button_idx = button_idx
        if button_idx != 0:
            if self._button_cooldown_remaining > 0:
                effective_button_idx = 0
            else:
                self._button_cooldown_remaining = BUTTON_COOLDOWN_DECISIONS
        if self._button_cooldown_remaining > 0:
            self._button_cooldown_remaining -= 1

        done = False
        truncated = False
        info = {}

        try:
            # FRAME_SKIP: one agent decision plays out over several internal
            # game frames rather than being re-chosen every single frame.
            # The button only needs to fire on the first internal frame -
            # it's a tap, not something that should repeat FRAME_SKIP times.
            for i in range(FRAME_SKIP):
                frame_start = time.perf_counter()
                this_frame_button_idx = effective_button_idx if i == 0 else 0

                down, reward, frame_done = self._run_one_internal_frame(
                    move_idx, this_frame_button_idx
                )
                self.frames.append(down)
                total_reward += reward

                self.current_step += 1
                truncated = self.current_step >= self.max_episode_steps

                if frame_done or truncated:
                    done = frame_done
                    break

                # Frame-rate lock with drift correction, same as before but
                # now per internal frame within the skip window.
                elapsed = time.perf_counter() - frame_start
                remaining = FRAME_BUDGET_SECONDS - elapsed
                if remaining > 0:
                    time.sleep(remaining)
                    self._drop_streak = 0
                else:
                    self._drop_streak += 1
                    if self.warn_on_fps_drop and self._drop_streak == TARGET_FPS:
                        print(
                            f"[DeltaruneEnv] Warning: can't sustain {TARGET_FPS} FPS "
                            f"(loop taking {elapsed * 1000:.0f}ms/frame). Consider "
                            f"lowering TARGET_FPS, FRAME_SKIP, or OBS_WIDTH/OBS_HEIGHT."
                        )
        except Exception as exc:
            # A transient capture/input failure (e.g. window briefly lost
            # focus) shouldn't kill the whole training run. End this episode
            # cleanly and let the next reset() start fresh instead.
            print(f"[DeltaruneEnv] step() error, ending episode early: {exc}")
            self._release_all_keys()
            obs = np.stack(self.frames, axis=0) if self.frames else np.zeros(
                self.observation_space.shape, dtype=np.uint8
            )
            return obs, total_reward - 1.0, True, False, {"error": str(exc)}

        obs = np.stack(self.frames, axis=0)

        if done or truncated:
            self._release_all_keys()

        info = {
            "turn_count": self.reward_estimator.turn_count,
            "last_action_label": self.reward_estimator.last_action_label,
        }

        return obs, total_reward, done, truncated, info

    def close(self):
        self._release_all_keys()
        self.sct.close()


def find_window_region(title_substring: str) -> dict:
    """
    Optional helper to auto-detect the Deltarune window's screen coordinates
    on Linux (X11/XWayland), using `xdotool` instead of hardcoding
    CAPTURE_REGION by hand.

    Requires the `xdotool` CLI to be installed (e.g. `sudo apt install xdotool`).

    Usage:
        CAPTURE_REGION = find_window_region("DELTARUNE")
    """
    import subprocess

    result = subprocess.run(
        ["xdotool", "search", "--name", title_substring],
        capture_output=True, text=True, check=True,
    )
    window_ids = result.stdout.strip().splitlines()
    if not window_ids:
        raise RuntimeError(f"No window found matching '{title_substring}'")
    window_id = window_ids[0]

    geo = subprocess.run(
        ["xdotool", "getwindowgeometry", "--shell", window_id],
        capture_output=True, text=True, check=True,
    )
    values = dict(line.split("=") for line in geo.stdout.strip().splitlines())
    return {
        "left": int(values["X"]),
        "top": int(values["Y"]),
        "width": int(values["WIDTH"]),
        "height": int(values["HEIGHT"]),
    }
