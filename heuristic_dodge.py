"""
heuristic_dodge.py

A standalone, non-RL script that plays Deltarune's dodging segments with a
simple reactive rule: find the SOUL (the red heart), look for white pixels
near it, and hold the movement key that steers away from them.

This has nothing to do with the RL training project in deltarune_env.py /
train.py - it's a hand-coded heuristic bot, not a learned policy. Useful as
a quick baseline to compare a trained agent against, or just to see the
dodge-detection idea working before building anything RL on top of it.

HOW IT WORKS
------------
1. Capture the game window region each loop iteration.
2. Find the SOUL's position by color: it's a solid red heart in Deltarune's
   battle UI, so a red color-threshold mask + centroid gives its (x, y).
3. Look at a box of radius DANGER_RADIUS pixels around the SOUL and count
   white pixels in it (most bullet patterns render as white/bright shapes).
4. If enough white pixels are found, compute their centroid relative to the
   SOUL and hold the movement key pointing the opposite way (dodge away).
   If nothing dangerous is nearby, release all movement keys and stay put.
5. Before each of the above, checks whether the game window is actually
   focused (see WINDOW_TITLE_SUBSTRING). If it's minimized or you've
   alt-tabbed away, the bot releases any held key and pauses rather than
   pressing keys into whatever window has focus instead, or reacting to a
   stale/garbage capture of your desktop.

TUNING
------
- RED_LOWER / RED_UPPER: HSV bounds for the SOUL color. Deltarune's SOUL is
  a fairly saturated red, but lighting/monitor color profiles vary - if
  detection isn't finding it, screenshot the game and sample the SOUL's
  actual HSV value to narrow these bounds.
- WHITE_BRIGHTNESS_THRESHOLD: how bright (per-channel) a pixel must be to
  count as "white". Deltarune bullets are usually near-pure white, but some
  patterns use off-white or colored bullets this script won't catch - this
  is a simple heuristic, not general bullet detection.
- DANGER_RADIUS: how far around the SOUL counts as "near". Too small and it
  won't react until bullets are already very close; too large and distant
  bullets that aren't actually a threat yet will trigger dodges.
- MIN_WHITE_BLOB_AREA / TEXT_STRIP_KERNEL_SIZE: dialogue and UI text is
  thin, fragmented white strokes - without filtering, enough of it inside
  DANGER_RADIUS could look like a bullet and trigger a false dodge. A
  morphological opening erases strokes thinner than TEXT_STRIP_KERNEL_SIZE
  before we count blobs, and MIN_WHITE_BLOB_AREA requires what's left to be
  a real contiguous shape, not noise. If a real bullet stops being detected,
  lower TEXT_STRIP_KERNEL_SIZE or MIN_WHITE_BLOB_AREA; if text still causes
  false dodges, raise them.

Before running, calibrate CAPTURE_REGION below (or import find_window_region
from deltarune_env.py) to match your Deltarune window.
"""

import time
import atexit
import subprocess
from typing import Optional

import numpy as np
import cv2
import mss

from input_backend import key_down, key_up

# ---------------------------------------------------------------------------
# CONFIG - calibrate these to your setup
# ---------------------------------------------------------------------------

CAPTURE_REGION = {"left": 0, "top": 32, "width": 1920, "height": 1048}

# Movement keys - matches deltarune_env.py's WASD remap.
KEY_UP = "w"
KEY_DOWN = "s"
KEY_LEFT = "a"
KEY_RIGHT = "d"

# HSV bounds for detecting the SOUL (red heart). Two ranges are used because
# red wraps around hue 0/180 in OpenCV's HSV space.
RED_LOWER_1 = np.array([0, 120, 100])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([170, 120, 100])
RED_UPPER_2 = np.array([180, 255, 255])

# A pixel counts as "white" if all three BGR channels exceed this value.
WHITE_BRIGHTNESS_THRESHOLD = 220

# How far around the SOUL (in pixels, at capture resolution) counts as "near".
DANGER_RADIUS = 40

# Minimum size (in pixels) of a single contiguous white blob before it's
# treated as a real bullet threat. Dialogue/UI text is made of thin,
# fragmented strokes - a morphological "opening" (erode then dilate) below
# wipes those out while leaving solid filled bullet shapes intact, so text
# on screen no longer triggers false dodges.
MIN_WHITE_BLOB_AREA = 20

# Kernel size for the text-stripping morphological opening. Roughly: strokes
# thinner than this (in pixels) get erased. Increase if thicker fonts still
# slip through; decrease if it's erasing small/thin real bullets too.
TEXT_STRIP_KERNEL_SIZE = 3

# Loop rate: seconds between each capture-decide-act cycle. Lower = more
# responsive dodging but more CPU/screen-capture load; higher = calmer,
# more deliberate action but slower reaction to fast-moving bullets.
ACTION_INTERVAL_SECONDS = 0.5
FRAME_BUDGET_SECONDS = ACTION_INTERVAL_SECONDS

# ---------------------------------------------------------------------------
# BATTLE BOX / CORNER AVOIDANCE - calibrate to your fight's bullet-hell box
# ---------------------------------------------------------------------------
# The original dodge rule only ever asks "which way is away from the nearest
# bullet?" with no idea where the box walls are. If bullets keep coming from
# roughly the same general direction, that steadily retreats the SOUL into a
# corner - and once it's there, "away from the bullet" and "away from the
# wall" can point in opposite directions with no safe option left. The fix
# is to blend a wall-repulsion force into every dodge decision, and to
# actively walk back toward the center when idle near a corner.
#
# Calibrate BATTLE_BOX by pausing during a bullet pattern and noting the
# pixel coordinates of the box's white border, full-res within
# CAPTURE_REGION (same coordinate space as MENU_SOUL_Y in deltarune_env.py).
BATTLE_BOX = {"x0": 400, "y0": 300, "x1": 900, "y1": 650}

# How close (in pixels) to a wall before it starts counter-steering the
# dodge. Larger = the SOUL starts respecting walls earlier/more cautiously.
WALL_MARGIN = 60

# How strongly "avoid the wall" competes with "avoid the bullet", in the
# same pixel-distance units _choose_dodge_key already compares. At
# WALL_MARGIN distance from a wall this contributes 0; right at the wall it
# contributes WALL_REPEL_STRENGTH. Tune down if the SOUL starts ignoring
# real bullets just to back off a wall; tune up if it's still getting
# cornered.
WALL_REPEL_STRENGTH = 70

# When no bullet is nearby (idle), how close to TWO walls at once (i.e. a
# corner) triggers a deliberate step back toward the box center, rather than
# just sitting still. This is what actually prevents "parked in a corner"
# rather than just softening dodges near one.
CORNER_ESCAPE_MARGIN = 45

# Some bullet-hell patterns place an actual hazard (spikes) in the box
# corners - not just a wall, but something that hurts the SOUL just for
# being there. If your fight has these, set this True to treat corners as
# more dangerous than a plain wall: both the margin and the repulsion
# strength get boosted specifically in the corner zones (see the two
# constants below), on top of the normal per-wall push. Leave False for
# fights where corners are just dead space against two walls.
SPIKES_AT_CORNERS = False

# Extra margin (pixels, added on top of WALL_MARGIN / CORNER_ESCAPE_MARGIN)
# applied near corners specifically, when SPIKES_AT_CORNERS is True. Makes
# the SOUL start backing off a corner earlier than it would a plain wall.
SPIKE_CORNER_MARGIN_BONUS = 30

# Extra repulsion strength (added on top of WALL_REPEL_STRENGTH) applied
# near corners specifically, when SPIKES_AT_CORNERS is True. Makes the
# SOUL back off harder once it's in that zone, since sitting near a spike
# corner is worse than sitting near a plain wall.
SPIKE_CORNER_REPEL_BONUS = 80

# Substring to match (case-insensitive) against the active window's title.
# Used to detect when the game isn't focused - minimized or alt-tabbed away
# - so the bot pauses instead of pressing keys into whatever window happens
# to have focus instead, or reacting to a capture of your desktop/taskbar.
# A minimized window can never be the active window, so this check covers
# minimizing as well as alt-tabbing away. Requires `xdotool`
# (sudo apt install xdotool) - same tool find_window_region() in
# deltarune_env.py uses. Adjust to match your Deltarune window's title.
WINDOW_TITLE_SUBSTRING = "DELTARUNE"

# How often to re-check window focus while paused (not focused). Doesn't
# need to be fast - regaining focus a moment late is harmless, and this
# just avoids busy-looping a tight capture loop while minimized.
FOCUS_POLL_INTERVAL_SECONDS = 0.25


class HeuristicDodger:
    def __init__(self):
        self.sct = mss.mss()
        self._held_key: Optional[str] = None
        self._warned_no_xdotool = False
        atexit.register(self._release_all_keys)

    # -- window focus -----------------------------------------------------

    def _is_game_focused(self) -> bool:
        """
        True if the currently active window's title matches
        WINDOW_TITLE_SUBSTRING - i.e. the game is focused and not minimized
        (a minimized window can never be the active one).

        Fails open (assumes focused) if xdotool isn't installed or errors,
        so a missing dependency degrades to the old always-react behavior
        instead of the bot silently refusing to do anything.
        """
        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True, text=True, timeout=1,
            )
            if result.returncode != 0:
                return True  # no active window info - don't block on it
            return WINDOW_TITLE_SUBSTRING.lower() in result.stdout.strip().lower()
        except Exception:
            if not self._warned_no_xdotool:
                print(
                    "[HeuristicDodger] xdotool unavailable - can't detect "
                    "focus/minimize state, will react even if the game "
                    "isn't focused. Install with: sudo apt install xdotool"
                )
                self._warned_no_xdotool = True
            return True

    # -- perception -----------------------------------------------------

    def _grab_frame_bgr(self) -> np.ndarray:
        raw = np.array(self.sct.grab(CAPTURE_REGION))
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

    def _find_soul(self, frame_bgr: np.ndarray) -> Optional[tuple[int, int]]:
        """Return (x, y) centroid of the largest red blob, or None if not found."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
        mask2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
        mask = mask1 | mask2

        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            return None  # no red pixels found - SOUL not visible this frame

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return cx, cy

    def _find_nearby_white_centroid(
        self, frame_bgr: np.ndarray, soul_pos: tuple[int, int]
    ) -> Optional[tuple[float, float, int]]:
        """
        Look within DANGER_RADIUS of soul_pos for a real bullet - a solid
        white blob, not scattered text pixels. Returns (mean_x, mean_y,
        area) of the largest qualifying blob relative to the crop, or None
        if nothing big enough is found.
        """
        h, w = frame_bgr.shape[:2]
        cx, cy = soul_pos
        x0, x1 = max(cx - DANGER_RADIUS, 0), min(cx + DANGER_RADIUS, w)
        y0, y1 = max(cy - DANGER_RADIUS, 0), min(cy + DANGER_RADIUS, h)
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return None

        white_mask = np.all(crop >= WHITE_BRIGHTNESS_THRESHOLD, axis=2).astype(np.uint8)

        # Morphological opening: erode away thin strokes (text), then dilate
        # back to restore the shape of whatever solid blobs survive (bullets).
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (TEXT_STRIP_KERNEL_SIZE, TEXT_STRIP_KERNEL_SIZE)
        )
        cleaned_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned_mask, connectivity=8
        )
        if num_labels <= 1:
            return None  # nothing survived the text-stripping pass

        # Pick the largest surviving blob (label 0 is background).
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = int(np.argmax(areas)) + 1
        largest_area = int(stats[largest_idx, cv2.CC_STAT_AREA])
        if largest_area < MIN_WHITE_BLOB_AREA:
            return None

        bx, by = centroids[largest_idx]
        mean_x = bx + x0  # back to full-frame coordinates
        mean_y = by + y0
        return mean_x, mean_y, largest_area

    # -- decision + action ------------------------------------------------

    def _wall_push(self, soul_pos: tuple[int, int]) -> tuple[float, float]:
        """
        Return a (push_x, push_y) vector pointing back toward the box
        center, scaled from 0 (at WALL_MARGIN or farther from every wall)
        up to WALL_REPEL_STRENGTH (pressed right against a wall). Zero
        near the middle of the box, nonzero and inward-pointing near any
        edge - and near a corner, both components kick in at once, which
        is exactly what steers the SOUL away from getting pinned there.

        If SPIKES_AT_CORNERS is set and the SOUL is currently close to TWO
        walls at once (i.e. actually in a corner zone, not just near one
        wall), both the margin and strength get boosted for this push -
        corners get treated as more dangerous than a plain wall.
        """
        sx, sy = soul_pos
        box = BATTLE_BOX

        # First pass with the base margin, just to check whether we're in
        # a corner zone (near two walls at once) at all.
        near_x_wall = (sx - box["x0"]) < WALL_MARGIN or (box["x1"] - sx) < WALL_MARGIN
        near_y_wall = (sy - box["y0"]) < WALL_MARGIN or (box["y1"] - sy) < WALL_MARGIN
        in_spike_corner = SPIKES_AT_CORNERS and near_x_wall and near_y_wall

        margin = WALL_MARGIN + (SPIKE_CORNER_MARGIN_BONUS if in_spike_corner else 0)
        strength = WALL_REPEL_STRENGTH + (SPIKE_CORNER_REPEL_BONUS if in_spike_corner else 0)

        push_x = push_y = 0.0

        dist_left = sx - box["x0"]
        dist_right = box["x1"] - sx
        if dist_left < margin:
            push_x += strength * (1 - max(dist_left, 0) / margin)
        if dist_right < margin:
            push_x -= strength * (1 - max(dist_right, 0) / margin)

        dist_top = sy - box["y0"]
        dist_bottom = box["y1"] - sy
        if dist_top < margin:
            push_y += strength * (1 - max(dist_top, 0) / margin)
        if dist_bottom < margin:
            push_y -= strength * (1 - max(dist_bottom, 0) / margin)

        return push_x, push_y

    def _corner_escape_key(self, soul_pos: tuple[int, int]) -> Optional[str]:
        """
        If idle (no bullet nearby) and pinned near a corner - i.e. close to
        TWO walls at once - return a key that steps back toward the box
        center instead of sitting still. Returns None if not near a corner.

        Uses a wider margin (CORNER_ESCAPE_MARGIN + SPIKE_CORNER_MARGIN_BONUS)
        when SPIKES_AT_CORNERS is set, so it starts retreating from a
        spiked corner sooner than it would an ordinary one.
        """
        sx, sy = soul_pos
        box = BATTLE_BOX
        margin = CORNER_ESCAPE_MARGIN + (
            SPIKE_CORNER_MARGIN_BONUS if SPIKES_AT_CORNERS else 0
        )
        near_left = (sx - box["x0"]) < margin
        near_right = (box["x1"] - sx) < margin
        near_top = (sy - box["y0"]) < margin
        near_bottom = (box["y1"] - sy) < margin

        if not ((near_left or near_right) and (near_top or near_bottom)):
            return None  # not near a corner - nothing to escape

        # Step along whichever axis is currently more cramped.
        x_room = min(sx - box["x0"], box["x1"] - sx)
        y_room = min(sy - box["y0"], box["y1"] - sy)
        if x_room <= y_room:
            return KEY_RIGHT if near_left else KEY_LEFT
        else:
            return KEY_DOWN if near_top else KEY_UP

    def _choose_dodge_key(
        self, soul_pos: tuple[int, int], white_centroid: tuple[float, float, int]
    ) -> str:
        """
        Pick the movement key that steers away from the white centroid,
        blended with a push away from any nearby wall so fleeing a bullet
        doesn't just walk the SOUL into a corner instead.
        """
        sx, sy = soul_pos
        wx, wy, _ = white_centroid
        dx, dy = sx - wx, sy - wy  # vector pointing away from the danger

        push_x, push_y = self._wall_push(soul_pos)
        dx += push_x
        dy += push_y

        # Move along whichever axis has the larger displacement, since we
        # can only hold one direction key at a time.
        if abs(dx) > abs(dy):
            return KEY_RIGHT if dx > 0 else KEY_LEFT
        else:
            return KEY_DOWN if dy > 0 else KEY_UP

    def _hold_key(self, key: Optional[str]):
        if key == self._held_key:
            return
        if self._held_key is not None:
            key_up(self._held_key)
        if key is not None:
            key_down(key)
        self._held_key = key

    def _release_all_keys(self):
        if self._held_key is not None:
            try:
                key_up(self._held_key)
            except Exception:
                pass
            self._held_key = None

    # -- main loop --------------------------------------------------------

    def run(self):
        print("Heuristic dodger running. Press Ctrl+C to stop.")
        was_focused = True
        try:
            while True:
                loop_start = time.perf_counter()

                if not self._is_game_focused():
                    if was_focused:
                        print("[HeuristicDodger] game window lost focus/minimized - pausing.")
                        was_focused = False
                    self._hold_key(None)  # don't leave a key stuck held while away
                    time.sleep(FOCUS_POLL_INTERVAL_SECONDS)
                    continue
                if not was_focused:
                    print("[HeuristicDodger] game window focused again - resuming.")
                    was_focused = True

                frame = self._grab_frame_bgr()
                soul_pos = self._find_soul(frame)

                if soul_pos is not None:
                    white = self._find_nearby_white_centroid(frame, soul_pos)
                    if white is not None:
                        dodge_key = self._choose_dodge_key(soul_pos, white)
                        self._hold_key(dodge_key)
                    else:
                        # Nothing dangerous nearby. Usually that means stay
                        # put - but if we're currently pinned in a corner,
                        # take this safe moment to step back toward the
                        # center so the NEXT bullet has somewhere to dodge to.
                        self._hold_key(self._corner_escape_key(soul_pos))
                else:
                    self._hold_key(None)  # SOUL not visible - don't move blindly

                elapsed = time.perf_counter() - loop_start
                remaining = FRAME_BUDGET_SECONDS - elapsed
                if remaining > 0:
                    time.sleep(remaining)
        except KeyboardInterrupt:
            print("Stopped.")
        finally:
            self._release_all_keys()
            self.sct.close()


def main():
    print("Starting in 5 seconds - switch to the Deltarune window now.")
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    dodger = HeuristicDodger()
    dodger.run()


if __name__ == "__main__":
    main()
