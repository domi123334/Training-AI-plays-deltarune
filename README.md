# Deltarune RL Agent (Screen-Capture, No Cheat Engine)

An RL starter project that trains a PPO agent to survive Deltarune's
battle/dodge segments, using screen capture + simulated key presses instead
of reading game memory. This avoids any need for Cheat Engine or manual
address-hunting, at the cost of a noisier, heuristic reward signal.

## Why this scope (battles, not the whole chapter)

Chapter 1's overworld, dialogue, and puzzles don't have a meaningful
"learning" problem for RL — they're better handled with scripted input
sequences. The bullet-hell dodging in battles is the part actually worth
training an agent on: dense feedback (survive/get hit), short episodes,
clear win condition. Scope your training sessions to a single battle wave
at a time.

## Setup

```bash
pip install -r requirements.txt
```

You'll also need Deltarune installed and running yourself — this project
doesn't include or distribute any game files, only automation code that
watches your screen and sends key presses to whatever window you point it at.

## Calibration (required before training)

The easiest way is the included helper script:

```bash
sudo apt install xdotool   # if not already installed
python3 calibrate_capture_region.py --list
```

This prints every open window's title. Find the Deltarune one (it may not
be exactly "DELTARUNE" - check the actual title in the list), then run:

```bash
python3 calibrate_capture_region.py --title "DELTARUNE"
```

This prints a `CAPTURE_REGION = {...}` line you can paste directly into
`deltarune_env.py` (and `heuristic_dodge.py` if you're using that script),
and saves `capture_region_preview.png` so you can visually confirm the
region is tightly framing the game window and not clipping part of it or
including your desktop/taskbar.

If that doesn't work (e.g. you're not on X11/XWayland, or `xdotool` isn't
available), you can still set `CAPTURE_REGION` by hand:

1. Open `deltarune_env.py` and find `CAPTURE_REGION` near the top.
2. Launch Deltarune windowed (not fullscreen) so you can read off its
   position/size, and set:
   ```python
   CAPTURE_REGION = {"left": <x>, "top": <y>, "width": <w>, "height": <h>}
   ```
3. Confirm the movement keys in `linux_input.py` (`KEY_MAP`) match your
   in-game bindings — these default to WASD plus Z/X/Space.

## Running training

```bash
python train.py
```

Get into the battle you want to train on *before* the countdown finishes —
the script gives you 5 seconds to alt-tab into the game window, then starts
sending inputs and reading rewards.

Training progress is logged to `./tb_logs` (view with `tensorboard --logdir tb_logs`),
and checkpoints save periodically to `./checkpoints`.

## The hard part: reward signal

Without memory access, we can't read HP or battle-state directly, so
`RewardEstimator` in `deltarune_env.py` uses pixel heuristics instead:

- Small reward per frame survived
- Penalty when a large brightness "flash" is detected (approximates taking
  a hit)
- Episode-end penalty when the screen goes mostly black for a sustained
  streak (approximates a game-over screen)

**This is the part you should expect to tune.** Pixel heuristics are
fragile and will misfire on some fights (e.g. fights with their own
flashing visual effects unrelated to damage). Two upgrade paths if the
heuristic reward isn't reliable enough:

1. **OCR-based reward**: use `pytesseract` to read the HP number directly
   off the screen each frame, and compute reward from HP deltas. More
   reliable than brightness heuristics, still no memory reading required.
2. **Memory reading**: since Deltarune is a GameMaker game, tools like
   `pymem` can read variables (HP, SOUL position, bullet list) directly
   from process memory. This gives clean, low-dimensional observations
   and reward, and trains much faster than pixels — worth doing later even
   without manually using Cheat Engine, by scripting a memory scan instead
   of doing it by hand.

## Why training will feel slow at first

This environment runs in *real time* against a live game window — there's
no way to fast-forward simulation like you could with a custom Gym env.
To keep this tractable:

- Keep `max_episode_steps` small (one dodge wave, not a whole fight)
- Consider slowing down or looping a single tricky pattern rather than
  running full battles end-to-end
- Expect tens of thousands of steps (a few hours of real playtime) before
  you see meaningfully better dodging, since this is pixel-based PPO, not
  memory-based

## Learning to use the fight menu (FIGHT/ACT/ITEM/SPARE/DEFEND)

Without memory access, the agent can't be told directly which menu option
means what — it only sees pixels, same as you would. Two pieces make this
work:

1. **Turn-progress reward.** The SOUL sits at a known row while the
   action-select menu is open, then leaves it once you confirm a choice.
   `deltarune_env.py` detects that transition and rewards it — this is what
   actually teaches the agent that the menu exists and using it matters,
   rather than only ever dodging in place. Calibrate `MENU_SOUL_Y` the same
   way you calibrated `CAPTURE_REGION`: pause on the action-select screen
   and note the SOUL's y pixel position.

2. **Optional OCR logging (not part of the reward).** `MenuActionReader`
   can read the text label the SOUL is hovering over via `pytesseract`,
   purely so *you* can see what it's picking during training (printed as
   `[turn N] action confirmed (read as: FIGHT)` etc. by `train.py`). Pixel
   font OCR isn't perfectly reliable, so this is for visibility/debugging,
   not ground truth the agent is trained against. It's optional — training
   works without it, you just won't get the label. Requires
   `pip install pytesseract` and the `tesseract-ocr` system package.

Note that the menu layout differs by character — Ralsei gets ACT/DEFEND/
ITEM/SPARE (no FIGHT) — so `MENU_VERB_X_CENTERS` in `deltarune_env.py` may
need adjusting depending on whose turn you're training on.

## Why movement can look random early in training (and how to see calculated play)

PPO deliberately samples actions from a probability distribution during
training rather than always taking its single best guess - that's how it
explores and discovers what works at all. With a freshly-initialized (i.e.
random) network, this looks like frantic, random key-pressing. That's
expected, not a bug: without exploration, the agent would just repeat its
initial random behavior forever with no way to improve.

Three things in `deltarune_env.py` also help reduce frantic-looking play
regardless of training stage:
- **FRAME_SKIP** (default 4): the agent makes one decision, which then holds
  for several real game frames, instead of re-deciding every single frame.
- **MOVEMENT_SWITCH_PENALTY**: a small cost each time it changes direction
  from its last decision, so holding a direction is cheaper than flip-flopping.
- **BUTTON_COOLDOWN_DECISIONS**: after tapping confirm/cancel/skip, it can't
  tap another button for a few decisions, preventing menu-mash spam.

To actually see what the agent has learned (not exploration noise), use
`play.py` instead of watching training directly - it loads a saved
checkpoint and always picks the model's single best-known action
(`deterministic=True`), which is the real, calculated behavior once the
model has had enough training to learn something.

## Groq episode-end coaching (optional)

`groq_teacher.py` reviews how each training episode went (reward, turns
completed, whether it died via a SOUL break or timed out) and prints back a
short coaching tip via Groq's API. This is **advisory only** — it doesn't
modify the rewards PPO already learned from for that episode; there's no
practical way to feed a slow API call back into a real-time 30 FPS training
loop as an actual gradient signal. Think of it as a coach watching from the
sidelines and telling you what it noticed, not something rewriting the RL
math.

Setup:
```bash
pip install groq
export GROQ_API_KEY="your-key-here"
```

Training works fine without this — if the key isn't set or `groq` isn't
installed, it silently disables itself. Toggle it off entirely with
`ENABLE_GROQ_TEACHER = False` in `train.py` / `train_jevil.py`.

## Files

- `deltarune_env.py` — Gymnasium environment (screen capture + key injection + reward heuristic)
- `linux_input.py` — key press/release via pynput (X11 XTest)
- `train.py` — PPO training script (Stable-Baselines3)
- `train_jevil.py` — training config tuned for the Jevil superboss fight (longer episodes, larger training budget, resumable across sessions)
- `groq_teacher.py` — optional episode-end coaching via Groq's API (advisory only, see below)
- `play.py` — watch a trained checkpoint play using its learned policy (deterministic, no exploration noise) — use this to see actual learned behavior rather than training-time randomness
- `heuristic_dodge.py` — standalone, non-RL script: finds the SOUL by color and holds a movement key away from nearby white pixels. No training involved — a quick reactive baseline, separate from the RL project.
- `calibrate_capture_region.py` — finds your Deltarune window's screen coordinates and saves a preview screenshot to confirm it, so you're not guessing `CAPTURE_REGION` by hand
- `requirements.txt` — Python dependencies

## A note on Wayland

Key injection here uses X11's XTest extension (via `pynput`), which works
for games running under X11 or XWayland — this covers most Linux gaming,
including Steam Proton and native titles. If you're on a pure-Wayland
compositor and key presses aren't reaching the game, check
`echo $XDG_SESSION_TYPE` and confirm the game is actually running under
XWayland (it usually is by default even in a Wayland session).
