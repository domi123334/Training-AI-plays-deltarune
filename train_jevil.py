"""
train_jevil.py

Reproducible PPO training configuration for Jevil - not a pretrained model,
since that can only come from actually running training against your live
game session. Training uses seeded PPO exploration; playback is
deterministic through play.py.

WHY THIS IS SEPARATE FROM train.py
-----------------------------------
Jevil's fight is meaningfully different from a regular Chapter 1 encounter:
faster, more erratic patterns (including a "chaos" attack that ignores
normal movement-blocking rules), multiple phases, and it's simply a longer
fight. This script uses a larger training budget and longer episode length
to match, but doesn't hardcode any Jevil-specific detection - the
environment still perceives everything through the same generic SOUL-break
and turn-progress signals as any other fight (see deltarune_env.py).

USAGE
-----
1. Launch Deltarune, get into the Jevil fight (Chapter 1 secret bonus area).
2. Calibrate CAPTURE_REGION and MENU_SOUL_Y in deltarune_env.py if you
   haven't already (see calibrate_capture_region.py for CAPTURE_REGION).
3. Run:  python train_jevil.py
4. Switch focus to the Deltarune window during the countdown.

This will likely take multiple sessions given how varied Jevil's attacks
are - set RESUME_FROM to a saved checkpoint path to continue an existing
run instead of starting over.

Once trained, watch it play with:  python play.py jevil_ppo_final
"""

import os
import random
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from deltarune_env import DeltaruneEnv
from groq_teacher import EpisodeTeacher

SEED = 12345
COUNTDOWN_SECONDS = 5
TOTAL_TIMESTEPS = 200_000
MAX_EPISODE_STEPS = 1_200
CHECKPOINT_DIR = "./checkpoints_jevil"
CHECKPOINT_EVERY = 5_000
FINAL_MODEL_PATH = "jevil_ppo_final"

# Set to False to skip Groq coaching entirely even if GROQ_API_KEY is set.
ENABLE_GROQ_TEACHER = True

# Set this to a checkpoint path (e.g. "checkpoints_jevil/jevil_ppo_50000_steps.zip")
# to continue training an existing run instead of starting from scratch.
RESUME_FROM = None


class TurnProgressLogger(BaseCallback):
    """Prints a line whenever the env reports a newly-completed turn."""

    def __init__(self):
        super().__init__()
        self._last_turn_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        dones = self.locals.get("dones", [False])
        info = infos[0]
        turn_count = info.get("turn_count", 0)

        if turn_count > self._last_turn_count:
            print(f"[turn {turn_count}] action confirmed")
        self._last_turn_count = turn_count

        if dones[0]:
            self._last_turn_count = 0
        return True


class GroqTeacherCallback(BaseCallback):
    """
    At the end of each episode, sends a summary of how it went to Groq and
    prints back a short coaching tip. Advisory/logged only - see
    groq_teacher.py for the scope note on why this doesn't feed back into
    PPO's actual training signal.
    """

    def __init__(self, teacher: EpisodeTeacher):
        super().__init__()
        self.teacher = teacher
        self._episode_reward = 0.0
        self._episode_steps = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [0.0])
        dones = self.locals.get("dones", [False])
        infos = self.locals.get("infos", [{}])

        self._episode_reward += rewards[0]
        self._episode_steps += 1

        if dones[0]:
            info = infos[0]
            died = "error" not in info and rewards[0] <= -5.0
            stats = {
                "total_reward": self._episode_reward,
                "turn_count": info.get("turn_count", 0),
                "steps": self._episode_steps,
                "died": died,
            }
            tip = self.teacher.review_episode(stats)
            if tip:
                print(f"[Teacher] {tip}")

            self._episode_reward = 0.0
            self._episode_steps = 0

        return True


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def countdown(seconds: int):
    print(f"Starting in {seconds} seconds - switch to the Deltarune window now.")
    print("Make sure you're actually in the Jevil fight, not just Chapter 1 generally.")
    for i in range(seconds, 0, -1):
        print(f"  {i}...")
        time.sleep(1)


def main():
    seed_everything()
    env = DeltaruneEnv(max_episode_steps=MAX_EPISODE_STEPS)

    countdown(COUNTDOWN_SECONDS)

    # Reset happens after the countdown, not before - resetting immediately
    # on env creation would grab a screen capture before you've had a
    # chance to switch focus to the game window.
    env.reset(seed=SEED)

    if RESUME_FROM and os.path.exists(RESUME_FROM):
        print(f"Resuming training from: {RESUME_FROM}")
        model = PPO.load(RESUME_FROM, env=env, seed=SEED)
    else:
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            n_steps=256,
            batch_size=64,
            learning_rate=3e-4,
            seed=SEED,
            device="auto",
            tensorboard_log="./tb_logs_jevil",
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_EVERY,
        save_path=CHECKPOINT_DIR,
        name_prefix="jevil_ppo",
    )
    turn_logger = TurnProgressLogger()
    callbacks = [checkpoint_callback, turn_logger]

    if ENABLE_GROQ_TEACHER:
        teacher = EpisodeTeacher()
        callbacks.append(GroqTeacherCallback(teacher))

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            reset_num_timesteps=(RESUME_FROM is None),
        )
    except KeyboardInterrupt:
        print("Training interrupted - saving current model.")
    finally:
        model.save(FINAL_MODEL_PATH)
        print(f"Saved to {FINAL_MODEL_PATH}.zip - resume later by setting RESUME_FROM to this path.")
        env.close()


if __name__ == "__main__":
    main()
