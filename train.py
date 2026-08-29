"""
train.py

Trains a PPO agent to survive a Deltarune battle/dodge segment using the
screen-capture environment defined in deltarune_env.py.

USAGE
-----
1. Launch Deltarune, get into the battle you want the agent to learn
   (e.g. hold at the start of a bullet-hell wave, menu already dismissed
   so the SOUL is visible and moveable).
2. Calibrate CAPTURE_REGION in deltarune_env.py to match your window.
3. Run:  python train.py
4. Switch focus to the Deltarune window immediately - you have a few
   seconds (see COUNTDOWN_SECONDS) before actions start firing.

Because this environment runs in real time against a live game, an episode
takes as long as it takes in-game. Keep max_episode_steps modest (see
deltarune_env.DeltaruneEnv) so episodes are short and PPO gets many episodes
per hour rather than a few very long ones.
"""

import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from deltarune_env import DeltaruneEnv
from groq_teacher import EpisodeTeacher

COUNTDOWN_SECONDS = 5
TOTAL_TIMESTEPS = 50_000
CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_EVERY = 2_000

# Set to False to skip Groq coaching entirely even if GROQ_API_KEY is set.
ENABLE_GROQ_TEACHER = True


class TurnProgressLogger(BaseCallback):
    """
    Prints a line whenever the env reports a newly-completed turn, so you
    can actually see the turn-progress reward (see RewardEstimator in
    deltarune_env.py) firing during training instead of it being invisible.
    """

    def __init__(self):
        super().__init__()
        self._last_turn_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        dones = self.locals.get("dones", [False])
        info = infos[0]
        turn_count = info.get("turn_count", 0)

        if turn_count > self._last_turn_count:
            label = info.get("last_action_label")
            label_str = f" (read as: {label})" if label else ""
            print(f"[turn {turn_count}] action confirmed{label_str}")
        self._last_turn_count = turn_count

        if dones[0]:
            self._last_turn_count = 0  # turn_count resets with each new episode
        return True


class GroqTeacherCallback(BaseCallback):
    """
    At the end of each episode, sends a summary of how it went to Groq and
    prints back a short coaching tip. See groq_teacher.py for the important
    scope note: this is advisory/logged, not a training signal - it doesn't
    modify the rewards PPO already learned from for that episode.
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
            died = "error" not in info and info.get("turn_count") is not None and rewards[0] <= -5.0
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


def countdown(seconds: int):
    print(f"Starting in {seconds} seconds - switch to the Deltarune window now.")
    for i in range(seconds, 0, -1):
        print(f"  {i}...")
        time.sleep(1)


def main():
    env = DeltaruneEnv(max_episode_steps=600)

    # Sanity-check the environment implements the Gym API correctly.
    # Comment this out once you've confirmed it passes - it resets/steps
    # the env a few times, which means live input during the check.
    # check_env(env, warn=True)

    countdown(COUNTDOWN_SECONDS)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=256,       # shorter rollout buffer since episodes are short
        batch_size=64,
        learning_rate=3e-4,
        tensorboard_log="./tb_logs",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_EVERY,
        save_path=CHECKPOINT_DIR,
        name_prefix="deltarune_ppo",
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
        )
    except KeyboardInterrupt:
        print("Training interrupted - saving current model.")
    finally:
        model.save("deltarune_ppo_final")
        env.close()


if __name__ == "__main__":
    main()
