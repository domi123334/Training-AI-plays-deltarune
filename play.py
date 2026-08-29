"""
play.py

Runs a TRAINED model against the live game using its learned policy, not
exploration noise. This is the script to use if you want to see "does the
agent actually know what it's doing" rather than watching training itself.

WHY THIS IS SEPARATE FROM train.py
-----------------------------------
During training, PPO deliberately samples actions from a probability
distribution rather than always picking its single best guess - that
randomness is how it discovers what works at all. Early in training, with
mostly-random initial weights, this looks like frantic/random movement.
That's expected, not a bug: without exploration, a randomly-initialized
policy would just repeat the same bad behavior forever with no way to
improve.

This script does the opposite: it loads a saved checkpoint and calls
model.predict(obs, deterministic=True), which always picks the single
highest-probability action instead of sampling. Once the model has actually
learned something, this is what shows you its real, calculated behavior -
no exploration randomness involved.

USAGE
-----
    python play.py                          # uses deltarune_ppo_final.zip
    python play.py checkpoints/deltarune_ppo_20000_steps.zip

Runs episodes indefinitely - press Ctrl+C to stop.

Run this against a model that's had a meaningful amount of training - a
model saved after only a few hundred steps will still look nearly random
simply because it hasn't learned anything yet. That's a training-amount
problem, not something this script (or determinism) can fix.
"""

import itertools
import sys
import time

from stable_baselines3 import PPO

from deltarune_env import DeltaruneEnv

COUNTDOWN_SECONDS = 5
DEFAULT_MODEL_PATH = "deltarune_ppo_final"


def countdown(seconds: int):
    print(f"Starting in {seconds} seconds - switch to the Deltarune window now.")
    for i in range(seconds, 0, -1):
        print(f"  {i}...")
        time.sleep(1)


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    env = DeltaruneEnv(max_episode_steps=600)
    countdown(COUNTDOWN_SECONDS)

    try:
        for episode in itertools.count(1):
            obs, _ = env.reset()
            done = truncated = False
            episode_reward = 0.0

            while not (done or truncated):
                # deterministic=True: always the model's single best-known
                # action, no exploration sampling. This is the "calculated,
                # not random" behavior a trained policy should show.
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward

            print(
                f"Episode {episode} finished - "
                f"reward: {episode_reward:.2f}, turns completed: {info.get('turn_count', 0)}"
            )
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
