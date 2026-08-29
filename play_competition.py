"""
play_competition.py

Runs either of two interchangeable agents against the live game, both
exposing the same predict(obs, deterministic=...) -> (action, state)
interface:

  --agent trained     the actual trained PPO model (see train.py/train_jevil.py)
  --agent heuristic   heuristic_llm_agent.py - zero training time, dodging via
                      heuristic_dodge.py's white-pixel-avoidance logic, menu
                      decisions via Groq (groq_menu_advisor.py)

Both are driven through the exact same env.step() / play loop below, so
this is a fair side-by-side: same environment, same action space, same
observation, only the decision-maker differs.

Runs episodes indefinitely - press Ctrl+C to stop.

USAGE
-----
    python play_competition.py --agent trained deltarune_ppo_final
    python play_competition.py --agent heuristic
"""

import argparse
import itertools
import time

from deltarune_env import DeltaruneEnv

COUNTDOWN_SECONDS = 5
DEFAULT_TRAINED_MODEL_PATH = "deltarune_ppo_final"


def countdown(seconds: int):
    print(f"Starting in {seconds} seconds - switch to the Deltarune window now.")
    for i in range(seconds, 0, -1):
        print(f"  {i}...")
        time.sleep(1)


def load_agent(agent_type: str, model_path: str):
    if agent_type == "trained":
        from stable_baselines3 import PPO
        print(f"Loading trained model from: {model_path}")
        return PPO.load(model_path)
    elif agent_type == "heuristic":
        from heuristic_llm_agent import HeuristicLLMAgent
        print("Using HeuristicLLMAgent - no training required, ready immediately.")
        return HeuristicLLMAgent()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent", choices=["trained", "heuristic"], required=True,
        help="Which agent to run.",
    )
    parser.add_argument(
        "model_path", nargs="?", default=DEFAULT_TRAINED_MODEL_PATH,
        help="Checkpoint path (only used with --agent trained).",
    )
    args = parser.parse_args()

    agent = load_agent(args.agent, args.model_path)
    env = DeltaruneEnv(max_episode_steps=1200)
    countdown(COUNTDOWN_SECONDS)

    try:
        for episode in itertools.count(1):
            obs, _ = env.reset()
            done = truncated = False
            episode_reward = 0.0

            while not (done or truncated):
                # Same call signature for both agent types - deterministic=True
                # means "no exploration sampling," which only matters for the
                # trained model; the heuristic agent is deterministic-by-rule
                # regardless (same inputs -> same computed action).
                action, _ = agent.predict(obs, deterministic=True)
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
