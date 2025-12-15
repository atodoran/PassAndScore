#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from env import PassAndScoreEnv


def run_one_episode_stats(
    model,
    seed: int,
    max_steps: int = 400,
    deterministic: bool = False,
):
    """
    Run a single episode without rendering and collect statistics:
      - did we score?
      - steps until score
      - time until score
      - number of touches until score (from env info)
      - env return
    """
    env = PassAndScoreEnv(centralized=True, ball_spawn_region="A", seed=seed)
    obs, info = env.reset()

    ret_env = 0.0
    scored = False
    first_score_step = None
    steps = 0
    touches = int(info.get("touches", 0))

    for t in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        ret_env += float(reward)
        steps = t + 1

        info_after = info.get("after", {})
        # cumulative touches from env
        touches = int(info_after.get("touches", touches))

        # goal detection
        if info_after.get("scored_goal", False) and not scored:
            scored = True
            first_score_step = steps

        done = terminated or truncated
        if done:
            break

    env.close()

    # Time to score in seconds if scored
    time_to_score = None
    if scored and first_score_step is not None:
        time_to_score = first_score_step

    return {
        "scored": scored,
        "steps": steps,
        "time_to_score": time_to_score,
        "touches_to_score": touches if scored else None,
        "env_return": ret_env,
    }


def mean_std(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return None, None
    if x.size == 1:
        return float(x[0]), 0.0
    return float(x.mean()), float(x.std(ddof=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/PPO/policy",
        help="Path to SB3 model (without .zip or with .zip)",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed")

    args = parser.parse_args()

    base_path = Path(args.model_path)
    if base_path.suffix != ".zip":
        model_file = base_path.with_suffix(".zip")
    else:
        model_file = base_path

    if not model_file.exists():
        raise FileNotFoundError(f"Could not find model file: {model_file}")

    model = PPO.load(str(model_file))

    rng = np.random.default_rng(args.seed)

    scored_flags = []
    times_to_score = []
    touches_to_score = []
    env_returns = []

    print(f"Running {args.episodes} evaluation episodes...")

    for ep in range(args.episodes):
        seed = int(rng.integers(0, 2**31 - 1))
        stats = run_one_episode_stats(
            model,
            seed=seed,
            max_steps=args.max_steps,
            deterministic=args.deterministic,
        )

        scored_flags.append(stats["scored"])
        env_returns.append(stats["env_return"])

        if stats["scored"]:
            times_to_score.append(stats["time_to_score"])
            touches_to_score.append(stats["touches_to_score"])

        print(
            f"Episode {ep+1:3d}/{args.episodes} | seed={seed} | "
            f"scored={stats['scored']} | "
            f"steps={stats['steps']} | "
            f"env_return={stats['env_return']:.3f} | "
            f"touches={stats['touches_to_score'] if stats['scored'] else 'N/A'}"
        )

    scored_flags = np.array(scored_flags, dtype=bool)
    env_returns = np.array(env_returns, dtype=float)

    num_episodes = args.episodes
    num_scored = int(scored_flags.sum())
    score_rate = num_scored / float(num_episodes)

    print("\n=== Evaluation summary ===")
    print(f"Episodes:                {num_episodes}")
    print(f"Scored episodes:         {num_scored}")
    print(f"Score rate:              {score_rate * 100.0:.2f}%")

    mean_env, std_env = mean_std(env_returns)
    print(f"Env return:              mean={mean_env:.3f}, std={std_env:.3f}")

    if num_scored > 0:
        times_to_score = np.array(times_to_score, dtype=float)
        touches_to_score = np.array(touches_to_score, dtype=float)

        mean_t, std_t = mean_std(times_to_score)
        mean_touch, std_touch = mean_std(touches_to_score)

        print("\n(Conditioned on scoring episodes only)")
        print(f"Time to score [s]:       mean={mean_t:.3f}, std={std_t:.3f}")
        print(f"Touches until score:     mean={mean_touch:.3f}, std={std_touch:.3f}")
    else:
        print("\nNo goals were scored; time/touches to score are undefined.")


if __name__ == "__main__":
    main()
