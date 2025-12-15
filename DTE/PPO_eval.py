#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
import tqdm

import sys
from pathlib import Path as _Path

# ---------------------------------------------------------------------
# Import env + PPO with same sys.path hack as training/rollout
# ---------------------------------------------------------------------
ROOT = _Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import PassAndScoreEnv
from PPO import PPOAgent

RUNS_ROOT = Path("models")


# ---------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------


def get_policy_source_path(player: str, args) -> Path | None:
    """
    Mirror training/rollout semantics:

    - If --resume RUN_NAME is given:
        load models/RUN_NAME/{A,B}_policy_last.pt
      (latest checkpoint of that run)

    - Else, if --A-policy / --B-policy RUN_NAME is given:
        load models/RUN_NAME/{A,B}_policy_best.pt
      (best checkpoint of that run, by raw reward)

    - Else: return None (player acts randomly in eval).
    """
    if args.resume is not None:
        # resume-from-run → use LAST policy
        return RUNS_ROOT / args.resume / f"{player}_policy_last.pt"

    policy_arg = getattr(args, f"{player}_policy")
    if policy_arg is not None:
        # explicit init run → use BEST policy
        return RUNS_ROOT / policy_arg / f"{player}_policy_best.pt"

    return None


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


def evaluate(args):
    # device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    players = args.players  # "A", "B", or "both"
    assert players in ("A", "B", "both")

    # env: decentralized obs (oA, oB)
    env = PassAndScoreEnv(
        centralized=False,
        ball_spawn_region=args.ball_spawn_region,
        max_steps=args.max_steps_per_episode,
    )

    # initial reset just to get obs dims for sanity (we don't need them for loading)
    obs_tuple, info = env.reset()
    obs_A, obs_B = obs_tuple
    obs_A = np.asarray(obs_A, dtype=np.float32)
    obs_B = np.asarray(obs_B, dtype=np.float32)

    # load models as needed
    agent_A = None
    agent_B = None

    if players in ("A", "both"):
        src_A = get_policy_source_path("A", args)
        if src_A is None:
            raise ValueError(
                "Player A is requested (players includes 'A') but no source run "
                "was provided via --resume or --A-policy."
            )
        if not src_A.exists():
            raise FileNotFoundError(f"Player A model not found at {src_A}")
        agent_A = PPOAgent.load(str(src_A), device=device)
        print(f"Loaded A from {src_A}")

    if players in ("B", "both"):
        src_B = get_policy_source_path("B", args)
        if src_B is None:
            raise ValueError(
                "Player B is requested (players includes 'B') but no source run "
                "was provided via --resume or --B-policy."
            )
        if not src_B.exists():
            raise FileNotFoundError(f"Player B model not found at {src_B}")
        agent_B = PPOAgent.load(str(src_B), device=device)
        print(f"Loaded B from {src_B}")

    rng = np.random.default_rng(args.seed)

    # per-episode stats
    episode_scored = []
    touches_until_score = []
    steps_until_score = []

    for ep in tqdm.tqdm(range(args.episodes), "Evaluating episodes..."):
        obs_tuple, info = env.reset()
        obs_A, obs_B = obs_tuple
        obs_A = np.asarray(obs_A, dtype=np.float32)
        obs_B = np.asarray(obs_B, dtype=np.float32)

        done = False
        ep_steps = 0
        ep_scored = False
        ep_score_steps = None
        ep_score_touches = None

        while not done:
            ep_steps += 1

            # select actions
            if agent_A is not None:
                act_A, _, _ = agent_A.act(obs_A, det=args.deterministic)
            else:
                act_A = int(rng.integers(0, 5))

            if agent_B is not None:
                act_B, _, _ = agent_B.act(obs_B, det=args.deterministic)
            else:
                act_B = int(rng.integers(0, 5))

            joint_action = np.array([act_A, act_B], dtype=np.int64)

            next_obs_tuple, env_rew, terminated, truncated, info = env.step(
                joint_action
            )
            done = bool(terminated or truncated)

            after = info.get("after", {})
            scored_step = bool(after.get("scored_goal", False))

            if scored_step and not ep_scored:
                ep_scored = True
                ep_score_steps = ep_steps
                ep_score_touches = int(after.get("touches", 0))

            if not done:
                obs_A, obs_B = next_obs_tuple
                obs_A = np.asarray(obs_A, dtype=np.float32)
                obs_B = np.asarray(obs_B, dtype=np.float32)

        episode_scored.append(1.0 if ep_scored else 0.0)
        if ep_scored:
            touches_until_score.append(float(ep_score_touches))
            steps_until_score.append(float(ep_score_steps))

    env.close()

    episode_scored = np.array(episode_scored, dtype=np.float32)
    score_rate_mean = float(episode_scored.mean())
    score_rate_std = float(episode_scored.std(ddof=0) / np.sqrt(len(episode_scored)))

    if touches_until_score:
        touches_arr = np.array(touches_until_score, dtype=np.float32)
        touches_mean = float(touches_arr.mean())
        touches_std = float(touches_arr.std(ddof=0) / np.sqrt(len(touches_arr)))
    else:
        touches_mean = float("nan")
        touches_std = float("nan")

    if steps_until_score:
        steps_arr = np.array(steps_until_score, dtype=np.float32)
        steps_mean = float(steps_arr.mean())
        steps_std = float(steps_arr.std(ddof=0) / np.sqrt(len(steps_arr)))
    else:
        steps_mean = float("nan")
        steps_std = float("nan")

    print("\n=== Evaluation results ===")
    print(f"episodes: {args.episodes}")
    print(f"score rate: mean={score_rate_mean:.3f}, std={score_rate_std:.3f}")
    print(
        f"touches until score (scoring episodes only): "
        f"mean={touches_mean:.3f}, std={touches_std:.3f}"
    )
    print(
        f"steps until score (scoring episodes only): "
        f"mean={steps_mean:.3f}, std={steps_std:.3f}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--players",
        type=str,
        default="both",
        choices=["A", "B", "both"],
        help="Which players use learned policies; others act randomly.",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps-per-episode", type=int, default=400)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions (argmax) instead of sampling.",
    )

    parser.add_argument(
        "--ball-spawn-region",
        type=str,
        default="A",
        choices=["A", "B", "A-pass"],
        help='Where the ball starts: "A", "B", or "A-pass" (incoming pass from A toward B).',
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--seed", type=int, default=0)

    # checkpoint selection (mirrors training/rollout)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Run name under models/ to load LAST A/B policies from.",
    )
    parser.add_argument(
        "--A-policy",
        type=str,
        default=None,
        help="Run name under models/ whose BEST A_policy is evaluated for A.",
    )
    parser.add_argument(
        "--B-policy",
        type=str,
        default=None,
        help="Run name under models/ whose BEST B_policy is evaluated for B.",
    )

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
