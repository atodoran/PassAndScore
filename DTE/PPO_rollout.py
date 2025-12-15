#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
from pathlib import Path as _Path

# -------------------------------------------------------------
# Import env + PPO with the same sys.path hack as training
# -------------------------------------------------------------
ROOT = _Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import PassAndScoreEnv
from PPO import PPOAgent, ShapingConfig, RewardShaper

# -------------------------------------------------------------
# Shared helpers (paths + shaping)
# -------------------------------------------------------------

RUNS_ROOT = Path("models")


def shaping_from_args(player_id: str, args) -> ShapingConfig:
    if player_id == "A":
        return ShapingConfig(
            r_first_touch=args.A_first_touch_bonus,
            r_bump_penalty=args.A_bump_penalty,
            r_pass=args.A_pass_bonus,
            w_player_to_ball=args.A_player_to_ball_weight,
            w_ball_to_target=args.A_target_weight,
            target_type=args.A_target_kind,
        )
    else:
        return ShapingConfig(
            r_first_touch=args.B_first_touch_bonus,
            r_bump_penalty=args.B_bump_penalty,
            r_pass=args.B_pass_bonus,
            w_player_to_ball=args.B_player_to_ball_weight,
            w_ball_to_target=args.B_target_weight,
            target_type=args.B_target_kind,
        )


def get_policy_source_path(player: str, args) -> Path | None:
    """
    Mirror training semantics:

    - If --resume RUN_NAME is given:
        load models/RUN_NAME/{A,B}_policy_last.pt
      (latest checkpoint of that run)

    - Else, if --A-policy / --B-policy RUN_NAME is given:
        load models/RUN_NAME/{A,B}_policy_best.pt
      (best checkpoint of that run, by raw reward)

    - Else: return None (fresh policy will be created).
    """
    if args.resume is not None:
        # resume-from-run → use LAST policy
        return RUNS_ROOT / args.resume / f"{player}_policy_last.pt"

    policy_arg = getattr(args, f"{player}_policy")
    if policy_arg is not None:
        # explicit init run → use BEST policy
        return RUNS_ROOT / policy_arg / f"{player}_policy_best.pt"

    return None


def make_agent(player: str, obs_dim: int, act_dim: int, args, device: str) -> PPOAgent:
    hidden_sizes = (args.hidden_size, args.hidden_size)
    src_path = get_policy_source_path(player, args)

    if src_path is not None and src_path.exists():
        print(f"Loading {player} policy from {src_path}")
        return PPOAgent.load(str(src_path), device=device)
    else:
        if src_path is not None:
            print(f"Warning: {player} policy not found at {src_path}, using fresh policy.")
        return PPOAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            lr=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            n_epochs=1,           # not training, but PPOAgent expects something
            batch_size=64,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device=device,
        )


# -------------------------------------------------------------
# Live rollout with env + reward plot
# -------------------------------------------------------------

def run_rollout(args):
    # device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # env: decentralized obs for both players
    env = PassAndScoreEnv(
        centralized=False,
        ball_spawn_region=args.ball_spawn_region,
        max_steps=args.max_steps,
    )

    # initial reset for shapes/dims
    obs_tuple, info = env.reset()
    obs_A, obs_B = obs_tuple
    obs_A = np.asarray(obs_A, dtype=np.float32)
    obs_B = np.asarray(obs_B, dtype=np.float32)

    obs_dim = int(obs_A.shape[0])
    act_dim = 5

    cfg_A = shaping_from_args("A", args)
    cfg_B = shaping_from_args("B", args)

    shaper_A = RewardShaper(env, "A", args.gamma, cfg_A)
    shaper_B = RewardShaper(env, "B", args.gamma, cfg_B)

    agent_A = make_agent("A", obs_dim, act_dim, args, device)
    agent_B = make_agent("B", obs_dim, act_dim, args, device)

    # Matplotlib setup
    plt.ion()
    fig, (ax_env, ax_reward) = plt.subplots(2, 1, figsize=(7, 9))
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    step_indices = []
    shaped_rewards = []   # per-step avg shaped reward over A/B

    done = False
    step = 0

    while not done and step < args.max_steps:
        step += 1

        act_A, _, _ = agent_A.act(obs_A, det=args.deterministic)
        act_B, _, _ = agent_B.act(obs_B, det=args.deterministic)
        joint_action = np.array([act_A, act_B], dtype=np.int64)

        next_obs_tuple, env_rew, terminated, truncated, info = env.step(joint_action)
        done = bool(terminated or truncated)

        before = info.get("before", {})
        after = info.get("after", {})

        r_A = shaper_A.shape(env_rew, before, after)
        r_B = shaper_B.shape(env_rew, before, after)
        r_mean = 0.5 * (r_A + r_B)

        step_indices.append(step)
        shaped_rewards.append(r_mean)

        # draw env
        env.render(ax=ax_env)

        # draw reward trace
        ax_reward.clear()
        ax_reward.plot(step_indices, shaped_rewards, marker="o")
        ax_reward.axhline(0.0, linestyle="--", linewidth=1)
        ax_reward.set_xlabel("Step")
        ax_reward.set_ylabel("Shaped reward (avg A/B)")
        ax_reward.set_title("Per-step shaped reward")

        y = np.array(shaped_rewards, dtype=np.float32)
        y_min = float(y.min())
        y_max = float(y.max())
        if y_max - y_min < 1e-6:
            margin = 0.5
        else:
            margin = 0.2 * (y_max - y_min)
        ax_reward.set_ylim(y_min - margin, y_max + margin)

        fig.canvas.draw()
        plt.pause(args.pause)

        if not done:
            obs_A, obs_B = next_obs_tuple
            obs_A = np.asarray(obs_A, dtype=np.float32)
            obs_B = np.asarray(obs_B, dtype=np.float32)

    print(f"Episode finished at step {step} (terminated={terminated}, truncated={truncated})")
    plt.ioff()
    plt.show()
    env.close()


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    # basic rollout
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument(
        "--ball-spawn-region",
        type=str,
        default="A",
        choices=["A", "B", "A-pass"],
        help='Where the ball starts: "A", "B", or "A-pass" (incoming pass from A toward B).',
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.05,
        help="Pause in seconds between frames (controls playback speed).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions (argmax) instead of sampling.",
    )

    # device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )

    # PPO net params (only used if we need to create fresh policies)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)

    # checkpoint loading logic
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Run name under models/ to load LAST A/B policies from "
             "(mirrors training --resume).",
    )
    parser.add_argument(
        "--A-policy",
        type=str,
        default=None,
        help="Run name under models/ whose BEST A_policy is used as initialization for A.",
    )
    parser.add_argument(
        "--B-policy",
        type=str,
        default=None,
        help="Run name under models/ whose BEST B_policy is used as initialization for B.",
    )

    # shaping args for A
    parser.add_argument("--A-first-touch-bonus", type=float, default=0.0)
    parser.add_argument("--A-bump-penalty", type=float, default=0.0)
    parser.add_argument("--A-pass-bonus", type=float, default=0.0)
    parser.add_argument("--A-player-to-ball-weight", type=float, default=0.0)
    parser.add_argument(
        "--A-target-kind",
        type=str,
        default="none",
        choices=["none", "goal", "regionB", "playerB"],
    )
    parser.add_argument("--A-target-weight", type=float, default=0.0)

    # shaping args for B
    parser.add_argument("--B-first-touch-bonus", type=float, default=0.0)
    parser.add_argument("--B-bump-penalty", type=float, default=0.0)
    parser.add_argument("--B-pass-bonus", type=float, default=0.0)
    parser.add_argument("--B-player-to-ball-weight", type=float, default=0.0)
    parser.add_argument(
        "--B-target-kind",
        type=str,
        default="none",
        choices=["none", "goal", "regionB", "playerB"],
    )
    parser.add_argument("--B-target-weight", type=float, default=0.0)

    args = parser.parse_args()
    run_rollout(args)


if __name__ == "__main__":
    main()
