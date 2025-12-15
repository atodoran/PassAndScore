#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch

import sys
from pathlib import Path as _Path

# Use a non-interactive backend so this works headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import imageio.v2 as imageio

# -------------------------------------------------------------
# Import env + PPO with the same sys.path hack as training
# -------------------------------------------------------------
ROOT = _Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import PassAndScoreEnv
from PPO import PPOAgent

# -------------------------------------------------------------
# Shared helpers (paths)
# -------------------------------------------------------------

RUNS_ROOT = Path("models")


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
# Rollout → video
# -------------------------------------------------------------

def figure_to_array(fig: plt.Figure) -> np.ndarray:
    """Convert a Matplotlib figure to a HxWx3 uint8 RGB array."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    # Get ARGB buffer from the canvas
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf = buf.reshape(h, w, 4)

    # Reorder from ARGB to RGB (drop alpha)
    rgb = buf[:, :, 1:4]
    return rgb


def run_rollout_and_record(args):
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

    agent_A = make_agent("A", obs_dim, act_dim, args, device)
    agent_B = make_agent("B", obs_dim, act_dim, args, device)

    # Offscreen Matplotlib figure for env rendering
    fig, ax_env = plt.subplots(figsize=(7, 7))
    ax_env.set_axis_off()

    frames = []

    done = False
    step = 0

    while not done and step < args.max_steps:
        step += 1

        act_A, _, _ = agent_A.act(obs_A, det=args.deterministic)
        act_B, _, _ = agent_B.act(obs_B, det=args.deterministic)
        joint_action = np.array([act_A, act_B], dtype=np.int64)

        next_obs_tuple, env_rew, terminated, truncated, info = env.step(joint_action)
        done = bool(terminated or truncated)

        # draw env
        ax_env.clear()
        ax_env.set_axis_off()
        env.render(ax=ax_env)

        frame = figure_to_array(fig)
        frames.append(frame)

        if not done:
            obs_A, obs_B = next_obs_tuple
            obs_A = np.asarray(obs_A, dtype=np.float32)
            obs_B = np.asarray(obs_B, dtype=np.float32)

    print(f"Episode finished at step {step} (terminated={terminated}, truncated={truncated})")

    # Save video
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # imageio will infer format from extension (e.g., .gif, .mp4, .avi)
    print(f"Saving video to {out_path} at {args.fps} fps")
    imageio.mimsave(out_path, frames, fps=args.fps)

    plt.close(fig)
    env.close()
    print("Done.")


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

    # video options
    parser.add_argument(
        "--out",
        type=str,
        default="rollouts/rollout.gif",
        help="Output video path (extension controls format, e.g. .gif, .mp4).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the output video.",
    )

    args = parser.parse_args()
    run_rollout_and_record(args)


if __name__ == "__main__":
    main()
