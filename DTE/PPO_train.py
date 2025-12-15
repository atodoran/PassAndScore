#!/usr/bin/env python3
import argparse
from pathlib import Path
import math

import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import PassAndScoreEnv

from PPO import (
    PPOAgent,
    RolloutBuffer,
    ShapingConfig,
    RewardShaper,
)

# ---------------------------------------------------------------------
# Paths / history config
# ---------------------------------------------------------------------

HISTORY_KEYS = [
    "steps",
    "shaped_mean",
    "shaped_std",
    "raw_mean",
    "raw_std",
    "score_rate_mean",
    "score_rate_std",
    "touches_mean",
    "touches_std",
    "steps_to_score_mean",
    "steps_to_score_std",
    "passes_mean",
    "passes_std",
]


RUNS_ROOT = Path("models")
RUN_SAVE_DIR = RUNS_ROOT / "run"          # always used for saving current run
HISTORY_FILENAME = "history.npz"


def load_history_from_path(path: Path):
    if path.exists():
        data = np.load(path, allow_pickle=True)
        history = {}
        for k in HISTORY_KEYS:
            if k in data:
                history[k] = data[k].tolist()
            else:
                history[k] = []
        print(f"Loaded training history from {path}")
        return history
    else:
        print(f"No history found at {path}, starting with empty history.")
        return {k: [] for k in HISTORY_KEYS}


def empty_history():
    return {k: [] for k in HISTORY_KEYS}


def save_history(history, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.array(v, dtype=np.float32) for k, v in history.items()}
    np.savez(path, **arrays)


# ---------------------------------------------------------------------
# Shaping config helpers
# ---------------------------------------------------------------------


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
            r_pass=args.A_pass_bonus,
            w_player_to_ball=args.B_player_to_ball_weight,
            w_ball_to_target=args.B_target_weight,
            target_type=args.B_target_kind,
        )


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------


def plot_training_metrics(history, path: str):
    if not history["steps"]:
        return

    steps = np.array(history["steps"], dtype=np.float32)

    def plot_with_band(ax, x, mean, std, color, label):
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        if len(mean) == 0:
            ax.set_visible(False)
            return
        ax.plot(x, mean, label=label, color=color)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()

    # 0: shaped reward
    plot_with_band(
        axes[0],
        steps,
        history["shaped_mean"],
        history["shaped_std"],
        "darkgreen",
        "avg shaped reward",
    )
    axes[0].set_title("Average shaped reward per step")

    # 1: raw env reward
    plot_with_band(
        axes[1],
        steps,
        history["raw_mean"],
        history["raw_std"],
        "darkblue",
        "avg raw reward",
    )
    axes[1].set_title("Average raw env reward per step")

    # 2: score rate
    plot_with_band(
        axes[2],
        steps,
        history["score_rate_mean"],
        history["score_rate_std"],
        "red",
        "score rate",
    )
    axes[2].set_title("Score rate (per episode)")

    # 3: touches until score (scoring episodes only)
    plot_with_band(
        axes[3],
        steps,
        history["touches_mean"],
        history["touches_std"],
        "black",
        "touches until score",
    )
    axes[3].set_title("Touches until score (scoring episodes)")

    # 4: steps until score (scoring episodes only)
    plot_with_band(
        axes[4],
        steps,
        history["steps_to_score_mean"],
        history["steps_to_score_std"],
        "gray",
        "steps until score",
    )
    axes[4].set_title("Steps until score (scoring episodes)")

    # 5: passes per episode (NEW) – only if we have data
    if "passes_mean" in history and history["passes_mean"]:
        plot_with_band(
            axes[5],
            steps,
            history["passes_mean"],
            history["passes_std"],
            "purple",
            "passes per episode",
        )
        axes[5].set_title("Passes per episode")
    else:
        axes[5].axis("off")

    for ax in axes[:5]:
        if not ax.get_visible():
            continue
        ax.set_xlabel("env steps")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved training metrics plot to {path}")


# ---------------------------------------------------------------------
# Evaluation of current policies
# ---------------------------------------------------------------------


def evaluate_policies(
    env: PassAndScoreEnv,
    agent_A: PPOAgent,
    agent_B: PPOAgent,
    cfg_A: ShapingConfig,
    cfg_B: ShapingConfig,
    args,
) -> dict:
    eval_env = PassAndScoreEnv(
        centralized=False,
        ball_spawn_region=args.ball_spawn_region,
        max_steps=args.eval_max_steps,
    )

    shaped_rewards_all = []
    raw_rewards_all = []
    ep_scored_flags = []
    ep_touch_to_score = []
    ep_steps_to_score = []
    ep_passes = []

    for _ in range(args.eval_episodes):
        obs_tuple, info = eval_env.reset()
        obs_A, obs_B = obs_tuple
        obs_A = np.asarray(obs_A, dtype=np.float32)
        obs_B = np.asarray(obs_B, dtype=np.float32)

        shaper_A = RewardShaper(eval_env, "A", args.gamma, cfg_A)
        shaper_B = RewardShaper(eval_env, "B", args.gamma, cfg_B)

        done = False
        steps = 0
        scored = False
        score_steps = None
        score_touches = None
        passes_in_ep = 0

        while not done:
            steps += 1

            act_A, _, _ = agent_A.act(obs_A)
            act_B, _, _ = agent_B.act(obs_B)

            joint_action = np.array([act_A, act_B], dtype=np.int64)
            next_obs_tuple, env_rew, terminated, truncated, info = eval_env.step(
                joint_action
            )
            done = bool(terminated or truncated)

            before = info.get("before", {})
            after = info.get("after", {})

            r_A = shaper_A.shape(env_rew, before, after)
            r_B = shaper_B.shape(env_rew, before, after)

            shaped_rewards_all.append(0.5 * (r_A + r_B))
            raw_rewards_all.append(float(env_rew))

            passes_in_ep += int(after.get("passes_step", 0))

            scored_step = bool(after.get("scored_goal", False))
            if scored_step and not scored:
                scored = True
                score_steps = steps
                score_touches = int(after.get("touches", 0))

            if not done:
                obs_A, obs_B = next_obs_tuple
                obs_A = np.asarray(obs_A, dtype=np.float32)
                obs_B = np.asarray(obs_B, dtype=np.float32)

        ep_scored_flags.append(1.0 if scored else 0.0)
        if scored:
            ep_steps_to_score.append(float(score_steps))
            ep_touch_to_score.append(float(score_touches))
        
        ep_passes.append(float(passes_in_ep))

    eval_env.close()

    if shaped_rewards_all:
        sr = np.array(shaped_rewards_all, dtype=np.float32)
        n_sr = sr.size
        shaped_mean = float(sr.mean())
        shaped_std = float(sr.std(ddof=0) / np.sqrt(n_sr))
    else:
        shaped_mean = float("nan")
        shaped_std = float("nan")

    if raw_rewards_all:
        rr = np.array(raw_rewards_all, dtype=np.float32)
        n_rr = rr.size
        raw_mean = float(rr.mean())
        raw_std = float(rr.std(ddof=0) / np.sqrt(n_rr))
    else:
        raw_mean = float("nan")
        raw_std = float("nan")

    if ep_scored_flags:
        sf = np.array(ep_scored_flags, dtype=np.float32)
        n_sf = sf.size
        score_rate_mean = float(sf.mean())
        score_rate_std = float(sf.std(ddof=0) / np.sqrt(n_sf))
    else:
        score_rate_mean = float("nan")
        score_rate_std = float("nan")

    if ep_touch_to_score:
        t_arr = np.array(ep_touch_to_score, dtype=np.float32)
        n_t = t_arr.size
        touches_mean = float(t_arr.mean())
        touches_std = float(t_arr.std(ddof=0) / np.sqrt(n_t))
    else:
        touches_mean = float("nan")
        touches_std = float("nan")

    if ep_steps_to_score:
        s_arr = np.array(ep_steps_to_score, dtype=np.float32)
        n_s = s_arr.size
        steps_mean = float(s_arr.mean())
        steps_std = float(s_arr.std(ddof=0) / np.sqrt(n_s))
    else:
        steps_mean = float("nan")
        steps_std = float("nan")
    
    if ep_passes:
        p_arr = np.array(ep_passes, dtype=np.float32)
        n_p = p_arr.size
        passes_mean = float(p_arr.mean())
        passes_std = float(p_arr.std(ddof=0) / np.sqrt(n_p))
    else:
        passes_mean = float("nan")
        passes_std = float("nan")

    return dict(
        shaped_mean=shaped_mean,
        shaped_std=shaped_std,
        raw_mean=raw_mean,
        raw_std=raw_std,
        score_rate_mean=score_rate_mean,
        score_rate_std=score_rate_std,
        touches_mean=touches_mean,
        touches_std=touches_std,
        steps_to_score_mean=steps_mean,
        steps_to_score_std=steps_std,
        passes_mean=passes_mean,
        passes_std=passes_std,
    )


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------


def train(args):
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    RUN_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    history_save_path = RUN_SAVE_DIR / HISTORY_FILENAME

    # load history (possibly from another run) ------------------------
    if args.resume is not None:
        src_run_dir = RUNS_ROOT / args.resume
        print(f"Resuming from run directory: {src_run_dir}")
        history_load_path = src_run_dir / HISTORY_FILENAME
        history = load_history_from_path(history_load_path)
    else:
        src_run_dir = None
        history = empty_history()

    # compute starting / target steps --------------------------------
    if history["steps"]:
        global_steps = int(history["steps"][-1])
        target_steps = global_steps + args.total_timesteps
        print(f"Starting from global_steps={global_steps}, target_steps={target_steps}")
    else:
        global_steps = 0
        target_steps = args.total_timesteps
        print(f"Starting fresh run, target_steps={target_steps}")

    # === best-raw tracker across full history =======================
    if history["steps"] and history["raw_mean"]:
        steps_hist = history["steps"]
        raw_hist = history["raw_mean"]
        best_idx = max(
            range(len(raw_hist)),
            key=lambda i: (raw_hist[i], steps_hist[i]),
        )
        best_raw_mean = float(raw_hist[best_idx])
        best_raw_step = int(steps_hist[best_idx])
        print(
            f"Best raw_mean so far from history: {best_raw_mean:.4f} "
            f"at step {best_raw_step}"
        )
    else:
        best_raw_mean = float("-inf")
        best_raw_step = -1
        print("No previous best raw_mean in history; starting fresh best tracker.")

    # env ------------------------------------------------------------
    env = PassAndScoreEnv(
        centralized=False,
        ball_spawn_region=args.ball_spawn_region,
        max_steps=args.max_steps,
    )

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

    hidden_sizes = (args.hidden_size, args.hidden_size)

    # paths for THIS run ---------------------------------------------
    path_A_last = RUN_SAVE_DIR / "A_policy_last.pt"
    path_B_last = RUN_SAVE_DIR / "B_policy_last.pt"
    path_A_best = RUN_SAVE_DIR / "A_policy_best.pt"
    path_B_best = RUN_SAVE_DIR / "B_policy_best.pt"

    train_A = args.student in ("A", "both")
    train_B = args.student in ("B", "both")

    # choose source paths for A/B policies ---------------------------

    def get_policy_source_path(player: str):
        if args.resume is not None:
            # when resuming a run, load LAST policy from that run
            return (RUNS_ROOT / args.resume) / f"{player}_policy_last.pt"
        policy_arg = getattr(args, f"{player}_policy")
        if policy_arg is not None:
            # explicit init run: load BEST policy from that run
            return (RUNS_ROOT / policy_arg) / f"{player}_policy_best.pt"
        return None

    src_A_path = get_policy_source_path("A")
    src_B_path = get_policy_source_path("B")

    if src_A_path is not None and src_A_path.exists():
        print(f"Loading A policy from {src_A_path}")
        agent_A = PPOAgent.load(str(src_A_path), device=device)
    else:
        if src_A_path is not None:
            print(f"Warning: A policy not found at {src_A_path}, creating fresh A.")
        agent_A = PPOAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            lr=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device=device,
        )

    if src_B_path is not None and src_B_path.exists():
        print(f"Loading B policy from {src_B_path}")
        agent_B = PPOAgent.load(str(src_B_path), device=device)
    else:
        if src_B_path is not None:
            print(f"Warning: B policy not found at {src_B_path}, creating fresh B.")
        agent_B = PPOAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            lr=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device=device,
        )

    buffer_A = RolloutBuffer(args.n_steps, obs_dim)
    buffer_B = RolloutBuffer(args.n_steps, obs_dim)

    episode_returns_A = []
    episode_returns_B = []
    ep_ret_A = 0.0
    ep_ret_B = 0.0
    done = False

    # eval / plot scheduling -----------------------------------------
    if args.eval_freq_steps > 0:
        if history["steps"]:
            last_eval_step = history["steps"][-1]
            k = int(last_eval_step // args.eval_freq_steps) + 1
            next_eval_step = k * args.eval_freq_steps
        else:
            next_eval_step = args.eval_freq_steps
    else:
        next_eval_step = None

    if args.plot_freq_steps > 0:
        if history["steps"]:
            last_plot_step = history["steps"][-1]
            k = int(last_plot_step // args.plot_freq_steps) + 1
            next_plot_step = k * args.plot_freq_steps
        else:
            next_plot_step = args.plot_freq_steps
    else:
        next_plot_step = None

    # -----------------------------------------------------------------
    # main loop
    # -----------------------------------------------------------------
    while global_steps < target_steps:
        buffer_A.reset()
        buffer_B.reset()
        shaper_A.reset()
        shaper_B.reset()

        for step in range(args.n_steps):
            act_A, logp_A, val_A = agent_A.act(obs_A)
            act_B, logp_B, val_B = agent_B.act(obs_B)

            joint_action = np.array([act_A, act_B], dtype=np.int64)
            next_obs_tuple, env_rew, terminated, truncated, info = env.step(
                joint_action
            )
            done = bool(terminated or truncated)

            before = info.get("before", {})
            after = info.get("after", {})

            r_A = shaper_A.shape(env_rew, before, after)
            r_B = shaper_B.shape(env_rew, before, after)

            buffer_A.add(obs_A, act_A, r_A, done, val_A, logp_A)
            buffer_B.add(obs_B, act_B, r_B, done, val_B, logp_B)

            ep_ret_A += r_A
            ep_ret_B += r_B

            if done:
                episode_returns_A.append(ep_ret_A)
                episode_returns_B.append(ep_ret_B)
                ep_ret_A = 0.0
                ep_ret_B = 0.0

                obs_tuple, info = env.reset()
                shaper_A.reset()
                shaper_B.reset()
            else:
                obs_tuple = next_obs_tuple

            obs_A, obs_B = obs_tuple
            obs_A = np.asarray(obs_A, dtype=np.float32)
            obs_B = np.asarray(obs_B, dtype=np.float32)

            global_steps += 1
            if global_steps >= target_steps:
                break

        last_val_A = agent_A.value(obs_A)
        last_val_B = agent_B.value(obs_B)

        buffer_A.compute_advantages(
            last_value=last_val_A,
            last_done=done,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        buffer_B.compute_advantages(
            last_value=last_val_B,
            last_done=done,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )

        if train_A:
            agent_A.update(buffer_A)
        if train_B:
            agent_B.update(buffer_B)

        if episode_returns_A:
            mean_A = float(np.mean(episode_returns_A[-10:]))
        else:
            mean_A = 0.0
        if episode_returns_B:
            mean_B = float(np.mean(episode_returns_B[-10:]))
        else:
            mean_B = 0.0

        print(
            f"steps={global_steps} | "
            f"mean_return_A(last10)={mean_A:.3f} | "
            f"mean_return_B(last10)={mean_B:.3f}"
        )

        # periodic evaluation -----------------------------------------
        if next_eval_step is not None and global_steps >= next_eval_step:
            print(f"Running evaluation at steps={global_steps} ...")
            metrics = evaluate_policies(env, agent_A, agent_B, cfg_A, cfg_B, args)

            history["steps"].append(global_steps)
            for k in [
                "shaped_mean",
                "shaped_std",
                "raw_mean",
                "raw_std",
                "score_rate_mean",
                "score_rate_std",
                "touches_mean",
                "touches_std",
                "steps_to_score_mean",
                "steps_to_score_std",
                "passes_mean",
                "passes_std",
            ]:
                history[k].append(metrics[k])

            save_history(history, history_save_path)

            # === update best according to raw_mean ===================
            raw_now = metrics["raw_mean"]
            step_now = global_steps
            better = False
            if raw_now > best_raw_mean:
                better = True
            elif math.isfinite(raw_now) and math.isfinite(best_raw_mean) and raw_now == best_raw_mean:
                if step_now > best_raw_step:
                    better = True

            if better:
                best_raw_mean = float(raw_now)
                best_raw_step = int(step_now)
                print(
                    f"New best raw_mean: {best_raw_mean:.4f} at step {best_raw_step}, "
                    "saving *_policy_best.pt"
                )
                agent_A.save(str(path_A_best))
                agent_B.save(str(path_B_best))

            next_eval_step += args.eval_freq_steps

        # periodic plotting + LAST checkpoint -------------------------
        if next_plot_step is not None and global_steps >= next_plot_step:
            agent_A.save(str(path_A_last))
            agent_B.save(str(path_B_last))
            print(f"Saved LAST A to {path_A_last}")
            print(f"Saved LAST B to {path_B_last}")

            save_history(history, history_save_path)
            plot_training_metrics(history, str(RUN_SAVE_DIR / "training_metrics.png"))

            next_plot_step = min(next_plot_step + args.plot_freq_steps, target_steps - 1)

    # final save ------------------------------------------------------
    agent_A.save(str(path_A_last))
    agent_B.save(str(path_B_last))
    print(f"Saved final LAST A to {path_A_last}")
    print(f"Saved final LAST B to {path_B_last}")

    save_history(history, history_save_path)
    plot_training_metrics(history, str(RUN_SAVE_DIR / "training_metrics.png"))

    env.close()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--max-steps", type=int, default=400)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--n-epochs", type=int, default=10)

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

    parser.add_argument(
        "--eval-freq-steps",
        type=int,
        default=20_000,
        help="Run evaluation every this many env steps (0 to disable).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=40,
        help="Number of episodes per evaluation run.",
    )
    parser.add_argument(
        "--eval-max-steps",
        type=int,
        default=400,
        help="Max steps per episode in the evaluation env.",
    )
    parser.add_argument(
        "--plot-freq-steps",
        type=int,
        default=20_000,
        help="Update training_metrics.png and checkpoint every this many env steps (0 to disable).",
    )

    parser.add_argument(
        "--student",
        type=str,
        default="both",
        choices=["A", "B", "both"],
        help='Which player to train: "A", "B", or "both"',
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Run name under models/ to resume from (load LAST A/B policies and history). "
             "Current run is always saved in models/run.",
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
    train(args)


if __name__ == "__main__":
    main()
