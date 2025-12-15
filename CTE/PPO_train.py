#!/usr/bin/env python3
import argparse
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env import PassAndScoreEnv


class ShapingWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        gamma: float,
        w_x: float = 0.0,
        w_goal: float = 0.0,
        r_first_touch_a: float = 0.0,
        r_first_touch_b: float = 0.0,
        w_speed_pass: float = 0.0,
        w_speed_goal: float = 0.0,
        r_wall_bump: float = 0.0,   # NEW: penalty when an agent bumps into region wall
        score_only: bool = False,
    ):
        super().__init__(env)
        self.gamma = float(gamma)
        self.w_x = float(w_x)
        self.w_goal = float(w_goal)
        self.r_first_touch_a = float(r_first_touch_a)
        self.r_first_touch_b = float(r_first_touch_b)
        self.w_speed_pass = float(w_speed_pass)
        self.w_speed_goal = float(w_speed_goal)
        self.r_wall_bump = float(r_wall_bump)
        self.score_only = bool(score_only)

        self._a_touched = False
        self._b_touched = False

        xmin = self.env.region_a.xmin - self.env.field_padding
        xmax = self.env.region_b.xmax + self.env.field_padding
        ymin = min(self.env.region_a.ymin, self.env.region_b.ymin) - self.env.field_padding
        ymax = max(self.env.region_a.ymax, self.env.region_b.ymax) + self.env.field_padding

        self._xmin = float(xmin)
        self._xmax = float(xmax)
        self._goal_scale = float(np.hypot(xmax - xmin, ymax - ymin))

    def _px(self, ball_x: float) -> float:
        # 0 on left, ~1 once ball is past mid-field (right of center)
        return float(
            np.clip((ball_x - self._xmin) / ((self._xmax - self._xmin) / 2 + 1e-9), 0.0, 1.0)
        )

    def _pgoal(self, d_goal: float) -> float:
        return float(np.clip(1.0 - d_goal / (self._goal_scale + 1e-9), 0.0, 1.0))

    def _phi(self, info_dict) -> float:
        bx = float(info_dict.get("ball_x", 0.0))
        d_goal = float(info_dict.get("dist_ball_to_goal", 0.0))

        p = 0.0
        if not self.score_only and self.w_x > 0.0:
            p += self.w_x * self._px(bx)
        if self.w_goal > 0.0:
            p += self.w_goal * self._pgoal(d_goal)
        return float(p)

    def _speed_terms(self, before, after) -> float:
        """
        Signed speed shaping:

        - Outside region B: reward forward (+x) ball velocity, penalize backward.
        - Inside region B: reward velocity toward the goal mouth, penalize away.
        """
        if self.w_speed_pass == 0.0 and self.w_speed_goal == 0.0:
            return 0.0

        vball = np.array(self.env.state["vball"], dtype=np.float32)
        pball = np.array(self.env.state["pball"], dtype=np.float32)

        # Forward along +x (toward region B)
        v_forward = float(vball[0])

        # Component toward goal mouth
        gx = float(np.clip(pball[0], self.env.goal_xmin, self.env.goal_xmax))
        gy = float(self.env.goal_y)
        goal_point = np.array([gx, gy], dtype=np.float32)
        to_goal = goal_point - pball
        dist_goal = float(np.linalg.norm(to_goal))

        if dist_goal > 1e-9:
            dir_goal = to_goal / dist_goal
            v_to_goal = float(np.dot(vball, dir_goal))
        else:
            v_to_goal = 0.0

        in_B_after = bool(after.get("passed_regions", False))

        if not in_B_after:
            return float(self.w_speed_pass * v_forward)
        else:
            return float(self.w_speed_goal * v_to_goal)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        base_info = {}
        if hasattr(self.env, "_get_info"):
            try:
                base_info = self.env._get_info()
            except Exception:
                base_info = {}

        self._a_touched = False
        self._b_touched = False

        info = dict(info or {})
        info.update(base_info)
        return obs, info

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        before = info.get("before", {})
        after = info.get("after", {})

        last_touch = str(after.get("last_touch", "A"))
        touches_step = int(after.get("touches_step", 0))

        phi_before = self._phi(before)
        phi_after = self._phi(after)
        potential_term = self.gamma * phi_after - phi_before

        # first-touch bonuses
        touch_bonus = 0.0
        if touches_step > 0:
            if last_touch == "A" and not self._a_touched:
                touch_bonus += self.r_first_touch_a
                self._a_touched = True
            elif last_touch == "B" and not self._b_touched:
                touch_bonus += self.r_first_touch_b
                self._b_touched = True

        # speed shaping
        speed_bonus = self._speed_terms(before, after)

        # NEW: wall-bump penalty if either agent hit their region boundary this step
        wall_bump_penalty = 0.0
        if self.r_wall_bump != 0.0:
            bumped_A = bool(after.get("bumped_A", False))
            bumped_B = bool(after.get("bumped_B", False))
            if bumped_A or bumped_B:
                wall_bump_penalty = -self.r_wall_bump

        reward = float(env_reward + potential_term + touch_bonus + speed_bonus + wall_bump_penalty)

        info["env_reward"] = float(env_reward)
        info["potential_term"] = float(potential_term)
        info["touch_bonus"] = float(touch_bonus)
        info["speed_bonus"] = float(speed_bonus)
        info["wall_bump_penalty"] = float(wall_bump_penalty)
        info["phi_before"] = float(phi_before)
        info["phi_after"] = float(phi_after)

        return obs, reward, terminated, truncated, info

def make_env(
    max_steps,
    gamma,
    w_x,
    w_goal,
    r_first_touch_a,
    r_first_touch_b,
    w_speed_pass,
    w_speed_goal,
    r_wall_bump,          # NEW
    score_only,
    ball_spawn_region,
):
    def _thunk():
        base_env = PassAndScoreEnv(
            centralized=True,
            ball_spawn_region=ball_spawn_region,
            max_steps=max_steps,
        )
        return ShapingWrapper(
            base_env,
            gamma=gamma,
            w_x=w_x,
            w_goal=w_goal,
            r_first_touch_a=r_first_touch_a,
            r_first_touch_b=r_first_touch_b,
            w_speed_pass=w_speed_pass,
            w_speed_goal=w_speed_goal,
            r_wall_bump=r_wall_bump,   # NEW
            score_only=score_only,
        )

    return _thunk


def plot_curve(points, path: str):
    if not points:
        return
    pts = sorted(points, key=lambda x: x[0])
    steps, rets = zip(*pts)
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, rets, marker="o", linewidth=2)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Mean return")
    ax.set_title("PPO on PassAndScoreEnv (shaped reward)")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_model_and_meta(model, points, timesteps: int, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.zip")
    model.save(str(tmp_path))

    meta = {
        "curve": [[float(s), float(r)] for (s, r) in points],
        "timesteps": int(timesteps),
    }
    meta_json = json.dumps(meta)

    with zipfile.ZipFile(tmp_path, "r") as zin, zipfile.ZipFile(path, "w") as zout:
        for item in zin.infolist():
            with zin.open(item.filename) as f:
                zout.writestr(item, f.read())
        zout.writestr("training_meta.json", meta_json)

    try:
        Path(tmp_path).unlink()
    except Exception:
        pass


def load_meta_from_zip(path: Path):
    if not path.exists():
        return [], 0
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            meta_names = [n for n in names if n == "training_meta.json"]
            if meta_names:
                data = json.loads(zf.read(meta_names[-1]).decode("utf-8"))
                curve_raw = data.get("curve", [])
                curve = [tuple(map(float, p)) for p in curve_raw]
                timesteps = int(data.get("timesteps", 0))
                return curve, timesteps
            if "curve.npy" in names:
                with zf.open("curve.npy") as f:
                    arr = np.load(f)
                curve = [tuple(map(float, p)) for p in arr.tolist()]
                return curve, 0
    except Exception:
        pass
    return [], 0


class EvalAndCheckpointCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_freq: int,
        n_eval_episodes: int,
        curve_points,
        model_zip_path: Path,
        best_zip_path: Path,
        plot_path: str,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.curve_points = curve_points
        self.model_zip_path = model_zip_path
        self.best_zip_path = best_zip_path
        self.plot_path = plot_path

        self.best_mean_reward = float("-inf")

        if self.curve_points:
            plot_curve(self.curve_points, self.plot_path)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        mean_reward, _ = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            warn=False,
        )

        self.curve_points.append((int(self.num_timesteps), float(mean_reward)))
        self._save_latest()

        if mean_reward > self.best_mean_reward:
            if self.verbose > 0:
                print(
                    f"New best mean reward: {mean_reward:.3f} (old {self.best_mean_reward:.3f})"
                )
            self.best_mean_reward = float(mean_reward)
            self._save_best()

        return True

    def _on_training_end(self) -> None:
        self._save_latest()

    def _save_latest(self):
        save_model_and_meta(
            self.model,
            self.curve_points,
            int(self.model.num_timesteps),
            self.model_zip_path,
        )
        plot_curve(self.curve_points, self.plot_path)

    def _save_best(self):
        save_model_and_meta(
            self.model,
            self.curve_points,
            int(self.model.num_timesteps),
            self.best_zip_path,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--eval-freq", type=int, default=30_000)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reset-curve",
        action="store_true",
        help="When resuming, ignore stored curve and restart plot/timesteps from 0",
    )

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--n-envs", type=int, default=1)

    parser.add_argument("--w-x", type=float, default=0.0)
    parser.add_argument("--w-goal", type=float, default=0.0)
    parser.add_argument("--r-first-touch-a", type=float, default=0.0)
    parser.add_argument("--r-first-touch-b", type=float, default=0.0)
    parser.add_argument("--w-speed-pass", type=float, default=0.0)
    parser.add_argument("--w-speed-goal", type=float, default=0.0)
    parser.add_argument(
        "--score-only", action="store_true", help="Keep only ball->goal shaping"
    )
    parser.add_argument(
        "--r-wall-bump",
        type=float,
        default=0.0,
        help="Penalty when an agent bumps into the region wall",
    )


    parser.add_argument(
        "--ball-spawn-region", type=str, default="A", choices=["A", "B"]
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for PPO networks",
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    env_fns = [
        make_env(
            args.max_steps,
            args.gamma,
            args.w_x,
            args.w_goal,
            args.r_first_touch_a,
            args.r_first_touch_b,
            args.w_speed_pass,
            args.w_speed_goal,
            args.r_wall_bump,   # NEW
            args.score_only,
            args.ball_spawn_region,
        )
        for _ in range(args.n_envs)
    ]
    VecCls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    env = VecCls(env_fns)

    eval_env = VecCls(
        [
            make_env(
                args.max_steps,
                args.gamma,
                args.w_x,
                args.w_goal,
                args.r_first_touch_a,
                args.r_first_touch_b,
                args.w_speed_pass,
                args.w_speed_goal,
                args.r_wall_bump,   # NEW
                args.score_only,
                args.ball_spawn_region,
            )
        ]
    )


    model_zip_path = Path("models/PPO/policy.zip")
    best_zip_path = Path("models/PPO/best.zip")
    plot_path = "plot.png"

    ppo_kwargs = dict(
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        n_epochs=args.n_epochs,
        device=device,
    )

    # Hidden size only matters when creating a *new* model.
    policy_kwargs = dict(net_arch=[args.hidden_size, args.hidden_size])

    if args.resume:
        if not model_zip_path.exists():
            raise FileNotFoundError(
                "models/PPO/policy.zip not found but --resume was given."
            )

        # Load the full model (architecture + weights) from disk.
        model = PPO.load(str(model_zip_path), env=env, device=device)

        curve_points, loaded_timesteps = load_meta_from_zip(model_zip_path)

        if args.reset_curve:
            print("Resuming weights but resetting curve and timesteps to 0.")
            curve_points = []
            model.num_timesteps = 0
            reset_flag = True
        else:
            model.num_timesteps = int(loaded_timesteps)
            reset_flag = False

    else:
        # New run: use the requested hidden size for both layers.
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            **ppo_kwargs,
        )
        curve_points = []
        reset_flag = True


    callback = EvalAndCheckpointCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        curve_points=curve_points,
        model_zip_path=model_zip_path,
        best_zip_path=best_zip_path,
        plot_path=plot_path,
        verbose=1,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        reset_num_timesteps=reset_flag,
    )

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
