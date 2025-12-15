#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from env import PassAndScoreEnv


def run_one_episode(
    model,
    seed=0,
    render_fps=30,
    max_steps=400,
    deterministic=False,
):
    base_env = PassAndScoreEnv(centralized=True, ball_spawn_region="A", seed=seed)
    obs, info = base_env.reset()

    plt.ion()
    dt = 1.0 / float(render_fps)

    base_env.render()
    fig = plt.gcf()
    closed = False

    def _on_close(evt):
        nonlocal closed
        closed = True

    cid = fig.canvas.mpl_connect("close_event", _on_close)

    ret_env = 0.0
    last_touches = int(info.get("touches", 0))

    for t in range(max_steps):
        if closed or not plt.fignum_exists(fig.number):
            print("\nWindow closed, exiting episode...")
            break

        base_env.render()
        plt.pause(dt)

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = base_env.step(action)

        ret_env += float(reward)

        touches = int(info.get("after", {}).get("touches", last_touches))
        last_touches = touches

        done = terminated or truncated

        print(
            f"\r[eval seed={seed}] step={t+1}/{max_steps}  "
            f"r_env={float(reward):.3f}  "
            f"R_env={ret_env:.3f}  "
            f"touches={touches}",
            end="",
            flush=True,
        )

        if done:
            break

    print(
        f"\n[eval seed={seed}] final env_return={ret_env:.3f}, "
        f"touches={last_touches}"
    )

    plt.ioff()
    try:
        fig.canvas.mpl_disconnect(cid)
    except Exception:
        pass
    try:
        plt.close(fig)
    except Exception:
        pass

    base_env.close()
    return ret_env, last_touches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/PPO/policy",
        help="Path to SB3 model (without .zip or with .zip)",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--deterministic", action="store_true")

    args = parser.parse_args()

    base_path = Path(args.model_path)
    if base_path.suffix != ".zip":
        model_file = base_path.with_suffix(".zip")
    else:
        model_file = base_path

    if not model_file.exists():
        raise FileNotFoundError(f"Could not find model file: {model_file}")

    model = PPO.load(str(model_file))

    rng = np.random.default_rng()

    for ep in range(args.episodes):
        seed = int(rng.integers(0, 2**31 - 1))
        print(f"\n=== Episode {ep + 1} / {args.episodes} | seed={seed} ===")
        run_one_episode(
            model,
            seed=seed,
            render_fps=args.fps,
            max_steps=args.max_steps,
            deterministic=args.deterministic,
        )


if __name__ == "__main__":
    main()
