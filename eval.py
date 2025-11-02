#!/usr/bin/env python3
import argparse, os, time, numpy as np
import matplotlib.pyplot as plt
import torch

from env import PassAndScoreEnv
from train import Policy, ValueNet  # actor + critic classes used in training

# ---------- Load actor & critic from A2C checkpoint ----------
def load_actor_critic(ckpt_path, hidden=64, device="cpu"):
    """
    Expects a checkpoint saved by the new actor-critic trainer, with keys:
      {"actor": state_dict, "critic": state_dict, "obs_dim": int, "n_actions": int}
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    if not (isinstance(ckpt, dict) and "actor" in ckpt and "critic" in ckpt):
        raise ValueError("Checkpoint must be from actor-critic training and contain 'actor' and 'critic'.")

    obs_dim   = int(ckpt["obs_dim"])
    n_actions = int(ckpt["n_actions"])

    actor  = Policy(obs_dim, n_actions, hidden).to(device)
    critic = ValueNet(obs_dim, hidden).to(device)
    actor.load_state_dict(ckpt["actor"], strict=True)
    critic.load_state_dict(ckpt["critic"], strict=True)
    actor.eval(); critic.eval()
    return actor, critic, obs_dim, n_actions

@torch.no_grad()
def run_one_episode(actor, critic, seed=0, render_fps=30, max_steps=400, device="cpu",
                    stochastic=True, print_prefix="[eval]"):
    env = PassAndScoreEnv(centralized=True, seed=seed)
    obs, _ = env.reset()
    plt.ion()
    dt = 1.0 / render_fps
    ret = 0.0

    for t in range(max_steps):
        # Compute value BEFORE acting (V(s_t))
        x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        V = float(critic(x).item())

        # Render and overlay V(s)
        env.render()
        try:
            ax = plt.gcf().axes[0]
            title = ax.get_title()
            if title:
                ax.set_title(f"{title}   |   V(s)={V:.3f}")
            else:
                ax.set_title(f"V(s)={V:.3f}")
        except Exception:
            pass
        plt.pause(dt)  # non-blocking GUI update

        # Act
        logits = actor(x)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample().item() if stochastic else torch.argmax(logits, dim=-1).item()
        aA, aB = divmod(int(a), 5)

        # Step
        obs, r, term, trunc, info = env.step((aA, aB))
        ret += r
        print(f"\r{print_prefix} step={t+1}/{max_steps}  reward={r:.3f}  return={ret:.3f}  V={V:.3f}",
              end="", flush=True)
        if term or trunc:
            break

    plt.ioff()
    env.close()
    print(f"\n{print_prefix} return={ret:.3f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to actor-critic checkpoint (policy.pth)")
    p.add_argument("--watch", action="store_true", help="Keep watching for new checkpoints")
    p.add_argument("--interval", type=float, default=1.0, help="Seconds between checks")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=None, help="Episode seed (default: random)")
    p.add_argument("--fps", type=int, default=30, help="Render FPS")
    p.add_argument("--max-steps", type=int, default=400, help="Max steps per episode")
    p.add_argument("--deterministic", action="store_true", help="Greedy actions at eval time")
    p.add_argument("--hidden", type=int, default=64, help="Hidden size (must match training)")
    args = p.parse_args()

    last_mtime = 0.0
    print(f"[eval] Watching {args.ckpt} (interval={args.interval}s)")
    while True:
        if os.path.exists(args.ckpt):
            mtime = os.path.getmtime(args.ckpt)
            if mtime > last_mtime:
                print("\n[eval] New checkpoint detected. Loading and rendering…")
                try:
                    actor, critic, _, _ = load_actor_critic(
                        args.ckpt, hidden=args.hidden, device=args.device
                    )
                    seed = args.seed if args.seed is not None else int(np.random.randint(0, 1_000_000))
                    run_one_episode(
                        actor, critic,
                        seed=seed,
                        render_fps=args.fps,
                        max_steps=args.max_steps,
                        device=args.device,
                        stochastic=(not args.deterministic),
                        print_prefix=f"[eval seed={seed}]",
                    )
                    last_mtime = mtime
                except Exception as e:
                    print(f"[eval] Failed to load/run: {e}")
        if not args.watch:
            break
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
