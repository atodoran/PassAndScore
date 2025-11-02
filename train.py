import os, math, time, sys, random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from env import PassAndScoreEnv

def joint_to_pair(a, n_each=5):
    return int(a // n_each), int(a % n_each)

def discount_cumsum(x, gamma):
    out = np.zeros_like(x, dtype=np.float32)
    g = 0.0
    for t in reversed(range(len(x))):
        g = x[t] + gamma * g
        out[t] = g
    return out

# -------------- Actor (unchanged) --------------
class Policy(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, x): 
        return self.net(x)  # logits
    @torch.no_grad()
    def act(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        return int(a.item()), float(logp.item())

# -------------- Critic --------------
class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)  # [B]

CKPT_PATH = "policy.pth"

def atomic_torch_save(state, path):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic on POSIX & Windows 10+

# ----- optional shaping term (safe if info missing) -----
def shaped_reward(info, gamma=0.99):
    if not isinstance(info, dict) or "before" not in info or "after" not in info:
        return 0.0
    diff = {k: info["after"][k] * gamma - info["before"][k] for k in info["before"]}
    r = 0.0
    # r -= 0.1 * diff.get("dist_A_to_ball", 0.0)
    # r += 0.1 * diff.get("passed_regions", 0.0)
    # r += 0.001 * diff.get("speed_ball", 0.0)
    r -= 0.1 * diff.get("dist_ball_to_goal", 0.0)
    return r

# ----- Generalized Advantage Estimation -----
def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """
    rewards: [T]
    values:  [T] value(s_t)
    dones:   [T] bools (True if episode ended at t)
    last_value: V(s_{T}) for bootstrapping (0 if done)
    returns (advantages[T], targets[T]) where targets are V-train targets
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 0.0 if dones[t] else 1.0
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
    targets = adv + values
    return adv, targets

def train(
    seed=0,
    episodes_per_update=10,
    updates=200,
    max_steps=400,
    gamma=0.99,
    lam=0.95,               # <-- GAE(λ)
    lr_actor=3e-4,
    lr_critic=1e-3,
    entropy_coef=0.01,
    value_coef=0.5,         # critic loss weight
    max_grad_norm=0.5,
    hidden=64,
    device="cpu",
):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    env = PassAndScoreEnv(centralized=True, seed=seed)

    # If you changed obs to include sin/cos for headings, set obs_dim=16; else 14.
    obs_dim = 12
    n_actions = 25

    actor = Policy(obs_dim, n_actions, hidden).to(device)
    critic = ValueNet(obs_dim, hidden).to(device)
    opt_actor  = optim.Adam(actor.parameters(), lr=lr_actor)
    opt_critic = optim.Adam(critic.parameters(), lr=lr_critic)

    # Load from existing checkpoint if available
    if os.path.isfile(CKPT_PATH):
        print(f"Loading checkpoint from {CKPT_PATH}...")
        ckpt = torch.load(CKPT_PATH, map_location=device)
        actor.load_state_dict(ckpt["actor"], strict=True)
        critic.load_state_dict(ckpt["critic"], strict=True)
        print("Checkpoint loaded.")

    for it in range(1, updates + 1):
        # ======= Collect trajectories (on-policy) =======
        obs_buf, act_buf, rew_buf, done_buf, val_buf = [], [], [], [], []
        ep_returns = []

        for _ in range(episodes_per_update):
            obs, _ = env.reset()
            ep_rews, ep_obs, ep_acts, ep_vals, ep_dones = [], [], [], [], []
            done, steps = False, 0

            while not done and steps < max_steps:
                # Actor picks action
                x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits = actor(x)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = int(dist.sample().item())
                    value  = float(critic(x).item())
                aA, aB = joint_to_pair(action, 5)

                # Step env
                nxt, r, term, trunc, info = env.step((aA, aB))
                r += shaped_reward(info, gamma=gamma)

                ep_obs.append(obs)
                ep_acts.append(action)
                ep_rews.append(float(r))
                ep_vals.append(value)
                ep_dones.append(bool(term or trunc))

                obs = nxt
                steps += 1
                done = term or trunc

            # Bootstrap value for final state
            if ep_dones and ep_dones[-1]:
                last_v = 0.0
            else:
                x_last = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    last_v = float(critic(x_last).item())

            # GAE for this episode
            adv, tgt = compute_gae(
                rewards=np.asarray(ep_rews, dtype=np.float32),
                values=np.asarray(ep_vals, dtype=np.float32),
                dones=np.asarray(ep_dones, dtype=np.bool_),
                last_value=last_v,
                gamma=gamma,
                lam=lam,
            )

            obs_buf.extend(ep_obs)
            act_buf.extend(ep_acts)
            rew_buf.extend(ep_rews)
            done_buf.extend(ep_dones)
            val_buf.extend(tgt.tolist())  # targets for critic

            ep_returns.append(float(np.sum(ep_rews)))

        # ======= Update actor & critic =======
        O   = torch.as_tensor(np.array(obs_buf, dtype=np.float32), device=device)
        AID = torch.as_tensor(np.array(act_buf, dtype=np.int64), device=device)
        Vt  = torch.as_tensor(np.array(val_buf, dtype=np.float32), device=device)

        # Advantages = targets - current values (recompute values for stability)
        with torch.no_grad():
            V_pred = critic(O)
        Adv = Vt - V_pred
        # Normalize advantages
        Adv = (Adv - Adv.mean()) / (Adv.std() + 1e-8)

        # Actor loss (+ entropy)
        logits = actor(O)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(AID)
        entropy = dist.entropy().mean()
        actor_loss = -(logp * Adv).mean() - entropy_coef * entropy

        # Critic loss (MSE to targets)
        V = critic(O)
        critic_loss = torch.mean((V - Vt) ** 2)

        # Joint update (separate optimizers so you can tune lrs independently)
        opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
        opt_actor.step()

        opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
        opt_critic.step()

        print(f"[{it:04d}] avg_return={np.mean(ep_returns):.3f} "
              f"actor_loss={actor_loss.item():.3f} "
              f"critic_loss={critic_loss.item():.3f} "
              f"H={entropy.item():.3f}")

        if it % 100 == 0:
            atomic_torch_save({"actor": actor.state_dict(),
                               "critic": critic.state_dict(),
                               "obs_dim": obs_dim,
                               "n_actions": n_actions}, CKPT_PATH)

    print("Training complete.")
    torch.save({"actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "obs_dim": obs_dim,
                "n_actions": n_actions}, 'policy.pth')

if __name__ == "__main__":
    train(seed=0, episodes_per_update=8, updates=1000, max_steps=1000,
          gamma=0.99, lam=0.95, lr_actor=3e-4, lr_critic=1e-3,
          entropy_coef=0.01, value_coef=0.5, hidden=64, device="cpu")