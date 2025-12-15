# dte_ppo.py

from dataclasses import dataclass
from typing import Dict, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ---------------------------------------------------------------------
# Reward shaping config + shaper
# ---------------------------------------------------------------------


@dataclass
class ShapingConfig:
    r_first_touch: float = 0.0
    r_bump_penalty: float = 0.0
    r_pass: float = 0.0
    w_player_to_ball: float = 0.0
    w_ball_to_target: float = 0.0
    target_type: str = "none"  # "none", "goal", "regionB", "playerB"

class RewardShaper:
    def __init__(
        self,
        env,
        player_id: str,
        gamma: float,
        config: ShapingConfig,
    ):
        assert player_id in ("A", "B")
        self.player_id = player_id
        self.gamma = float(gamma)
        self.config = config

        # geometry from env
        self.region_b_xmin = float(env.region_b.xmin)

        xmin = env.region_a.xmin - env.field_padding
        xmax = env.region_b.xmax + env.field_padding
        ymin = min(env.region_a.ymin, env.region_b.ymin) - env.field_padding
        ymax = max(env.region_a.ymax, env.region_b.ymax) + env.field_padding

        dx = xmax - xmin
        dy = ymax - ymin
        diag = float(np.hypot(dx, dy))

        # global distance scales
        self.player_ball_scale = max(diag, 1e-6)   # player <-> ball
        self.goal_scale = max(diag, 1e-6)          # ball <-> goal mouth

        # horizontal distance to region B border
        regionB_scale = self.region_b_xmin - xmin
        if regionB_scale <= 0.0:
            regionB_scale = dx
        self.regionB_scale = max(float(regionB_scale), 1e-6)

        # ball <-> player B
        self.playerB_scale = max(diag, 1e-6)

        self._touched = False

    def reset(self):
        self._touched = False

    # --- raw distances from info ---

    def _player_dist_to_ball(self, info: Dict[str, Any]) -> float:
        if self.player_id == "A":
            return float(info.get("dist_A_to_ball", 0.0))
        else:
            return float(info.get("dist_B_to_ball", 0.0))

    def _ball_dist_to_goal(self, info: Dict[str, Any]) -> float:
        return float(info.get("dist_ball_to_goal", 0.0))

    def _ball_dist_to_regionB_border(self, info: Dict[str, Any]) -> float:
        bx = float(info.get("ball_x", 0.0))
        in_B = bool(info.get("passed_regions", False))
        if in_B:
            return 0.0
        return max(0.0, self.region_b_xmin - bx)

    def _ball_dist_to_playerB(self, info: Dict[str, Any]) -> float:
        return float(info.get("dist_B_to_ball", 0.0))

    def _ball_dist_to_target(self, info: Dict[str, Any]) -> float:
        t = self.config.target_type
        if t == "none" or self.config.w_ball_to_target == 0.0:
            return 0.0

        if t == "goal":
            return self._ball_dist_to_goal(info)
        if t == "regionB":
            return self._ball_dist_to_regionB_border(info)
        if t == "playerB":
            return self._ball_dist_to_playerB(info)

        return 0.0

    # --- potentials: closeness in [0, 1] ---

    @staticmethod
    def _closeness_from_distance(d: float, scale: float) -> float:
        if scale <= 0.0:
            return 0.0
        c = 1.0 - d / (scale + 1e-9)
        if c < 0.0:
            c = 0.0
        elif c > 1.0:
            c = 1.0
        return float(c)

    def _phi_player_to_ball(self, info: Dict[str, Any]) -> float:
        if self._touched or self.config.w_player_to_ball == 0.0:
            return 0.0
        d = self._player_dist_to_ball(info)
        c = self._closeness_from_distance(d, self.player_ball_scale)
        return self.config.w_player_to_ball * c

    def _phi_ball_to_target(self, info: Dict[str, Any]) -> float:
        if self.config.target_type == "none" or self.config.w_ball_to_target == 0.0:
            return 0.0

        d = self._ball_dist_to_target(info)
        if self.config.target_type == "goal":
            scale = self.goal_scale
        elif self.config.target_type == "regionB":
            scale = self.regionB_scale
        elif self.config.target_type == "playerB":
            scale = self.playerB_scale
        else:
            scale = self.goal_scale

        c = self._closeness_from_distance(d, scale)
        return self.config.w_ball_to_target * c

    def _phi(self, info: Dict[str, Any]) -> float:
        return self._phi_player_to_ball(info) + self._phi_ball_to_target(info)

    # --- main API ---

    def shape(self, env_reward: float, before: Dict[str, Any], after: Dict[str, Any]) -> float:
        env_r = float(env_reward)

        phi_before = self._phi(before)
        phi_after = self._phi(after)
        potential_term = self.gamma * phi_after - phi_before

        touch_bonus = 0.0
        last_touch = str(after.get("last_touch", "A"))
        touches_step = int(after.get("touches_step", 0))
        if (
            touches_step > 0
            and last_touch == self.player_id
            and not self._touched
            and self.config.r_first_touch != 0.0
        ):
            touch_bonus = self.config.r_first_touch
            self._touched = True
        
        bump_penalty = 0.0
        if after.get(("bumped_" + self.player_id), False):
            bump_penalty = -self.config.r_bump_penalty
    
        passes_step = int(after.get("passes_step", 0))
        pass_bonus = self.config.r_pass * passes_step if passes_step > 0 else 0.0

        return env_r + potential_term + touch_bonus + bump_penalty + pass_bonus



# ---------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------


class RolloutBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
    ):
        self.buffer_size = int(buffer_size)
        self.obs_dim = int(obs_dim)

        self.reset()

    def reset(self):
        self.obs = np.zeros((self.buffer_size, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size,), dtype=np.int64)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size,), dtype=np.float32)
        self.values = np.zeros((self.buffer_size,), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size,), dtype=np.float32)

        self.advantages = np.zeros((self.buffer_size,), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size,), dtype=np.float32)

        self.pos = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ):
        assert self.pos < self.buffer_size, "RolloutBuffer overflow"
        self.obs[self.pos] = obs
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = float(done)
        self.values[self.pos] = float(value)
        self.log_probs[self.pos] = float(log_prob)
        self.pos += 1

    def compute_advantages(
        self,
        last_value: float,
        last_done: bool,
        gamma: float,
        gae_lambda: float,
    ):
        last_value = float(last_value)
        last_done = bool(last_done)
        gamma = float(gamma)
        gae_lambda = float(gae_lambda)

        last_gae = 0.0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]

            delta = (
                self.rewards[step]
                + gamma * next_value * next_non_terminal
                - self.values[step]
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae
            self.returns[step] = self.advantages[step] + self.values[step]

    def get(self) -> Dict[str, np.ndarray]:
        assert self.pos == self.buffer_size, "Buffer not full"
        return dict(
            obs=self.obs,
            actions=self.actions,
            log_probs=self.log_probs,
            advantages=self.advantages,
            returns=self.returns,
        )

    def iter_minibatches(self, batch_size: int):
        batch_size = int(batch_size)
        indices = np.arange(self.buffer_size)
        np.random.shuffle(indices)
        for start in range(0, self.buffer_size, batch_size):
            end = start + batch_size
            if end > self.buffer_size:
                break
            mb_idx = indices[start:end]
            yield mb_idx


# ---------------------------------------------------------------------
# PPO agent
# ---------------------------------------------------------------------


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Tuple[int, int]):
        super().__init__()
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(h2, act_dim)
        self.value_head = nn.Linear(h2, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(obs)
        logits = self.policy_head(x)
        values = self.value_head(x).squeeze(-1)
        return logits, values

    def act(self, obs: torch.Tensor, det=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        if det:
            action = torch.argmax(logits, dim=-1)
            log_prob = torch.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(-1)).squeeze(-1)
            return action, log_prob, value
        else:
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob, value


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Tuple[int, int],
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        n_epochs: int = 10,
        batch_size: int = 64,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_sizes = tuple(hidden_sizes)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_range = float(clip_range)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.ent_coef = float(ent_coef)
        self.vf_coef = float(vf_coef)
        self.max_grad_norm = float(max_grad_norm)

        self.device = torch.device(device)

        self.ac = MLPActorCritic(self.obs_dim, self.act_dim, self.hidden_sizes).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=lr)

        self.total_updates = 0

        self.config = dict(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=self.hidden_sizes,
            lr=lr,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
        )

    # --- acting / value ---

    def act(self, obs_np: np.ndarray, det=False) -> Tuple[int, float, float]:
        obs_t = torch.as_tensor(
            obs_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            action_t, log_prob_t, value_t = self.ac.act(obs_t, det=det)
        action = int(action_t.item())
        log_prob = float(log_prob_t.item())
        value = float(value_t.item())
        return action, log_prob, value

    def value(self, obs_np: np.ndarray) -> float:
        obs_t = torch.as_tensor(
            obs_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            _, value_t = self.ac.forward(obs_t)
        return float(value_t.item())

    # --- training ---

    def _evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.ac(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values

    def update(self, buffer: RolloutBuffer):
        data = buffer.get()
        obs = torch.as_tensor(data["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(
            data["actions"], dtype=torch.long, device=self.device
        )
        old_log_probs = torch.as_tensor(
            data["log_probs"], dtype=torch.float32, device=self.device
        )
        returns = torch.as_tensor(
            data["returns"], dtype=torch.float32, device=self.device
        )
        advantages = torch.as_tensor(
            data["advantages"], dtype=torch.float32, device=self.device
        )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.n_epochs):
            for mb_idx in buffer.iter_minibatches(self.batch_size):
                mb_idx_t = torch.as_tensor(mb_idx, dtype=torch.long, device=self.device)

                obs_mb = obs[mb_idx_t]
                actions_mb = actions[mb_idx_t]
                old_log_probs_mb = old_log_probs[mb_idx_t]
                returns_mb = returns[mb_idx_t]
                adv_mb = advantages[mb_idx_t]

                log_probs_mb, entropy_mb, values_mb = self._evaluate_actions(
                    obs_mb, actions_mb
                )

                ratio = torch.exp(log_probs_mb - old_log_probs_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                ) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values_mb, returns_mb)

                entropy_loss = -entropy_mb.mean()

                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.ac.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

        self.total_updates += 1

    # --- save / load ---

    def save(self, path: str):
        ckpt = dict(
            state_dict=self.ac.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            config=self.config,
            total_updates=self.total_updates,
        )
        torch.save(ckpt, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "PPOAgent":
        ckpt = torch.load(path, map_location=device)
        cfg = ckpt["config"]
        agent = cls(
            obs_dim=cfg["obs_dim"],
            act_dim=cfg["act_dim"],
            hidden_sizes=tuple(cfg["hidden_sizes"]),
            lr=cfg["lr"],
            gamma=cfg["gamma"],
            gae_lambda=cfg["gae_lambda"],
            clip_range=cfg["clip_range"],
            n_epochs=cfg["n_epochs"],
            batch_size=cfg["batch_size"],
            ent_coef=cfg["ent_coef"],
            vf_coef=cfg["vf_coef"],
            max_grad_norm=cfg["max_grad_norm"],
            device=device,
        )
        agent.ac.load_state_dict(ckpt["state_dict"])
        agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        agent.total_updates = int(ckpt.get("total_updates", 0))
        return agent
