import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

# Minimal spaces stubs
class Space:
    def sample(self, rng=None):
        raise NotImplementedError

class Discrete(Space):
    def __init__(self, n:int):
        self.n = int(n)
    def sample(self, rng=None):
        import numpy as _np
        rng = rng or _np.random.default_rng()
        return int(rng.integers(0, self.n))

class MultiDiscrete(Space):
    def __init__(self, nvec):
        import numpy as _np
        self.nvec = _np.array(nvec, dtype=int)
    def sample(self, rng=None):
        import numpy as _np
        rng = rng or _np.random.default_rng()
        return _np.array([rng.integers(0, n) for n in self.nvec], dtype=int)

class Box(Space):
    def __init__(self, low, high, shape=None, dtype=np.float32):
        import numpy as _np
        self.low = _np.full(shape, low, dtype=dtype) if _np.isscalar(low) else _np.array(low, dtype=dtype)
        self.high = _np.full(shape, high, dtype=dtype) if _np.isscalar(high) else _np.array(high, dtype=dtype)
        self.shape = shape or self.low.shape
        self.dtype = dtype
    def sample(self, rng=None):
        import numpy as _np
        rng = rng or _np.random.default_rng()
        return rng.uniform(self.low, self.high).astype(self.dtype)

@dataclass
class Region:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    def sample_point(self, rng: np.random.Generator, interior_ratio: float = 1.0) -> np.ndarray:
        p = float(interior_ratio)
        if p <= 0.0:
            raise ValueError("interior_ratio must be > 0.0")
        if p > 1.0:
            raise ValueError("interior_ratio must be <= 1.0")

        cx = (self.xmin + self.xmax) / 2.0
        cy = (self.ymin + self.ymax) / 2.0
        half_w = (self.xmax - self.xmin) * 0.5 * interior_ratio
        half_h = (self.ymax - self.ymin) * 0.5 * interior_ratio

        xmin = cx - half_w
        xmax = cx + half_w
        ymin = cy - half_h
        ymax = cy + half_h

        return np.array([
            rng.uniform(xmin, xmax),
            rng.uniform(ymin, ymax)
        ], dtype=np.float32)

    def contains_point(self, p: np.ndarray) -> bool:
        return (self.xmin <= p[0] <= self.xmax) and (self.ymin <= p[1] <= self.ymax)
    
class PassAndScoreEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        centralized: bool = True,
        seed: Optional[int] = None,
        dt: float = 0.05,
        max_steps: int = 400,
        field_padding: float = 0.2,
        region_a: Optional[Region] = None,
        region_b: Optional[Region] = None,
        goal_width: float = 0.8,
        agent_radius: float = 0.05,
        ball_radius: float = 0.03,
        max_speed_agent: float = 1.2,
        max_speed_ball: float = 2.0,
        accel: float = 1.0,
        friction_agent: float = 0.95,
        friction_ball: float = 0.995,
        restitution: float = 0.5,
        sample_interior_ratio: float = 0.8,
        ball_spawn_region: str = "A",
        pass_cone_deg: float = 30.0,
        pass_speed_min_frac: float = 0.1,
    ):
        super().__init__()
        self.centralized = centralized
        self.dt = dt
        self.max_steps = max_steps
        self.field_padding = field_padding
        self.region_a = region_a or Region(-2.0, -0.2, -1.0, 1.0)
        self.region_b = region_b or Region(0.2, 2.0, -1.0, 1.0)
        self.goal_width = goal_width

        self.agent_radius = agent_radius
        self.ball_radius = ball_radius
        self.max_speed_agent = max_speed_agent
        self.max_speed_ball = max_speed_ball
        self.accel = accel
        self.friction_agent = friction_agent
        self.friction_ball = friction_ball
        self.restitution = restitution

        self.np_random: np.random.Generator = np.random.default_rng(seed)
        self.sample_interior_ratio = sample_interior_ratio
        self.ball_spawn_region = ball_spawn_region.upper()
        self.pass_cone_rad = math.radians(pass_cone_deg)
        self.pass_speed_min_frac = float(pass_speed_min_frac)

        self.goal_y = self.region_b.ymax
        gx_center = (self.region_b.xmin + self.region_b.xmax) / 2.0
        self.goal_xmin = gx_center - self.goal_width / 2.0
        self.goal_xmax = gx_center + self.goal_width / 2.0
        self.pass_line_x = (self.region_a.xmax + self.region_b.xmin) / 2.0
        
        # actions: 0 stay, 1 up, 2 down, 3 left, 4 right
        self.single_action_space = spaces.Discrete(5)
        self.action_space = spaces.MultiDiscrete([5, 5])

        # Observations:
        # centralized: pA(2), pB(2), vA(2), vB(2), pball(2), vball(2) -> 12
        self.observation_space_central = spaces.Box(
            -np.inf, np.inf, shape=(12,), dtype=np.float32
        )
        # per-agent (decentralized): own p(2), own v(2), rel p ball(2), rel v ball(2), rel p other(2), rel v other(2) -> 12
        self.observation_space_agent = spaces.Box(
            -np.inf, np.inf, shape=(12,), dtype=np.float32
        )

        # This is what SB3 will look at
        if self.centralized:
            self.observation_space = self.observation_space_central
        else:
            # not used with SB3 here, but keep it consistent
            self.observation_space = spaces.Tuple(
                [self.observation_space_agent, self.observation_space_agent]
            )

        self.state: Dict[str, Any] = {}
        self.fig = None
        self.ax = None

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed(seed)

        pA = self.region_a.sample_point(self.np_random, interior_ratio=self.sample_interior_ratio)
        pB = self.region_b.sample_point(self.np_random, interior_ratio=self.sample_interior_ratio)
        vA = np.zeros(2, dtype=np.float32)
        vB = np.zeros(2, dtype=np.float32)

        # --- NEW: ball spawn logic with A-pass support ---
        if self.ball_spawn_region == "A":
            region_ball = self.region_a
            pball = region_ball.sample_point(self.np_random, interior_ratio=self.sample_interior_ratio)
            vball = np.zeros(2, dtype=np.float32)

        elif self.ball_spawn_region == "B":
            region_ball = self.region_b
            pball = region_ball.sample_point(self.np_random, interior_ratio=self.sample_interior_ratio)
            vball = np.zeros(2, dtype=np.float32)

        elif self.ball_spawn_region == "A-PASS":
            # position: still random in region A
            region_ball = self.region_a
            pball = region_ball.sample_point(self.np_random, interior_ratio=self.sample_interior_ratio)

            # direction: cone around +x, symmetric
            angle = float(self.np_random.uniform(-self.pass_cone_rad, self.pass_cone_rad))
            dir_vec = np.array(
                [math.cos(angle), math.sin(angle)],
                dtype=np.float32,
            )

            # speed: between pass_speed_min_frac*max_speed_ball and max_speed_ball
            speed_min = self.pass_speed_min_frac * self.max_speed_ball
            speed = float(self.np_random.uniform(speed_min, self.max_speed_ball))

            vball = dir_vec * speed

        else:
            raise ValueError(f"Unknown ball_spawn_region: {self.ball_spawn_region!r}")

        self.state = dict(
            pA=pA, pB=pB, vA=vA, vB=vB,
            pball=pball, vball=vball, steps=0,
            last_touch="A",
            touches=0,
            passes=0,
        )

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: Union[np.ndarray, Tuple[int, int]]):
        aA, aB = int(action[0]), int(action[1])
        s = self.state
        info = {"before": self._get_info()}
        pball_before = s["pball"].copy()

        step_touches = 0
        step_passes = 0

        # action -> direction vector mapping
        def action_vec(a: int) -> np.ndarray:
            if a == 1:   # up
                return np.array([0.0, 1.0], dtype=np.float32)
            elif a == 2: # down
                return np.array([0.0, -1.0], dtype=np.float32)
            elif a == 3: # left
                return np.array([-1.0, 0.0], dtype=np.float32)
            elif a == 4: # right
                return np.array([1.0, 0.0], dtype=np.float32)
            else:        # stay / unknown
                return np.array([0.0, 0.0], dtype=np.float32)

        s["vA"] = s["vA"] + self.accel * self.dt * action_vec(aA)
        s["vB"] = s["vB"] + self.accel * self.dt * action_vec(aB)

        def cap(v, m):
            sp = np.linalg.norm(v)
            return v * (m / sp) if sp > m else v

        s["vA"] = cap(s["vA"], self.max_speed_agent)
        s["vB"] = cap(s["vB"], self.max_speed_agent)

        # integrate positions
        s["pA"] = s["pA"] + s["vA"] * self.dt
        s["pB"] = s["pB"] + s["vB"] * self.dt
        s["pball"] = s["pball"] + s["vball"] * self.dt

        # --- detect wall bumps from clamping ---
        pA_before = s["pA"].copy()
        pB_before = s["pB"].copy()

        # Clamp agents to their regions
        s["pA"][0] = np.clip(s["pA"][0], self.region_a.xmin, self.region_a.xmax)
        s["pA"][1] = np.clip(s["pA"][1], self.region_a.ymin, self.region_a.ymax)
        s["pB"][0] = np.clip(s["pB"][0], self.region_b.xmin, self.region_b.xmax)
        s["pB"][1] = np.clip(s["pB"][1], self.region_b.ymin, self.region_b.ymax)

        bumped_A = bool(
            (s["pA"][0] != pA_before[0]) or (s["pA"][1] != pA_before[1])
        )
        bumped_B = bool(
            (s["pB"][0] != pB_before[0]) or (s["pB"][1] != pB_before[1])
        )

        # --- contact / touch detection ---
        radius_sum = self.agent_radius + self.ball_radius

        def try_contact(p_agent, v_agent, who_label: str):
            nonlocal step_touches
            d = s["pball"] - p_agent
            dist = float(np.linalg.norm(d))
            if dist < radius_sum:
                # This *step* counts as a touch for 'who_label'
                step_touches += 1
                s["touches"] = s.get("touches", 0) + 1

                vel_norm = np.linalg.norm(v_agent)
                if vel_norm > 1e-6:
                    dir_imp = v_agent / (vel_norm + 1e-12)
                else:
                    if dist > 1e-6:
                        dir_imp = d / (dist + 1e-12)
                    else:
                        dir_imp = np.array([0.0, 1.0], dtype=np.float32)
                impulse = 0.8 * dir_imp
                s["vball"] += impulse
                n = d / (dist + 1e-6) if dist > 0.0 else dir_imp
                s["pball"] = p_agent + n * (self.agent_radius + self.ball_radius + 1e-3)
                s["last_touch"] = who_label  # record last player to touch

        try_contact(s["pA"], s["vA"], "A")
        try_contact(s["pB"], s["vB"], "B")

        s["vA"] *= self.friction_agent
        s["vB"] *= self.friction_agent
        s["vball"] *= self.friction_ball

        # Bounce off world bounds (reflect velocity with restitution)
        xmin = self.region_a.xmin - self.field_padding
        xmax = self.region_b.xmax + self.field_padding
        ymin = min(self.region_a.ymin, self.region_b.ymin) - self.field_padding
        ymax = max(self.region_a.ymax, self.region_b.ymax) + self.field_padding

        # X bounds
        if s["pball"][0] - self.ball_radius < xmin:
            s["pball"][0] = xmin + self.ball_radius
            s["vball"][0] = -s["vball"][0] * self.restitution
        elif s["pball"][0] + self.ball_radius > xmax:
            s["pball"][0] = xmax - self.ball_radius
            s["vball"][0] = -s["vball"][0] * self.restitution

        # Y bounds
        if s["pball"][1] - self.ball_radius < ymin:
            s["pball"][1] = ymin + self.ball_radius
            s["vball"][1] = -s["vball"][1] * self.restitution
        elif s["pball"][1] + self.ball_radius > ymax:
            in_mouth = (self.goal_xmin <= s["pball"][0] <= self.goal_xmax)
            if not in_mouth:
                s["pball"][1] = ymax - self.ball_radius
                s["vball"][1] = -s["vball"][1] * self.restitution

        rad = self.ball_radius
        if pball_before[1] >= self.region_b.ymax and s["pball"][1] >= self.region_b.ymax:
            # Left back wall at x = goal_xmin
            if (pball_before[0] + rad <= self.goal_xmin) and (s["pball"][0] + rad > self.goal_xmin):
                s["pball"][0] = self.goal_xmin - rad
                s["vball"][0] = -abs(s["vball"][0]) * self.restitution

            # Right back wall at x = goal_xmax
            if (pball_before[0] - rad >= self.goal_xmax) and (s["pball"][0] - rad < self.goal_xmax):
                s["pball"][0] = self.goal_xmax + rad
                s["vball"][0] = abs(s["vball"][0]) * self.restitution

        # enforce max speed for ball after impulses and bounces
        s["vball"] = cap(s["vball"], self.max_speed_ball)

        pass_line_x = self.pass_line_x
        crossed_pass_line = (
            (pball_before[0] < pass_line_x) and
            (s["pball"][0] >= pass_line_x)
        )
        if crossed_pass_line:
            step_passes += 1
            s["passes"] = s.get("passes", 0) + 1

        # Termination & reward
        terminated = False
        reward = -0.01

        # Detect crossing the goal line inside the posts FROM THE FIELD SIDE (front)
        crossed_goal_line = (
            (pball_before[1] < self.region_b.ymax) and  # previously in front
            (s["pball"][1] >= self.region_b.ymax) and   # now at/behind the line
            (self.goal_xmin <= s["pball"][0] <= self.goal_xmax) and
            (self.region_b.xmin <= s["pball"][0] <= self.region_b.xmax)
        )

        # Only count as a goal if last touch was by B
        is_goal = bool(crossed_goal_line and (s.get("last_touch", "B") == "B"))

        if is_goal:
            reward = 10.0
            terminated = True
        
        terminated_back_cross = False
        if self.ball_spawn_region == "A-PASS":
            mid_x = self.pass_line_x
            if (pball_before[0] > mid_x) and (s["pball"][0] <= mid_x):
                terminated_back_cross = True
                terminated = True

        s["steps"] += 1
        truncated = (s["steps"] >= self.max_steps) and not terminated

        obs = self._get_obs()
        info_after = self._get_info()
        info_after["crossed_goal_mouth"] = bool(crossed_goal_line)
        info_after["scored_goal"] = bool(is_goal)
        info_after["bumped_A"] = bumped_A
        info_after["bumped_B"] = bumped_B
        info_after["touches"] = int(s.get("touches", 0))      # cumulative over episode
        info_after["touches_step"] = int(step_touches)        # touches this step
        info_after["passes"] = int(s.get("passes", 0))
        info_after["passes_step"] = int(step_passes)
        info_after["terminated_back_cross"] = bool(terminated_back_cross)
        info["after"] = info_after

        return obs, reward, terminated, truncated, info

    def _dist_to_goal_mouth(self, bx: float, by: float) -> float:
        # vertical gap (only if below the line)
        dy = max(0.0, self.goal_y - float(by))

        # horizontal gap to the segment (0 if within posts)
        if self.goal_xmin <= float(bx) <= self.goal_xmax:
            dx = 0.0
        else:
            dx = min(abs(float(bx) - self.goal_xmin), abs(float(bx) - self.goal_xmax))

        # Euclidean distance in the forward half-space
        return float(math.hypot(dx, dy))

    def _relative(self, ref_p: np.ndarray, ref_v: np.ndarray, obj_p: np.ndarray, obj_v: np.ndarray):
        # no orientation; relative = simple difference in world frame
        rel_p = obj_p - ref_p
        rel_v = obj_v - ref_v
        return rel_p, rel_v

    def _get_obs(self):
        s = self.state
        if self.centralized:
            return np.array([
                *s["pA"], *s["pB"], *s["vA"], *s["vB"], *s["pball"], *s["vball"]
            ], dtype=np.float32)
        else:
            p_rel_ball_A, v_rel_ball_A = self._relative(s["pA"], s["vA"], s["pball"], s["vball"])
            p_rel_B_from_A, v_rel_B_from_A = self._relative(s["pA"], s["vA"], s["pB"], s["vB"])

            p_rel_ball_B, v_rel_ball_B = self._relative(s["pB"], s["vB"], s["pball"], s["vball"])
            p_rel_A_from_B, v_rel_A_from_B = self._relative(s["pB"], s["vB"], s["pA"], s["vA"])

            oA = np.array([*s["pA"], *s["vA"],
                           *p_rel_ball_A, *v_rel_ball_A,
                           *p_rel_B_from_A, *v_rel_B_from_A], dtype=np.float32)
            oB = np.array([*s["pB"], *s["vB"],
                           *p_rel_ball_B, *v_rel_ball_B,
                           *p_rel_A_from_B, *v_rel_A_from_B], dtype=np.float32)
            return (oA, oB)
    
    def _get_info(self):
        s = self.state
        dist_B_to_ball = float(np.linalg.norm(s["pB"] - s["pball"]))
        info = {
            "ball_x": float(s["pball"][0]),
            "ball_y": float(s["pball"][1]),
            "dist_ball_to_goal": self._dist_to_goal_mouth(s["pball"][0], s["pball"][1]),
            "dist_A_to_ball": float(np.linalg.norm(s["pA"] - s["pball"])),
            "dist_B_to_ball": dist_B_to_ball,
            "passed_regions": bool(self.region_b.contains_point(s["pball"])),
            "speed_ball": float(np.linalg.norm(s["vball"])),
            "last_touch": s.get("last_touch", "A"),
            "touches": int(s.get("touches", 0)),
            "passes": int(s.get("passes", 0)),
        }
        return info


    def render(self, ax=None):
        """
        If ax is None: behave like before (manage self.fig/self.ax).
        If ax is provided: draw state onto that axes and do NOT show/flush.
        """
        s = self.state

        external_ax = ax is not None

        if not external_ax:
            if self.fig is None or self.ax is None:
                self.fig, self.ax = plt.subplots(figsize=(6.5, 4))
            ax = self.ax

        ax.clear()

        xmin = self.region_a.xmin - self.field_padding
        xmax = self.region_b.xmax + self.field_padding
        ymin = min(self.region_a.ymin, self.region_b.ymin) - self.field_padding
        ymax = max(self.region_a.ymax, self.region_b.ymax) + self.field_padding

        # Regions
        ax.add_patch(plt.Rectangle((self.region_a.xmin, self.region_a.ymin),
                       self.region_a.xmax - self.region_a.xmin,
                       self.region_a.ymax - self.region_a.ymin,
                       fill=False, linewidth=2))
        ax.text((self.region_a.xmin+self.region_a.xmax)/2, self.region_a.ymax+0.05,
                "Region A", ha="center", va="bottom")

        ax.add_patch(plt.Rectangle((self.region_b.xmin, self.region_b.ymin),
                       self.region_b.xmax - self.region_b.xmin,
                       self.region_b.ymax - self.region_b.ymin,
                       fill=False, linewidth=2))
        ax.text((self.region_b.xmin+self.region_b.xmax)/2, self.region_b.ymax+0.05,
                "Region B", ha="center", va="bottom")

        # Goal line + posts
        ax.plot([self.goal_xmin, self.goal_xmax],
                [self.region_b.ymax, self.region_b.ymax],
                linewidth=6)

        back_depth = 0.4
        ax.plot([self.goal_xmin, self.goal_xmin],
                [self.region_b.ymax, self.region_b.ymax + back_depth],
                linewidth=6, color='black')
        ax.plot([self.goal_xmax, self.goal_xmax],
                [self.region_b.ymax, self.region_b.ymax + back_depth],
                linewidth=6, color='black')

        # Agents
        def draw_agent(p, v, label):
            circ = plt.Circle((p[0], p[1]), self.agent_radius, fill=False, linewidth=2)
            ax.add_patch(circ)
            speed = np.linalg.norm(v)
            if speed > 1e-6:
                head = (v / (speed + 1e-12)) * (self.agent_radius * 1.5)
            else:
                head = np.array([0.0, 1.0]) * (self.agent_radius * 1.5)
            ax.arrow(p[0], p[1], head[0], head[1],
                     head_width=0.02, length_includes_head=True)
            ax.text(p[0], p[1]-0.08, label, ha="center", va="top")

        draw_agent(s["pA"], s["vA"], "A")
        draw_agent(s["pB"], s["vB"], "B")

        # Ball
        ball = plt.Circle((s["pball"][0], s["pball"][1]), self.ball_radius, fill=True)
        ax.add_patch(ball)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Pass-and-Score: Episode State")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # HUD text
        def fmt_item(x):
            try:
                it = list(x)
                return "[" + ", ".join(f"{float(v):.2f}" for v in it) + "]"
            except Exception:
                return f"{float(x):.2f}"

        text_lines = [
            f"steps: {int(s.get('steps', 0))}",
            f"last_touch: {s.get('last_touch', 'A')}",
            f"pA: {fmt_item(s['pA'])}",
            f"vA: {fmt_item(s['vA'])}",
            f"pB: {fmt_item(s['pB'])}",
            f"vB: {fmt_item(s['vB'])}",
            f"pball: {fmt_item(s['pball'])}",
            f"vball: {fmt_item(s['vball'])}",
        ]
        state_text = "\n".join(text_lines)

        ax.text(
            0.98,
            0.98,
            state_text,
            transform=ax.transAxes,
            fontsize=8,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"),
        )

        if not external_ax:
            # standalone mode: draw/update
            self.fig.canvas.draw()
            plt.pause(0.001)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
        self.fig, self.ax = None, None


class PassAndScoreGym(gym.Env):
    metadata = PassAndScoreEnv.metadata

    def __init__(self):
        super().__init__()
        self.env = PassAndScoreEnv(centralized=True)
        self.action_space = gym_spaces.MultiDiscrete([5, 5])
        self.observation_space = gym_spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()