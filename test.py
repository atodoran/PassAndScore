import matplotlib.pyplot as plt
import numpy as np
from env import PassAndScoreEnv

def towards_ball_policy(obs):
    p_agent = obs[0:2]
    theta_agent = obs[8]
    p_ball = obs[10:12]
    direction = p_ball - p_agent
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    # Action has to be 0: move forward, 1: move backward, 2: turn left, 3: turn right, 4: stay still
    angle_to_ball = np.arctan2(direction[1], direction[0])
    angle_diff = angle_to_ball - theta_agent
    # Normalize angle_diff to [-pi, pi]
    angle_to_ball = angle_diff % (2 * np.pi)
    action = np.array([0, 4])  # default: stay still
    if np.random.rand() < 0.5:
        if angle_to_ball > np.pi:
            action[0] = 3  # turn right
        else:
            action[0] = 2  # turn left
    return action

env = PassAndScoreEnv(centralized=True, seed=np.random.randint(0, 10000))
obs, info = env.reset()

plt.ion()  # interactive mode
fps = env.metadata.get("render_fps", 30)
dt = 1.0 / fps

for t in range(300):
    env.render()                     # draw current frame
    plt.pause(dt)                    # let the GUI update (non-blocking)
    # action = np.array([env.single_action_space.sample(),
    #                 env.single_action_space.sample()])  # random joint action
    action = towards_ball_policy(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break

plt.ioff()
env.close()