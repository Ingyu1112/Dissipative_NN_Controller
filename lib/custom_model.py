import os
import gymnasium as gym
from gymnasium.envs.mujoco.inverted_pendulum_v4 import InvertedPendulumEnv
from gymnasium import spaces
import numpy as np

ENV_ID = "CustomInvertedPendulum-v0"
ENTRY = "pybullet_envs.bullet.minitaur_gym_env:MinitaurBulletEnv"
LQR_PARAM = 1

class CustomInvertedPendulumEnv(InvertedPendulumEnv):
    def __init__(self, **kwargs):
        # Enforce the modified xml file (+ torque)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        xml_path = os.path.join(parent_dir, "custom_xml", "custom_inverted_pendulum.xml")

        from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_path,
            frame_skip=2,
            observation_space=observation_space,
            **kwargs,
        )

    def step(self, action):
        # random disturban in 30 %
        if np.random.rand() < 0.3:
            disturbance_torque = np.random.uniform(-25.0, 25.0)
            disturbance_linear = np.random.uniform(-25.0, 25.0)
            self.data.qfrc_applied[0] = disturbance_linear
            self.data.qfrc_applied[1] = disturbance_torque
        else:
            self.data.qfrc_applied[0] = 0.0
            self.data.qfrc_applied[1] = 0.0
        # one step value from the original step function
        obs, reward, is_done, is_trunc, info = super().step(action)

        pos, ang, vel, ang_vel = obs

        # LQR parameters
        q_lqr = 10
        v_lqr = 0.1
        r_lqr = 0.01

        cost_q = q_lqr * (pos ** 2 + ang ** 2)
        cost_v = v_lqr * (vel ** 2 + ang_vel ** 2)
        cost_u = r_lqr * (action[0] ** 2 + action[1] ** 2)

        # LQR penalty
        lqr_penalty = -LQR_PARAM * (cost_q + cost_v + cost_u)

        custom_reward = reward + lqr_penalty

        info["lqr_penalty"] = lqr_penalty
        info["cost_q"] = cost_q
        info["cost_v"] = cost_v
        info["cost_u"] = cost_u

        return obs, custom_reward, is_done, is_trunc, info

def register_env():
    # register environment in gymnasium registry, not gym's
    gym.register(
        id=ENV_ID, 
        entry_point="lib.custom_model:CustomInvertedPendulumEnv",
        max_episode_steps=1000, reward_threshold=15.0,
    )

if __name__ == "__main__":
    register_env()
    env = gym.make(ENV_ID)
    print("Action Space:", env.action_space)
