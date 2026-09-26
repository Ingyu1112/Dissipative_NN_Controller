import os
import gymnasium as gym
from gymnasium.envs.mujoco.inverted_pendulum_v4 import InvertedPendulumEnv
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np
import mujoco

ENV_ID = "CustomCartPoleEnv-v0"
NETWORK_ID = "CustomCartPoleEnv-v0"
ENTRY_POINT="lib.custom_model:CustomCartPoleEnv"
LQR_PARAM = 1
DT = 0.005
MAX_ENV_LEN = 1000 * 0.02 / DT

class CustomCartPoleEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, 
        dt : float = DT,
        rho : float = 1000.0,
        g : float = 9.81,
        max_input: float = 1.0,
        max_force: float = 300.0,
        max_torque: float = 30.0,
        q_weight: float = 10,
        v_weight: float = .1,
        u_weight: float = 0.01,
        lqr_penalty: float = 1,
        episode_len: int = MAX_ENV_LEN,
        terminal_pos: float = 0.8,
        terminal_ang: float = 0.2,
        cart_damping: float = 1.0,
        pole_damping: float = 1.0,
        use_is_done: bool = True,
        act_clip: bool = True,
        apply_dist: str = "sinc",
        xml_file="passive_inverted_pendulum.xml", 
        render_mode = None
    ):
        super().__init__()
        self.render_mode = render_mode
        self.dt = dt
        self.rho = rho
        self.g = g

        self.metadata = dict(self.metadata)
        self.metadata["render_fps"] = int(round(1.0 / self.dt))

        self.max_input = max_input
        self.max_force = max_force
        self.max_torque = max_torque

        self.q_weight = q_weight
        self.v_weight = v_weight
        self.u_weight = u_weight
        self.lqr_penalty = lqr_penalty
        
        self.episode_len = episode_len
        self.terminal_pos = terminal_pos
        self.terminal_ang = terminal_ang

        self.cart_damping = cart_damping
        self.pole_damping = pole_damping

        self.use_is_done = use_is_done
        self.act_clip = act_clip
        self.apply_dist = apply_dist

        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        xml_path = os.path.join(parent_dir, "custom_xml", xml_file)

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        cart_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cart")
        pole_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cpole")

        self.cart_radius = self.model.geom_size[cart_id][0]
        self.cart_length = self.model.geom_size[cart_id][1]*2.0
        self.cart_mass = self._cal_mass(self.cart_radius, self.cart_length, self.rho)

        self.pole_radius = self.model.geom_size[pole_id][0]
        self.pole_length = self.model.geom_size[pole_id][1]*2.0
        self.pole_mass = self._cal_mass(self.pole_radius, self.pole_length, self.rho)
        self.pole_inertia = self._cal_inertia(self.pole_mass, self.pole_radius, self.pole_length)

        # high = np.array([np.inf, np.inf, np.inf, np.inf], dtype = np.float32)
        self.observation_space = spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape = (4,),
            dtype = np.float32
        )
        self.action_space = spaces.Box(
            low = -self.max_input,
            high = self.max_input,
            shape = (2,),
            dtype = np.float32
        )

        self.state = None

    def _cal_mass(self, radius, length, rho):
        return np.pi*radius**2*length*rho

    def _cal_inertia(self, mass, radius, length):
        return 1/12*mass*length**2 + 1/4*mass*radius**2

    def render(self):
        if self.render_mode is None:
            return

        self.data.qpos[0] = self.state[0]
        self.data.qpos[1] = self.state[1]
        self.data.qvel[0] = self.state[2]
        self.data.qvel[1] = self.state[3]

        mujoco.mj_forward(self.model, self.data)

        if not hasattr(self, 'renderer'):
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        
        self.renderer.update_scene(self.data, camera=-1)
        return self.renderer.render()

    def close(self):
        if hasattr(self, 'renderer'):
            self.renderer.close()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "state" in options:
            state = np.asarray(options["state"], dtype=np.float32)
        else:
            init_qpos = np.zeros(self.model.nq, dtype=np.float32)
            init_qvel = np.zeros(self.model.nv, dtype=np.float32)

            qpos = init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
            qvel = init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
            state = np.concatenate([qpos, qvel]).astype(np.float32)
        self.state = state
        self.t = 0.0     
        self.steps = 0   
        self.dist_mag = np.asarray([self.np_random.uniform(-3, 3), 
                                    self.np_random.uniform(-3, 3)], dtype=np.float32)
        return self.state.copy(), self._info(np.zeros(2, dtype=np.float32))

    def step(self, action):
        state = self.state
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -self.max_input, self.max_input) if self.act_clip else action
        u = np.asarray([action[0]*self.max_force, action[1]*self.max_torque], dtype=np.float32)

        next_state = self._rk_step(self.state, u + self._disturbance()) 
        self.t += self.dt
        self.steps += 1
        is_trunc = self.steps >= self.episode_len
        
        is_done = (abs(next_state[1]) >= self.terminal_ang or abs(next_state[0]) >= self.terminal_pos) if self.use_is_done else False
            
        cost_val, lqr_cost = self._qsr_reward(next_state, action)
        reward = (1 - self.lqr_penalty * cost_val) * self.dt / 0.02
        self.state = next_state
        info = self._info(u)
        info.update(lqr_cost)            

        return next_state, reward, is_done, is_trunc, info

    def _disturbance(self):
        dist_force = np.zeros(self.action_space.shape)
        if self.apply_dist == 'sinc':
            dist_force = self.dist_mag * np.sin(2.0 * np.pi * self.t) / max(self.t,self.dt)

        return dist_force.astype(np.float32)

    def _qsr_reward(self, state, action):
        x, theta, x_dot, theta_dot = state

        cost_q = self.q_weight * (x**2 + theta**2)
        cost_v = self.v_weight * (x_dot**2 + theta_dot**2)
        cost_u = self.u_weight * (action[0]**2 + action[1]**2)

        return np.float32(cost_q + cost_v + cost_u), {
            "cost_q": cost_q,
            "cost_v": cost_v,
            "cost_u": cost_u
        }

    def _dynamics(self, state, u):
        x, theta, x_dot, theta_dot = state
        f, tau = u

        l_c = self.pole_length / 2.0
        
        M11 = self.cart_mass + self.pole_mass
        M12 = self.pole_mass * l_c * np.cos(theta)
        M22 = self.pole_mass * l_c**2 + self.pole_inertia
        det = M11*M22 - M12**2

        force = f + self.pole_mass * l_c * theta_dot**2 * np.sin(theta) - self.cart_damping * x_dot
        torque = tau + self.pole_mass * self.g * l_c * np.sin(theta) - self.pole_damping * theta_dot

        ddx = (force * M22 - torque * M12) / det
        ddt = (-force * M12 + torque * M11) / det
        return np.asarray([x_dot, theta_dot, ddx, ddt]).astype(np.float32)

    def _rk_step(self, state, u):
        state = np.asarray(state, dtype=np.float32)
        u = np.asarray(u, dtype=np.float32)

        k1 = self._dynamics(state, u)
        k2 = self._dynamics(state + 0.5*self.dt*k1, u)
        k3 = self._dynamics(state + 0.5*self.dt*k2, u)
        k4 = self._dynamics(state + self.dt*k3, u)

        next_state = state + self.dt/6*(k1+2*k2+2*k3+k4)
        return next_state.astype(np.float32)

    def _info(self, u):
        q = self.state[:2]
        v = self.state[2:]
        return {
            "q": q.copy(),
            "v": v.copy(),
            "u": np.asarray(u, dtype=np.float32).copy(),
            "state_norm": float(np.linalg.norm(self.state)),
            "action_norm": float(np.linalg.norm(u)),
        }

class PassiveInvertedPendulumEnv(InvertedPendulumEnv):
    # Passive inverted pendulum model
    # input:  [F tau]
    # output: [p theta dp dtheta]
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }
    def __init__(self, use_is_done=True, **kwargs):
        # Enforce the modified xml file (+ torque)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        xml_path = os.path.join(parent_dir, "custom_xml", "passive_inverted_pendulum.xml")

        from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        
        self.current_dist = "l2"
        self.lqr_parameter = LQR_PARAM

        self.use_is_done = use_is_done

        MujocoEnv.__init__(
            self,
            xml_path,
            frame_skip=1,
            observation_space=observation_space,
            **kwargs,
        )
        
    def reset_model(self):
        obs = super().reset_model()
        self.data.time = 0.0
        
        self.dist_mag = [self.np_random.uniform(-10.0, 10.0), self.np_random.uniform(-6.0, 6.0)]

        return obs

    def do_simulation(self, ctrl, n_frames):
        # apply agent's action to MuJoCo ctrl
        self.data.ctrl[:] = ctrl

        # simulate for frame_skip(n_frames)
        for _ in range(n_frames):
            if self.current_dist == "random":
                if self.np_random.rand() < 0.3:
                    self.data.qfrc_applied[0] = self.np_random.uniform(-25.0, 25.0)
                    self.data.qfrc_applied[1] = self.np_random.uniform(-25.0, 25.0)
                else:
                    self.data.qfrc_applied[0] = 0.0
                    self.data.qfrc_applied[1] = 0.0
            
            elif self.current_dist == "l2":
                t = self.data.time
                self.data.qfrc_applied[0] = self.dist_mag[0] * np.sin(2.0 * np.pi * t) / max(t,0.02)
                self.data.qfrc_applied[1] = self.dist_mag[1] * np.sin(2.0 * np.pi * t) / max(t,0.02)
                
            elif self.current_dist == "none":
                t = self.data.time
                self.data.qfrc_applied[0] = 0.0
                self.data.qfrc_applied[1] = 0.0

            # phsics engine 1step proceeds (qfrc_applied=0 after this function)
            mujoco.mj_step(self.model, self.data)

    def step(self, action):
        # self.current_dist = dist    # apply disturbance in do_simulation

        # one step value from the original step function
        obs, reward, is_done, is_trunc, info = super().step(action)

        if not self.use_is_done:
            is_done = False

        pos, ang, vel, ang_vel = obs

        # LQR parameters
        q_lqr = 10
        v_lqr = 0.1
        r_lqr = 0.01

        cost_q = q_lqr * (pos ** 2 + ang ** 2)
        cost_v = v_lqr * (vel ** 2 + ang_vel ** 2)
        cost_u = r_lqr * (action[0] ** 2 + action[1] ** 2)

        # LQR penalty
        lqr_penalty = -self.lqr_parameter * (cost_q + cost_v + cost_u)

        custom_reward = reward + lqr_penalty

        info["lqr_penalty"] = lqr_penalty
        info["cost_q"] = cost_q
        info["cost_v"] = cost_v
        info["cost_u"] = cost_u

        return obs, custom_reward, is_done, is_trunc, info

class NetworkInvertedPendulumEnv(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }
    def __init__(self, num_pendulums=4, xml_file="multi_pendulum.xml", **kwargs):
        # Enforce the modified xml file (+ torque)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        xml_path = os.path.join(parent_dir, "custom_xml", xml_file)

        self.num_pendulums = num_pendulums
        self.dist = []
        for _ in range(self.num_pendulums):
            self.dist.append(np.random.uniform(-25.0, 25.0))

        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4*self.num_pendulums,), dtype=np.float64)
        
        super().__init__(
            xml_path,
            frame_skip=2,
            observation_space=observation_space,
            **kwargs,
        )

    def _get_obs(self):
        # change the observation order (q, q, v, v) -> (q, v, q, v)
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        qpos_reshaped = qpos.reshape(self.num_pendulums, 2)
        qvel_reshaped = qvel.reshape(self.num_pendulums, 2)

        obs_matrix = np.concatenate([qpos_reshaped, qvel_reshaped], axis=1)

        return obs_matrix.ravel()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)

        self.set_state(qpos, qvel)
        self.data.time = 0.0

        return self._get_obs()

    def step(self, action, dist='l2'):
        # random disturban in 30 %
        if dist == "random":
            if np.random.rand() < 0.3:
                for i in range(self.num_pendulums):
                    self.data.qfrc_applied[i * 2] = np.random.uniform(-25.0, 25.0)
            else:
                for i in range(self.num_pendulums):
                    self.data.qfrc_applied[i * 2] = 0.0
        elif dist == "l2":
            t = self.data.time
            for i in range(self.num_pendulums):
                self.data.qfrc_applied[i * 2] = self.dist[i] * np.sin(2.0*np.pi*t) / max(t,0.02)

        # set up observation
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()   # next observation from RK 4

        obs_matrix = obs.reshape(self.num_pendulums, 4)
        pos = obs_matrix[:, 0]
        ang = obs_matrix[:, 1]
        vel = obs_matrix[:, 2]
        ang_vel = obs_matrix[:, 3]

        # network interconnection 4 - 1 - 2 - 3
        self.data.qfrc_applied[0] = -(2*pos[0] - pos[3] - pos[1])
        self.data.qfrc_applied[2] = -(2*pos[1] - pos[0] - pos[2])
        self.data.qfrc_applied[4] = -(pos[2] - pos[1])
        self.data.qfrc_applied[6] = -(pos[3] - pos[0])

        q_val = np.concatenate([pos, ang])
        v_val = np.concatenate([vel, ang_vel])

        # LQR parameters
        q_lqr = 10.0
        v_lqr = 0.1
        r_lqr = 0.01

        cost_q = q_lqr * (q_val.T@q_val)
        cost_v = v_lqr * (v_val.T@v_val)
        cost_u = r_lqr * action.T@action

        # LQR penalty
        lqr_penalty = -LQR_PARAM * (cost_q + cost_v + cost_u)

        # set up reward
        reward = 1.0
        custom_reward = 1.0 + lqr_penalty

        # set up termination
        is_done = bool(not np.isfinite(obs).all() or np.any(np.abs(ang) > 0.2))
        is_trunc = False

        # set up information for debugging
        info = {
            "lqr_penalty": lqr_penalty,
            "cost_q": cost_q,
            "cost_v": cost_v,
            "cost_u": cost_u
        }

        return obs, custom_reward, is_done, is_trunc, info

def register_env(id: str=NETWORK_ID, 
                entry_point: str=ENTRY_POINT):
    # register environment in gymnasium registry, not gym's
    gym.register(
        id=id, 
        entry_point=entry_point,
        max_episode_steps=MAX_ENV_LEN*10, reward_threshold=15.0,
    )

if __name__ == "__main__":
    register_env(id=NETWORK_ID,
        entry_point=ENTRY_POINT)
    env = gym.make(NETWORK_ID)
    state, _ = env.reset()
    next_state, reward, is_done, is_trunc, _ = env.step(env.action_space.sample())
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)
