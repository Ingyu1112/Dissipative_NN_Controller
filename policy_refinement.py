import os
os.environ["MUJOCO_GL"] = "egl" 
os.environ["MOSEKLM_LICENSE_FILE"] = "/hpc/group/bridgemanlab/ij40/mosek/mosek.lic"

import cvxpy as cp
from cvxpy.error import SolverError
import numpy as np

import argparse
import torch
import gymnasium as gym

from lib import model, common, custom_model
from lib.certificate import *
from lib.draw_results import *

import scipy
from scipy.sparse import bmat, block_array, coo_array
from types import SimpleNamespace
from typing import List

import copy

L2_DIST = 'sinc'     # [sinc, exp, inv_prop]
DT = 0.001
CUTOFF = 4
VIDEO_SAVE = True

np.set_printoptions(precision=4, suppress=True, linewidth=200)

class ControllerSynthesis():
    def __init__(self,
        env : gym.Env,
        directory : str,
        model_name : str,
        device : str = 'cuda',
        linear_gear : int = 300,
        rotational_gear : int = 30,
        struct : str = 'linear',
        verbose : bool = True
    ) -> None:
    
        self.env = env
        self.directory = directory
        self.model = model_name
        self.device = device
        self.linear_gear = linear_gear
        self.rotational_gear = rotational_gear
        self.struct = struct
        self.verbose = verbose

    def get_data(self) -> None:
        self.data_path = os.path.join(os.getcwd(), self.directory, self.model)
        actor_state = torch.load(self.data_path, map_location=torch.device(self.device),weights_only=True)
        
        if self.struct == 'linear':
            self.actor = model.DDPGActorLinear(self.env.observation_space.shape[0],
                                               self.env.action_space.shape[0],
                                               bias=False)
            self.actor.load_state_dict(actor_state)
            self.W_linear_train = -np.diag([self.linear_gear, self.rotational_gear])@self.actor.net[0].weight.detach().cpu().numpy()
            if self.verbose: 
                print("  Trained linear controller:")
                print("    " + str(self.W_linear_train).replace("\n", "\n    "))
        elif self.struct == 'resnet':
            layer = []
            for layer_name, tensor in list(actor_state.items())[:-2]:
                layer.append(actor_state[layer_name].shape[0])
            self.actor = model.DDPGActorResidual(self.env.observation_space.shape[0], 
                                                 self.env.action_space.shape[0], 
                                                 layer, bias=False)
            self.actor.load_state_dict(actor_state)
            self.W_residual_train = -np.diag([self.linear_gear, self.rotational_gear])@self.actor.residual.weight.detach().cpu().numpy()
            if self.verbose: 
                print("  Trained residual of resnet:")
                print("    " + str(self.W_residual_train).replace("\n", "\n    ")) 

            layer_depth = (len(self.actor.net)+1) // 2
            W_layer_train = [self.actor.net[2*j].weight.detach().cpu().numpy() for j in range(layer_depth)]

            W_blocks = [[None for _ in range(layer_depth)] for _ in range(layer_depth)]
            W_blocks[0][-1] = W_layer_train[-1]
            for i in range(1,layer_depth):
                W_blocks[i][i-1] = W_layer_train[i-1]
            self.W_linear_block = bmat(W_blocks).toarray()
            self.W_linear_block[:2,-10:] = np.diag([self.linear_gear, self.rotational_gear])@self.W_linear_block[:2,-10:]
        else:
            raise ValueError(f"Unsupported actor structure: {self.struct}")

    def roll_out(self,
        dt : float = 0.001,
        l2_dist : str = 'sinc',
        linear_mag : float = 10,
        rotational_mag : float = 6,
        initial_state : np.ndarray | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        float,
        int,
        np.ndarray
    ]:
        if initial_state is None:
            initial_state = np.zeros(4, dtype=np.float32)

        if not hasattr(self, "actor"):
            raise RuntimeError(
                "get_data() must be called before roll_out()."
            )

        self.env.reset()

        self.env.unwrapped.dist_mag = np.asarray([linear_mag, rotational_mag], dtype=np.float32)
        self.env.unwrapped.dt = dt
        self.env.unwrapped.episode_len = 1000 * 0.02 / dt

        self.env.unwrapped.state = np.asarray(initial_state, dtype=np.float32).copy()
        obs = np.array(self.env.unwrapped.state, copy=True)
        if self.verbose: print("  initial state: ", self.env.unwrapped.state)

        x_roll, a_roll, disturbance = [], [], []
        total_reward = 0.0
        steps = 0

        while True:
            t = self.env.unwrapped.t
            obs_t = torch.Tensor(obs)

            with torch.no_grad():
                if hasattr(self, "actor_stable"):
                    action = np.diag([self.linear_gear, self.rotational_gear])@self.actor_stable(obs_t).detach().cpu().numpy()
                else:
                    action = np.diag([self.linear_gear, self.rotational_gear])@self.actor(obs_t).detach().cpu().numpy()

            a_roll.append(action.copy())

            if l2_dist == "sinc":
                disturbance += [self.env.unwrapped.dist_mag * np.sin(2.0 * np.pi * t) / max(t,dt)]
            elif l2_dist == "exp":
                disturbance += [self.env.unwrapped.dist_mag * t**2 / np.exp(t)]
            elif l2_dist == "inv_prop":
                disturbance += [self.env.unwrapped.dist_mag / (t+1)]
            else:
                raise ValueError(
                    f"Unsupported disturbance type: {l2_dist}, choose in (sinc, exp, inv_prop)"
                )

            action += disturbance[steps]

            obs, reward, is_done, is_trunc, info = self.env.step(action)
            total_reward += float(reward)
            x_roll.append(np.array(obs, copy=True))
            steps += 1

            if is_done or is_trunc:
                break

        x_roll = np.array(x_roll)
        a_roll = np.array(a_roll)
        disturbance = np.array(disturbance)

        return x_roll, a_roll, total_reward, steps, disturbance

    def policy_refinement(self, 
        m1, m2, e,
        K, D, q_plant, s_plant, r_plant,
        q_linear = None, s_linear = None, r_linear = None,
        q_activation = None, s_activation = None, r_activation = None,
        q_network = None, pre_constraint = None,
        W = None, X = None, r1 = 1, r2 = 1, e1 = 2000, ek = 1, ed = 1,
        solver = 'MOSEK', tol = -1e-7
    ):
        self.actor_stable = copy.deepcopy(self.actor)
        self.actor_stable.requires_grad_(False)

        if self.struct == 'linear':
                 
            cost = cp.norm(K,"fro") \
                 + cp.norm(D,"fro")
            constraints = pre_constraint["plant"]

            prob = cp.Problem(cp.Minimize(cost),
                            constraints)
                            
            prob.solve(solver=solver, verbose=False)
            print("  feasibility problem result: ", prob.status)
            self.W_linear_stable = np.concatenate([K.value, D.value], axis = 1)

            if self.verbose: 
                print("  Stable linear controller:")
                print("    " + str(np.hstack((self.W_linear_train, self.W_linear_stable))).replace("\n", "\n    "))

            self.actor_stable.net[0].weight.copy_(
                torch.tensor(-np.diag([1/self.linear_gear, 1/self.rotational_gear])@self.W_linear_stable,
                    dtype=self.actor_stable.net[0].weight.dtype,
                    device=self.actor_stable.net[0].weight.device
                )
            )

            if self.verbose: 
                print("  ", self.actor_stable.net[0].weight)

        if self.struct == 'resnet':
            kyp_linear = (
                - r_linear 
                - s_linear.T@self.W_linear_block - self.W_linear_block.T@s_linear 
                - self.W_linear_block.T@q_linear@self.W_linear_block
            )

            cost = cp.norm(self.W_residual_train[:,:2]-K,"fro") \
                 + cp.norm(self.W_residual_train[:,2:]-D,"fro")

            constraints = [kyp_linear << 0] \
                + pre_constraint["network"] \
                + pre_constraint["activation"] \
                + pre_constraint["K"] \
                + pre_constraint["D"]

            prob = cp.Problem(cp.Minimize(cost), constraints)

            try:
                prob.solve(solver=solver, verbose=False)
                print("  feasibility problem result: ", prob.status)

                success = prob.status in ["optimal", "optimal_inaccurate"]

            except SolverError as err:
                success = False

            if success:
                self.W_residual_stable = np.concatenate([K.value, D.value], axis=1)
                if self.verbose: 
                    print("  Stable residual of resnet:")
                    print("    " + str(np.hstack((self.W_residual_train, self.W_residual_stable))).replace("\n", "\n    ")) 
                self.actor_stable.residual.weight.copy_(
                    torch.tensor(-np.diag([1/self.linear_gear,1/self.rotational_gear])@self.W_residual_stable, 
                                dtype=self.actor_stable.residual.weight.dtype,
                                device=self.actor_stable.residual.weight.device)
                )
            else:
                print("  Infeasible or Numerical problem (Use ADMM).")

                (q_plant_val, s_plant_val, r_plant_val,
                q_linear_val, s_linear_val, r_linear_val, q_linear_0,
                q_activation_val, s_activation_val, r_activation_val) \
                    = self._feasible_certificate(
                        m1, m2, e, cost, pre_constraint, kyp_linear, q_network,
                        K, D, q_plant, q_linear, q_activation
                    )

                X_val = s_linear_val.T@self.W_linear_block
                Y_val = np.zeros(X_val.shape)

                iteration = 0

                while True:
                    iteration += 1
                    print(f"  - {iteration} - admm iteration")
                    
                    # X, W update
                    cost_xw = e1*cp.norm(self.W_linear_block-W_sdp,"fro") \
                            + r1/2*cp.norm(X - s_linear_val.T@W_sdp + Y_val,"fro")
                    kyp_xw = cp.bmat([
                        [-r_linear_val-(X+X.T),   W_sdp.T@q_linear_0],
                        [q_linear_0@W_sdp,        2*q_linear_0 - q_linear_val]
                    ])
                    constraints_xw = [kyp_xw << np.zeros(kyp_xw.shape)]

                    prob_xw = cp.Problem(cp.Minimize(cost_xw), constraints_xw)
                    prob_xw.solve(solver=solver, verbose=False)

                    if prob_xw.status in ["optimal", "optimal_inaccurate"]:
                        W_val = W_sdp.value
                        X_val = X.value

                    # Q, S, R update
                    cost_qsr = ek*cp.norm(self.W_residual_train[:,:2]-K,"fro") \
                            + ed*cp.norm(self.W_residual_train[:,2:]-D,"fro") \
                            + r2/2*cp.norm(X_val - s_linear.T@W_val + Y_val,"fro")
                    kyp_qsr = cp.bmat([
                        [-r_linear-(X_val+X_val.T), W_val.T@q_linear_0],
                        [q_linear_0@W_val,          2*q_linear_0 - q_linear]
                    ])
                    constraints_qsr = [kyp_qsr << np.zeros(kyp_qsr.shape)] \
                        + pre_constraint["network"] + pre_constraint["plant"] + pre_constraint["linear"] \
                        + pre_constraint["activation"] + pre_constraint["K"] + pre_constraint["D"]
                                        
                    prob_qsr = cp.Problem(cp.Minimize(cost_qsr), constraints_qsr)
                    prob_qsr.solve(solver=solver, verbose=False)

                    if prob_qsr.status in ["optimal", "optimal_inaccurate"]:
                        q_plant_val = q_plant.value
                        s_plant_val = s_plant.value
                        r_plant_val = r_plant.value

                        q_linear_val = q_linear.value
                        s_linear_val = s_linear.value
                        r_linear_val = r_linear.value
                        
                        q_activation_val = q_activation.value
                        s_activation_val = s_activation.value
                        r_activation_val = r_activation.value

                    # Y update
                    Y_val = Y_val + (X_val - s_linear_val.T@W_val)

                    # feasibility check
                    feasible, kyp_check = self._feasible_check(
                        K, D, W_val, q_linear, s_linear, r_linear, pre_constraint
                    )
                    if feasible:
                        self.actor_stable.residual.weight.copy_(
                            torch.tensor(-np.diag([1/self.linear_gear,1/self.rotational_gear])@self.W_residual_stable, 
                                        dtype=self.actor_stable.residual.weight.dtype,
                                        device=self.actor_stable.residual.weight.device)
                        )
                        self.actor_stable.net[0].weight.copy_(
                            torch.tensor(self.W_linear_block_stable[2:22,:4], 
                                        dtype=self.actor_stable.net[0].weight.dtype,
                                        device=self.actor_stable.net[0].weight.device)
                        )
                        self.actor_stable.net[2].weight.copy_(
                            torch.tensor(self.W_linear_block_stable[22:,4:24], 
                                        dtype=self.actor_stable.net[2].weight.dtype,
                                        device=self.actor_stable.net[2].weight.device)
                        )
                        self.actor_stable.net[4].weight.copy_(
                            torch.tensor(np.diag([1/self.linear_gear,1/self.rotational_gear])@self.W_linear_block_stable[:2,-10:], 
                                        dtype=self.actor_stable.net[4].weight.dtype,
                                        device=self.actor_stable.net[4].weight.device)
                        )
                        if self.verbose:
                            print("    kyp feasibility: %.6g" % np.linalg.eigvalsh(kyp_check.value).max())
                            print("    stability feasibility: %.6g" % np.linalg.eigvalsh(q_network.value).max())
                            print("  Stable final layer:")
                            print("    " + str(self.W_linear_block_stable[:2,-10:]).replace("\n", "\n    "))
                            print("  Stable residual of resnet:")
                            print("    " + str(np.hstack((self.W_residual_train, self.W_residual_stable))).replace("\n", "\n    "))
                        break
                    else:
                        print("    MOSEK failed.")

    def _feasible_check(self,
        K, D, W, q_linear, s_linear, r_linear, pre_constraint,
        solver = "MOSEK"
    ):
        kyp_check = - r_linear - s_linear.T@W - W.T@s_linear  - W.T@q_linear@W
        constraints_check = [kyp_check << np.zeros(kyp_check.shape)] \
            + pre_constraint["network"] + pre_constraint["plant"] + pre_constraint["linear"] \
            + pre_constraint["activation"] + pre_constraint["K"] + pre_constraint["D"]

        cost_check = cp.norm(self.W_residual_train[:,:2]-K,"fro") \
                   + cp.norm(self.W_residual_train[:,2:]-D,"fro")
        prob_check = cp.Problem(cp.Minimize(cost_check), constraints_check)
        try:
            prob_check.solve(solver=solver, verbose=False)
            print("  feasibility problem result: ", prob_check.status)

            if prob_check.status in ["optimal", "optimal_inaccurate"]:
                self.W_residual_stable = np.concatenate([K.value, D.value], axis=1)
                self.W_linear_block_stable = W
                return True, kyp_check

        except SolverError as err:
            return False, kyp_check

    def _feasible_certificate(self, 
        m1, m2, e, 
        cost, pre_constraint, kyp_linear, q_network, 
        K, D, q_plant, q_linear, q_activation, 
        solver = 'MOSEK', tol = -1e-7
    ):
        eps = cp.Variable(1)
        constraints = [
            kyp_linear  << 0,
            q_network   << eps*np.eye(q_network.shape[0])
        ] + pre_constraint["plant"] + pre_constraint["linear"] + pre_constraint["activation"] \
            + pre_constraint["K"] + pre_constraint["D"]
            
        cost += eps
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=solver, verbose=False)

        q_plant_v = q_plant.value
        s_plant_v = s_plant.value
        r_plant_v = r_plant.value

        q_linear_v = q_linear.value
        s_linear_v = s_linear.value
        r_linear_v = r_linear.value

        q_linear_0 = (np.floor(min(np.linalg.eig(q_linear_v)[0]))-1) * np.eye(q_linear.shape[0])
        
        q_activation_v = q_activation.value
        s_activation_v = s_activation.value
        r_activation_v = r_activation.value

        return (q_plant_v, s_plant_v, r_plant_v,
                q_linear_v, s_linear_v, r_linear_v, q_linear_0,
                q_activation_v, s_activation_v, r_activation_v)
   
        
def mean_std_norm(traj: list[np.ndarray]):
    traj_np = np.array(traj)
    traj_mean = np.mean(traj_np, axis=0)
    traj_std = np.std(traj_np, axis=0)

    traj_norm = np.array([np.sum(traj_np[i]**2, axis=1) for i in range(len(traj))])
    traj_norm_mean = np.mean(traj_norm, axis=0)
    traj_norm_std = np.std(traj_norm, axis=0)

    return (traj_np, traj_mean, traj_std,
            traj_norm, traj_norm_mean, traj_norm_std)


if __name__ == "__main__":
    args_linear = [
        SimpleNamespace(
            model="best_+975.574_1200000.dat",  
            actor="linear",    
            directory="saves/linear/ddpg_custom_0_linear_a_20_10_c_100_75_0703_2211",
            environment="CustomCartPole-v0",            
            env_directory="lib.custom_model:CustomCartPoleEnv"            
        ),
        SimpleNamespace(
            model="best_+881.955_420000.dat",  
            actor="linear",    
            directory="saves/linear/ddpg_custom_1_linear_a_20_10_c_100_75_0703_1842",
            environment="CustomCartPole-v0",            
            env_directory="lib.custom_model:CustomCartPoleEnv"            
        ),
        SimpleNamespace(
            model="best_+855.387_880000.dat",  
            actor="linear",    
            directory="saves/linear/ddpg_custom_2_linear_a_20_10_c_100_75_0703_2211",
            environment="CustomCartPole-v0",            
            env_directory="lib.custom_model:CustomCartPoleEnv"            
        ),
        SimpleNamespace(
            model="best_+987.935_1004000.dat",  
            actor="linear",    
            directory="saves/linear/ddpg_custom_3_linear_a_20_10_c_100_75_0703_2211",
            environment="CustomCartPole-v0",            
            env_directory="lib.custom_model:CustomCartPoleEnv"            
        ),
        SimpleNamespace(
            model="best_+891.159_1080000.dat",  
            actor="linear",    
            directory="saves/linear/ddpg_custom_4_linear_a_20_10_c_100_75_0704_0032",
            environment="CustomCartPole-v0",            
            env_directory="lib.custom_model:CustomCartPoleEnv"            
        )
    ]

    args_residual = [
        SimpleNamespace(
            model="best_+857.693_220000.dat",  
            actor="residual",    
            directory="saves/resnet/ddpg_custom_0_residual_a_20_10_c_100_75_0703_1817",
            environment="CustomCartPole-v0",            
            env_directory="lib.custom_model:CustomCartPoleEnv"            
        ),
        SimpleNamespace(
            model="best_+862.095_368000.dat",  
            actor="residual",    
            directory="saves/resnet/ddpg_custom_1_residual_a_20_10_c_100_75_0703_1817",
            environment="CustomCartPole-v0",            
            env_directory="lib.custom_model:CustomCartPoleEnv"            
        ),
        SimpleNamespace(
            model="best_+917.026_652000.dat",  
            actor="residual",    
            directory="saves/resnet/ddpg_custom_2_residual_a_20_10_c_100_75_0703_1817",
            environment="CustomCartPole-v0",            
            env_directory="lib.custom_model:CustomCartPoleEnv"            
        ),
        SimpleNamespace(
            model="best_+966.037_484000.dat",  
            actor="residual",    
            directory="saves/resnet/ddpg_custom_3_residual_a_20_10_c_100_75_0703_1817",
            environment="CustomCartPole-v0",            
            env_directory="lib.custom_model:CustomCartPoleEnv"            
        ),
        SimpleNamespace(
            model="best_+894.430_472000.dat",  
            actor="residual",    
            directory="saves/resnet/ddpg_custom_4_residual_a_20_10_c_100_75_0703_1817",
            environment="CustomCartPole-v0",            
            env_directory="lib.custom_model:CustomCartPoleEnv"            
        )
    ]

    num_seeds = len(args_linear)

    custom_model.register_env(id=args_linear[0].environment, entry_point=args_linear[0].env_directory)
    env = gym.make(args_linear[0].environment, max_force = 1.0, max_torque = 1.0, use_is_done=False, act_clip=False, apply_dist = False)

    linear_controller = [
        ControllerSynthesis(
            env = env,
            directory = args_linear[i].directory,
            model_name = args_linear[i].model,
            device = 'cuda',
            linear_gear = 300,
            rotational_gear = 30,
            struct = 'linear',
            verbose = True
        ) for i in range(num_seeds)
    ]

    resnet_controller = [
        ControllerSynthesis(
            env = env,
            directory = args_residual[i].directory,
            model_name = args_residual[i].model,
            device = 'cuda',
            linear_gear = 300,
            rotational_gear = 30,
            struct = 'resnet',
            verbose = True
        ) for i in range(num_seeds)
    ]

    x_linear_train_roll = [None] * num_seeds
    a_linear_train_roll = [None] * num_seeds
    total_reward_linear_train = [None] * num_seeds
    
    x_residual_train_roll = [None] * num_seeds
    a_residual_train_roll = [None] * num_seeds
    total_reward_residual_train = [None] * num_seeds
    steps = [None] * num_seeds

    for i in range(num_seeds):
        print(f"{i}-th linear controller")
        linear_controller[i].get_data()
        x_linear_train_roll[i], a_linear_train_roll[i], total_reward_linear_train[i], steps[i], _ = \
            linear_controller[i].roll_out(
                dt = DT, l2_dist = L2_DIST, linear_mag = 10, rotational_mag = 6
            )

        if VIDEO_SAVE:
            save_mujoco_video(x_linear_train_roll[i], f"video/{L2_DIST}_linear_control_"+str(i)+".mp4", fps=int(round(1 / DT)))
            print(f"  Finished saving {i}-th video")
            
    for i in range(num_seeds):
        print(f"{i}-th resnet controller")
        resnet_controller[i].get_data()
        x_residual_train_roll[i], a_residual_train_roll[i], total_reward_residual_train[i], steps[i], _ = \
            resnet_controller[i].roll_out(
                dt = DT, l2_dist = L2_DIST, linear_mag = 10, rotational_mag = 6
            )

        if VIDEO_SAVE:
            save_mujoco_video(x_residual_train_roll[i], f"video/{L2_DIST}_resnet_control_"+str(i)+".mp4", fps=int(round(1 / DT)))
            print(f"  Finished saving {i}-th video")
        
    m1, m2, c, g = cal_param(
        linear_controller[0].env.unwrapped.pole_mass,
        linear_controller[0].env.unwrapped.pole_length,
        linear_controller[0].env.unwrapped.pole_inertia,
        linear_controller[0].env.unwrapped.cart_mass
    )
    e = cal_e(m1, m2, c, 2, g)
    q1, q2 = cal_q_matrix(m1, m2, c, 2, g, e)

    K, D, q_plant, s_plant, r_plant = qsr_plant_var(2,2,e,q1,q2,g)
    (q_linear, s_linear, r_linear,
    q_activation, s_activation, r_activation, W_sdp) = qsr_resnet_var([4,20,10,2])

    X_sdp = cp.Variable((34,34))

    H = np.block([[ np.concatenate([np.zeros((2,6)), np.eye(2), np.zeros((2,60))], axis=1) ],
                  [ np.zeros((2,68)) ],
                  [ np.concatenate([np.eye(4), np.zeros((4,64))], axis=1) ],
                  [ np.concatenate([np.zeros((30,38)), np.eye(30)], axis=1) ],
                  [ np.concatenate([np.zeros((30,8)), np.eye(30), np.zeros((30,30))], axis=1) ]])

    q_network, _, _, _ = qsr_network_var(
            q_plant, s_plant, r_plant,
            q_linear, s_linear, r_linear,
            q_activation, s_activation, r_activation, H
        )

    pre_constraint = set_constraint(e, m1, m2, q_network,
        K, D, q_plant, s_plant, r_plant,
        q_linear, s_linear, r_linear,
        q_activation, s_activation, r_activation
    )
    
    x_linear_stable_roll = [None] * num_seeds
    a_linear_stable_roll = [None] * num_seeds
    total_reward_linear_stable = [None] * num_seeds
    
    x_residual_stable_roll = [None] * num_seeds
    a_residual_stable_roll = [None] * num_seeds
    total_reward_residual_stable = [None] * num_seeds
    steps = [None] * num_seeds

    for i in range(num_seeds):
        print(f"{i}-th linear stable controller")
        linear_controller[i].policy_refinement(m1, m2, e, K, D, q_plant, s_plant, r_plant, pre_constraint = pre_constraint)

        x_linear_stable_roll[i], a_linear_stable_roll[i], total_reward_linear_stable[i], steps[i], _ = \
            linear_controller[i].roll_out(
                dt = DT, l2_dist = L2_DIST, linear_mag = 10, rotational_mag = 6
            )
            
        if VIDEO_SAVE:
            save_mujoco_video(x_linear_stable_roll[i], f"video/{L2_DIST}_linear_stable_control_"+str(i)+".mp4", fps=int(round(1 / DT)))
            print(f"  Finished saving {i}-th video")

            
    for i in range(num_seeds):
        print(f"{i}-th resnet stable controller")
        resnet_controller[i].policy_refinement(m1, m2, e, K, D,
            q_plant, s_plant, r_plant,
            q_linear, s_linear, r_linear,
            q_activation, s_activation, r_activation,
            q_network, pre_constraint, 
            W_sdp, X_sdp
        )
        x_residual_stable_roll[i], a_residual_stable_roll[i], total_reward_residual_stable[i], steps[i], _ = \
            resnet_controller[i].roll_out(
                dt = DT, l2_dist = L2_DIST, linear_mag = 10, rotational_mag = 6
            )
            
        if VIDEO_SAVE:
            save_mujoco_video(x_residual_stable_roll[i], f"video/{L2_DIST}_resnet_stable_control_"+str(i)+".mp4", fps=int(round(1 / DT)))
            print(f"  Finished saving {i}-th video")
            
    (x_linear_train_roll, x_linear_train_roll_mean, x_linear_train_roll_std,
    x_linear_train_norm, x_linear_train_norm_mean, x_linear_train_norm_std) = mean_std_norm(x_linear_train_roll)
    (a_linear_train_roll, a_linear_train_roll_mean, a_linear_train_roll_std,
    a_linear_train_norm, a_linear_train_norm_mean, a_linear_train_norm_std) = mean_std_norm(a_linear_train_roll)
    draw_response(DT, x_linear_train_roll, f"linear controller response", L2_DIST)
    draw_norm(CUTOFF, DT, [x_linear_train_norm, a_linear_train_norm], f"linear controller norm", L2_DIST)

    (x_residual_train_roll, x_residual_train_roll_mean, x_residual_train_roll_std,
    x_residual_train_norm, x_residual_train_norm_mean, x_residual_train_norm_std) = mean_std_norm(x_residual_train_roll)
    (a_residual_train_roll, a_residual_train_roll_mean, a_residual_train_roll_std,
    a_residual_train_norm, a_residual_train_norm_mean, a_residual_train_norm_std) = mean_std_norm(a_residual_train_roll)
    draw_response(DT, x_residual_train_roll, f"resnet controller response", L2_DIST)
    draw_norm(CUTOFF, DT, [x_residual_train_norm, a_residual_train_norm], f"resnet controller norm", L2_DIST)
            
    (x_linear_stable_roll, x_linear_stable_roll_mean, x_linear_stable_roll_std,
    x_linear_stable_norm, x_linear_stable_norm_mean, x_linear_stable_norm_std) = mean_std_norm(x_linear_stable_roll)
    (a_linear_stable_roll, a_linear_stable_roll_mean, a_linear_stable_roll_std,
    a_linear_stable_norm, a_linear_stable_norm_mean, a_linear_stable_norm_std) = mean_std_norm(a_linear_stable_roll)
    draw_response(DT, x_linear_stable_roll, f"linear stable controller response", L2_DIST)
    draw_norm(CUTOFF, DT, [x_linear_stable_norm, a_linear_stable_norm], f"linear stable controller norm", L2_DIST)

    (x_residual_stable_roll, x_residual_stable_roll_mean, x_residual_stable_roll_std,
    x_residual_stable_norm, x_residual_stable_norm_mean, x_residual_stable_norm_std) = mean_std_norm(x_residual_stable_roll)
    (a_residual_stable_roll, a_residual_stable_roll_mean, a_residual_stable_roll_std,
    a_residual_stable_norm, a_residual_stable_norm_mean, a_residual_stable_norm_std) = mean_std_norm(a_residual_stable_roll)
    draw_response(DT, x_residual_stable_roll, f"resnet stable controller response", L2_DIST)
    draw_norm(CUTOFF, DT, [x_residual_stable_norm, a_residual_stable_norm], f"resnet stable controller norm", L2_DIST)

    for i in range(num_seeds):
        np.savetxt(f"results/data/x_linear_train_seed_{i}.txt", x_linear_train_roll[i,:,:], delimiter=",")
        np.savetxt(f"results/data/x_residual_train_seed_{i}.txt", x_residual_train_roll[i,:,:], delimiter=",")
        np.savetxt(f"results/data/x_linear_stable_seed_{i}.txt", x_linear_stable_roll[i,:,:], delimiter=",")
        np.savetxt(f"results/data/x_residual_stable_seed_{i}.txt", x_residual_stable_roll[i,:,:], delimiter=",")
        
        np.savetxt(f"results/data/a_linear_train_seed_{i}.txt", a_linear_train_roll[i,:,:], delimiter=",")
        np.savetxt(f"results/data/a_residual_train_seed_{i}.txt", a_residual_train_roll[i,:,:], delimiter=",")
        np.savetxt(f"results/data/a_linear_stable_seed_{i}.txt", a_linear_stable_roll[i,:,:], delimiter=",")
        np.savetxt(f"results/data/a_residual_stable_seed_{i}.txt", a_residual_stable_roll[i,:,:], delimiter=",")

    group_trajectory = {
        "Trained linear controller": x_linear_train_roll,
        "Stable linear controller": x_linear_stable_roll,
        "Trained resnet controller": x_residual_train_roll,
        "Stable resnet controller": x_residual_stable_roll
    }
    group_trajectory_mean_std = {
        "Trained linear controller": [x_linear_train_roll_mean, x_linear_train_roll_std],
        "Stable linear controller": [x_linear_stable_roll_mean, x_linear_stable_roll_std],
        "Trained resnet controller": [x_residual_train_roll_mean, x_residual_train_roll_std],
        "Stable resnet controller": [x_residual_stable_roll_mean, x_residual_stable_roll_std]
    }
    group_state_norm = {
        "Trained linear controller": x_linear_train_norm,
        "Stable linear controller": x_linear_stable_norm,
        "Trained resnet controller": x_residual_train_norm,
        "Stable resnet controller": x_residual_stable_norm
    }
    group_action_norm = {
        "Trained linear controller": a_linear_train_norm,
        "Stable linear controller": a_linear_stable_norm,
        "Trained resnet controller": a_residual_train_norm,
        "Stable resnet controller": a_residual_stable_norm
    }
    group_state_norm_mean_std = {
        "Trained linear controller": [x_linear_train_norm_mean, x_linear_train_norm_std],
        "Stable linear controller": [x_linear_stable_norm_mean, x_linear_stable_norm_std],
        "Trained resnet controller": [x_residual_train_norm_mean, x_residual_train_norm_std],
        "Stable resnet controller": [x_residual_stable_norm_mean, x_residual_stable_norm_std]
    }
    group_action_norm_mean_std = {
        "Trained linear controller": [a_linear_train_norm_mean, a_linear_train_norm_std],
        "Stable linear controller": [a_linear_stable_norm_mean, a_linear_stable_norm_std],
        "Trained resnet controller": [a_residual_train_norm_mean, a_residual_train_norm_std],
        "Stable resnet controller": [a_residual_stable_norm_mean, a_residual_stable_norm_std]
    }

    draw_response_comparison(DT, group_trajectory, f"response comparison", L2_DIST)
    draw_response_mean_comparison(DT, group_trajectory_mean_std, f"response comparison mean", L2_DIST)
    draw_norm_comparison(CUTOFF, DT, [group_state_norm, group_action_norm], f"norm comparison", L2_DIST)
    draw_norm_mean_comparison(CUTOFF, DT, [group_state_norm_mean_std, group_action_norm_mean_std], f"norm comparison mean", L2_DIST)

    print("Total reward (linear trained):")
    print("  ", np.array(total_reward_linear_train))
    print("Total reward (linear stable):")
    print("  ", np.array(total_reward_linear_stable))
    print("Total reward (resnet trained):")
    print("  ", np.array(total_reward_residual_train))
    print("Total reward (resnet stable):")
    print("  ", np.array(total_reward_residual_stable))

    