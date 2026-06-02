import gymnasium as gym
import numpy as np
import torch
import ptan

def unpack_batch_ddpg(batch, device="cpu"):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    states_v = torch.FloatTensor(states).to(device)
    actions_v = torch.FloatTensor(actions).to(device)
    rewards_v = torch.FloatTensor(rewards).to(device)
    last_states_v = torch.FloatTensor(last_states).to(device)
    dones_t = torch.BoolTensor(dones).to(device)
    return states_v, actions_v, rewards_v, dones_t, last_states_v