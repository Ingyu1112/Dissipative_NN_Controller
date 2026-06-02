import ptan
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

HID_SIZE = 128

class DDPGActor(nn.Module): 
    def __init__(self, obs_size: int, act_size: int, bias: bool = False): 
        super(DDPGActor, self).__init__() 
 
        if not bias:
            self.net = nn.Sequential( 
                nn.Linear(obs_size, 400, bias=False), 
                nn.ReLU(), 
                nn.Linear(400, 300, bias=False), 
                nn.ReLU(), 
                nn.Linear(300, act_size, bias=False), 
                nn.Tanh()   # to squeeze the values to -1 ... 1 range
            )
        else:
            self.net = nn.Sequential( 
                nn.Linear(obs_size, 400), 
                nn.ReLU(), 
                nn.Linear(400, 300), 
                nn.ReLU(), 
                nn.Linear(300, act_size), 
                nn.Tanh()   # to squeeze the values to -1 ... 1 range
            )

    def forward(self, x: torch.Tensor):
        return self.net(x)

class DDPGCritic(nn.Module):    # real implementation of Q-value Q(s,a)
    def __init__(self, obs_size: int, act_size: int): 
        super(DDPGCritic, self).__init__() 
 
        self.obs_net = nn.Sequential( 
            nn.Linear(obs_size, 400), 
            nn.ReLU(), 
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300), # include action input
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor): 
        obs = self.obs_net(x) 
        return self.out_net(torch.cat([obs, a], dim=1))
        
class AgentDDPG(ptan.agent.BaseAgent):
    """
    Agent implementing Orstein0Uhlenbeck exploration process
    """
    def __init__(self, net: DDPGActor, device: torch.device = torch.device('cpu'),
                 ou_enabled: bool = True, ou_mu: float = 0.0, ou_teta: float = 0.15,
                 ou_sigma: float = 0.2, ou_epsilon: float = 1.0):
        self.net = net
        self.device = device
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_teta = ou_teta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon

    def initial_state(self):
        return None

    def __call__(self, states: ptan.agent.States, agent_states: ptan.agent.AgentStates):
        # convert the observed state and internal agent state into the action
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)
        mu_v = self.net(states_v) # compute deterministic actions
        actions = mu_v.data.cpu().numpy()

        # adding the exploration noise by applying the OU process
        # in the loop, we iterate over
        # the batch of observations and the list of the agent states from the previous call
        # and we update the OU process value
        if self.ou_enabled and self.ou_epsilon > 0:
            new_a_states = []
            for a_state, action in zip(agent_states, actions):
                if a_state is None:
                    a_state = np.zeros(shape=action.shape, dtype=np.float32)
                a_state += self.ou_teta * (self.ou_mu - a_state)
                a_state += self.ou_sigma * np.random.normal(size=action.shape)

                action += self.ou_epsilon * a_state # add the noise from the OU process to our actions
                new_a_states.append(a_state)    # save the noise value for the next step
        else:
            new_a_states = agent_states
            
        actions = np.clip(actions, -1, 1)
        return actions, new_a_states

