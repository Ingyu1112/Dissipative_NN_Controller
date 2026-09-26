import ptan
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

HID_SIZE = 128

class DDPGActor(nn.Module): 
    def __init__(self, obs_size: int, act_size: int, hidden: list, bias: bool = False): 
        super(DDPGActor, self).__init__() 
 
        if not bias:
            self.net = nn.Sequential( 
                nn.Linear(obs_size, hidden[0], bias=False), 
                nn.ReLU(), 
                nn.Linear(hidden[0], hidden[1], bias=False), 
                nn.ReLU(), 
                nn.Linear(hidden[1], act_size, bias=False) # to squeeze the values to -1 ... 1 range,
            )
        else:
            self.net = nn.Sequential( 
                nn.Linear(obs_size, hidden[0]), 
                nn.ReLU(), 
                nn.Linear(hidden[0], hidden[1]), 
                nn.ReLU(), 
                nn.Linear(hidden[1], act_size), 
                nn.Tanh()   # to squeeze the values to -1 ... 1 range
            )

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        action = torch.tanh(out/3)*3
        return action

class DDPGActorResidual(nn.Module): 
    def __init__(self, obs_size: int, act_size: int, hidden: list, bias: bool = False): 
        super(DDPGActorResidual, self).__init__() 
 
        if not bias:
            self.net = nn.Sequential( 
                nn.Linear(obs_size, hidden[0], bias=False), 
                nn.ReLU(), 
                nn.Linear(hidden[0], hidden[1], bias=False), 
                nn.ReLU(), 
                nn.Linear(hidden[1], act_size, bias=False) # to squeeze the values to -1 ... 1 range,
            )
        else:
            self.net = nn.Sequential( 
                nn.Linear(obs_size, hidden[0]), 
                nn.ReLU(), 
                nn.Linear(hidden[0], hidden[1]), 
                nn.ReLU(), 
                nn.Linear(hidden[1], act_size), 
                nn.Tanh()   # to squeeze the values to -1 ... 1 range
            )

        self.residual = nn.Linear(obs_size, act_size, bias=False)

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        out_residual = self.residual(x)
        return out + out_residual

        
class DDPGActorLinear(nn.Module): 
    def __init__(self, obs_size: int, act_size: int, bias: bool = False): 
        super(DDPGActorLinear, self).__init__() 
 
        if not bias:
            self.net = nn.Sequential( 
                nn.Linear(obs_size, act_size, bias=False)
            )
        else:
            self.net = nn.Sequential( 
                nn.Linear(obs_size, act_size)
            )

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        return out


class DDPGCritic(nn.Module):    # real implementation of Q-value Q(s,a)
    def __init__(self, obs_size: int, act_size: int, hidden: list): 
        super(DDPGCritic, self).__init__() 
 
        self.obs_net = nn.Sequential( 
            nn.Linear(obs_size, hidden[0]), 
            nn.ReLU(), 
        )

        self.out_net = nn.Sequential(
            nn.Linear(hidden[0] + act_size, hidden[1]), # include action input
            nn.ReLU(),
            nn.Linear(hidden[1], 1)
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor): 
        obs = self.obs_net(x) 
        return self.out_net(torch.cat([obs, a], dim=1))
        
class AgentDDPG(ptan.agent.BaseAgent):
    """
    Agent implementing Orstein-Uhlenbeck exploration process
    """
    def __init__(self, net: DDPGActor, device: torch.device = torch.device('cpu'),
                 ou_enabled: bool = True, ou_mu: float = 0.0, ou_teta: float = 0.15,
                 ou_sigma: float = 0.2, ou_epsilon: float = 1.0,
                 act_clip: bool = True, max_input: float = 1.0):
        self.net = net
        self.device = device
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_teta = ou_teta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon
        self.act_clip = act_clip
        self.max_input = max_input

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
            
        actions = np.clip(actions, -self.max_input, self.max_input) if self.act_clip else actions
        return actions, new_a_states

class MultiAgentWrapper(ptan.agent.BaseAgent):
    def __init__(self, act_nets: list, device: torch.device = torch.device('cpu')):
        self.agents = [AgentDDPG(net, device) for net in act_nets]
        self.num_agents = len(act_nets)

    def __call__(self, states, agent_states=None):
        states_np = np.array(states)
        agent_actions = []

        for i, agent in enumerate(self.agents):
            local_states = states_np[:, i*self.num_agents : (i+1)*self.num_agents]

            actions, _ = agent(local_states, agent_states)
            agent_actions.append(actions)

        joint_actions = np.concatenate(agent_actions, axis=1)

        return joint_actions, agent_states