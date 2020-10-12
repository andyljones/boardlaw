"""
Right, MCTS:
    * 
"""
from re import L
import torch
import numpy as np

class MCTS:

    def __init__(self, n_sims, env, agent):
        self.device = env.device
        self.n_envs = env.n_envs
        self.n_sims = n_sims

        self.envs = torch.arange(env.n_envs, device=self.device).cuda()

        n_actions = np.prod(env.action_space.size())
        self.children = self.envs.new_full((env.n_envs, n_sims, n_actions))
        self.parents = self.envs.new_full((env.n_envs, n_sims), -1)
        self.relation = self.envs.new_full((env.n_envs, n_sims), -1)

        self.log_pi = torch.full((env.n_envs, n_sims, n_actions), np.nan, device=self.device)
        self.v = torch.full((env.n_envs, n_sims), np.nan, device=self.device)
        self.n = torch.full((env.n_envs, n_sims, n_actions), 0, device=self.device, dtype=torch.int)
        self.w = torch.full((env.n_envs, n_sims, n_actions), np.nan, device=self.device)
        self.r = torch.full((env.n_envs, n_sims, n_actions), np.nan, device=self.device)
        self.terminal = torch.full((env.n_envs, n_sims, n_actions), False, device=self.device, dtype=torch.long)

        self.sim = 0

        # https://github.com/LeelaChessZero/lc0/issues/694
        self.c_puct = 2.5

    def sample(self, envs, nodes):
        pi = self.log_pi[envs, nodes].exp()
        q = self.w[envs, nodes]/self.n[envs, nodes]
        n = self.n[envs, nodes]

        N = n.sum(-1, keepdims=True)

        values = q + self.c_puct*pi*N/(1 + n)
        return values.max(-1)
    
    def descend(self):
        trace = []
        current = torch.zeros_like(self.envs)
        next = torch.zeros_like(self.envs)
        while True:
            active = torch.isnan(self.v[self.envs, next])
            if not active.any():
                break
            current = next

            actions = torch.full_like(self.parents, -1)
            actions[active] = self.sample(self.envs[active], current[active])
            trace.append(actions)

            next = torch.where(active, current, self.children[active, current, actions])

            self.parents

        return torch.stack(trace), next

    def replay(self, env, inputs, trace):
        for a in trace:
            dummies = inputs.mask.nonzero(-1)
            dummies[a != -1] = a[a != -1]
            responses, inputs = env.step(dummies)

            self.r[self.envs, self.sim, dummies] = responses.reward
            self.terminal[self.envs, self.sim, dummies] = responses.terminal
        
        return inputs

    def backup(self, trace, current):
        current = torch.full_like(self.envs, current)
        while True:
            active = (self.parents[self.envs, current] != -1)
            self.n[self.envs[active], current[active], ]

    
    def simulate(self, env, inputs, agent):
        original_state = env.state_dict()

        trace, current = self.descend()
        inputs = self.replay(env, inputs, trace)

        decisions = agent(inputs[None], value=True).squeeze(-1)
        self.log_pi[:, self.sim] = decisions.logits
        self.w[:, self.sim] = 0.
        self.n[:, self.sim] = 0 

        self.parents[self.envs, self.sim] = current
        self.relation[self.envs, self.sim] = 

        self.sim += 1

        env.load_state_dict(original_state)

        pass