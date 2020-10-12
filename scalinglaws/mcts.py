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
        pass

        n_actions = np.prod(env.action_space.size())
        self.children = torch.full((env.n_envs, n_sims, n_actions), -1, device=self.device)
        self.parents = torch.full((env.n_envs, n_sims), -1, device=self.device)

        self.log_pi = torch.full((env.n_envs, n_sims, n_actions), np.nan, device=self.device)
        self.v = torch.full((env.n_envs, n_sims), np.nan, device=self.device)
        self.n = torch.full((env.n_envs, n_sims, n_actions), 0, device=self.device)
        self.q = torch.full((env.n_envs, n_sims, n_actions), np.nan, device=self.device)
        self.r = torch.full((env.n_envs, n_sims, n_actions), np.nan, device=self.device)

        self.envs = torch.arange(env.n_envs, device=self.device).cuda()

        self.sim = 0

        # https://github.com/LeelaChessZero/lc0/issues/694
        self.c_puct = 2.5

    def sample(self, envs, nodes):
        q = self.q[envs, nodes]
        pi = self.log_pi[envs, nodes].exp()
        n = self.n[envs, nodes]

        N = n.sum(-1, keepdims=True)

        values = q + self.c_puct*pi*N/(1 + n)
        return values.max(-1)
    
    def descend(self):
        trace = []
        current = torch.zeros_like(self.parents)
        while True:
            active = torch.isnan(self.v[self.envs, current])
            if not active.any():
                break

            actions = torch.full_like(self.parents, -1)
            actions[active] = self.sample(self.envs[active], current[active])
            trace.append(actions)

            current[active] = self.children[active, current, actions]

        return trace
    
    def tmp(self, env, inputs, agent):
        original_state = env.state_dict()

        current = torch.zeros_like(self.parents)
        active = torch.ones_like(self.parents, dtype=torch.bool)
        while True:
            is_leaf = (self.parents[current] == -1)
            to_eval = is_leaf & active

            # Evaluate the envs that have reached a leaf
            decisions = agent(inputs[to_eval][None], value=True).squeeze(0)
            self.log_pi[to_eval, self.sim] = decisions.logits
            self.v[to_eval, self.sim] = decisions.values 

            active[is_leaf] = False

            if is_leaf.all():
                break

            actions = self.sample(current[active])
            self.children[to_eval, current, actions] = self.sim
            self.parents[to_eval, self.sim] = current

            current = self.children[active.nonzero(), current, actions]

            dummies = decisions.mask.reshape(self.n_envs, -1).nonzero(-1)
            dummies[active] = actions
            responses, inputs = env.step(dummies)

            self.r[active, self.sim, actions] = responses.reward
            active[active] = ~inputs.terminal

        self.sim += 1

        env.load_state_dict(original_state)

        pass