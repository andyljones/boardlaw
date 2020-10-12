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

        self.q = torch.full((env.n_envs, n_sims, n_actions), np.nan, device=self.device)
        self.log_pi = torch.full((env.n_envs, n_sims, n_actions), np.nan, device=self.device)
        self.n = torch.full((env.n_envs, n_sims, n_actions), 0, device=self.device)
        self.v = torch.full((env.n_envs, n_sims), np.nan, device=self.device)
        self.r = torch.full((env.n_envs, n_sims, n_actions), np.nan, device=self.device)

        self.sim = 0
    
    def descend(self, env, inputs, agent):
        original_state = env.state_dict()

        current = torch.zeros_like(self.parents)
        active = torch.ones_like(self.parents, dtype=torch.bool)
        while True:
            is_leaf = (self.parents[current] == -1)
            to_eval = is_leaf & active

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