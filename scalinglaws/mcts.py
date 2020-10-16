"""
Right, MCTS:
    * 
"""
from collections import namedtuple
from numpy.core.overrides import array_function_dispatch
import torch
import numpy as np
from rebar import arrdict

class MCTS:

    def __init__(self, env, n_sims):
        self.device = env.device
        self.n_envs = env.n_envs
        self.n_nodes = n_sims+1

        self.envs = torch.arange(env.n_envs, device=self.device).cuda()

        n_actions = np.prod(env.action_space)
        self.children = self.envs.new_full((env.n_envs, self.n_nodes, n_actions), -1)
        self.parents = self.envs.new_full((env.n_envs, self.n_nodes), -1)
        self.relation = self.envs.new_full((env.n_envs, self.n_nodes), -1)

        self.log_pi = torch.full((env.n_envs, self.n_nodes, n_actions), np.nan, device=self.device)
        self.v = torch.full((env.n_envs, self.n_nodes), np.nan, device=self.device)
        self.n = torch.full((env.n_envs, self.n_nodes, n_actions), 0, device=self.device, dtype=torch.int)
        self.w = torch.full((env.n_envs, self.n_nodes, n_actions), np.nan, device=self.device)
        self.terminal = torch.full((env.n_envs, self.n_nodes), False, device=self.device, dtype=torch.bool)

        self.sim = 0

        # https://github.com/LeelaChessZero/lc0/issues/694
        self.c_puct = 2.5

    def sample(self, envs, nodes):
        pi = self.log_pi[envs, nodes].exp()
        q = self.w[envs, nodes]/self.n[envs, nodes]
        n = self.n[envs, nodes]

        N = n.sum(-1, keepdims=True)

        values = q + self.c_puct*pi*N/(1 + n)
        return values.max(-1).indices

    def descend(self):
        trace = []
        current = torch.full_like(self.envs, -1)
        next = torch.full_like(self.envs, 0)
        while True:
            interior = (next != -1)
            if not interior.any():
                break

            current[interior] = next[interior]

            actions = torch.full_like(self.envs, -1)
            actions[interior] = self.sample(self.envs[interior], current[interior])
            trace.append(actions)

            next = torch.where(interior, self.children[interior, current, actions], current)

        return torch.stack(trace), current

    def play(self, env, inputs, trace):
        inputs = inputs.clone()
        #TODO: Handle termination
        for a in trace:
            active = a != -1
            dummies = torch.distributions.Categorical(probs=inputs.mask.float()).sample()
            dummies[active] = a
            r, i = env.step(dummies)

            #TODO: Generalise this
            for k in inputs:
                inputs[k][active] = i[k][active]

        return inputs

    def backup(self, current, v):
        v = v.clone()
        current = torch.full_like(self.envs, current)
        while True:
            active = (self.parents[self.envs, current] != -1)
            if not active.any():
                break

            parent = self.parents[self.envs[active], current[active]]
            relation = self.relation[self.envs[active], current[active]]

            v[self.terminal[self.envs[active], current[active]]] = 0. 
            self.n[self.envs[active], parent, relation] += 1
            self.w[self.envs[active], parent, relation] += v

            current[active] = parent

    def initialize(self, env, inputs, agent):
        original_state = env.state_dict()

        decisions = agent(inputs, value=True)
        self.log_pi[:, self.sim] = decisions.logits
        self.v[:, self.sim] = decisions.v
        self.w[:, self.sim] = decisions.v
        self.n[:, self.sim] = 1 

        self.sim += 1

        env.load_state_dict(original_state)

    def simulate(self, env, inputs, agent):
        if self.sim >= self.n_nodes:
            raise ValueError('Called simulate more times than were declared in the constructor')

        original_state = env.state_dict()

        trace, leaf = self.descend()
        self.children[self.envs, leaf, trace[-1]] = self.sim
        self.parents[self.envs, self.sim] = leaf
        self.relation[self.envs, self.sim] = trace[-1]

        inputs = self.play(env, inputs, trace)
        self.terminal[:, self.sim] = inputs.terminal

        decisions = agent(inputs, value=True)
        self.log_pi[:, self.sim] = decisions.logits
        self.v[:, self.sim] = decisions.v
        self.w[:, self.sim] = 0.
        self.n[:, self.sim] = 0 

        self.backup(self.sim, decisions.v)

        self.sim += 1

        env.load_state_dict(original_state)

    def root(self):
        return arrdict.arrdict(
            logits=self.log_pi[:, 0],
            v=self.v[:, 0])

def mcts(env, inputs, agent, n_sims=100):
    mcts = MCTS(env, n_sims)

    mcts.initialize(env, inputs, agent)
    for _ in range(n_sims):
        mcts.simulate(env, inputs, agent)

    return mcts.root()

class TestEnv:

    def __init__(self, n_envs=1, length=4):
        self.device = 'cpu'
        self.n_envs = n_envs
        self.length = length

        self.action_space = namedtuple('Vector', ('dim',))((2,))
        self.history = torch.full((n_envs, length), -1, dtype=torch.long)
        self.idx = torch.full((n_envs,), 0, dtype=torch.long)

    def _observe(self):
        return arrdict.arrdict(
            obs=(self.history[:, :-1] == 0).all(-1),
            mask=torch.ones((self.n_envs, 2), dtype=torch.bool, device=self.device),
            terminal=(self.idx == self.length))

    def reset(self):
        return self._observe()

    def step(self, actions):
        self.history[:, self.idx] = actions

        self.idx += 1 
        response = arrdict.arrdict(
            reward=torch.zeros((self.n_envs,), device=self.device))

        inputs = self._observe().clone()

        self.idx[inputs.terminal] = 0
        self.history[inputs.terminal] = -1
        
        return response, inputs

    def state_dict(self):
        return arrdict.arrdict(
            history=self.history,
            idx=self.idx)

    def load_state_dict(self, d):
        self.history = d.history
        self.idx = d.idx

class TestAgent:

    def __call__(self, inputs, sample=True, value=True):
        B, A = inputs.mask.shape
        return arrdict.arrdict(
            logits=torch.full((B, A), -np.log(A)),
            v=inputs.obs.float())
    
def test():
    env = TestEnv()
    agent = TestAgent()
    inputs = env.reset()

    mcts = MCTS(2, env, agent)

    mcts.initialize(env, inputs, agent)
    mcts.simulate(env, inputs, agent)
    mcts.simulate(env, inputs, agent)