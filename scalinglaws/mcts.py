from collections import namedtuple

from networkx.algorithms.graphical import is_valid_degree_sequence_havel_hakimi
from scalinglaws.testgames import RandomAgent
from numpy.core.overrides import array_function_dispatch
import torch
import numpy as np
from rebar import arrdict

class MCTS:

    def __init__(self, env, n_nodes, c_puct=2.5):
        self.device = env.device
        self.n_envs = env.n_envs
        self.n_nodes = n_nodes
        self.n_seats = env.n_seats

        self.envs = torch.arange(env.n_envs, device=self.device)

        n_actions = np.prod(env.action_space)
        self.children = self.envs.new_full((env.n_envs, self.n_nodes, n_actions), -1)
        self.parents = self.envs.new_full((env.n_envs, self.n_nodes), -1)
        self.relation = self.envs.new_full((env.n_envs, self.n_nodes), -1)

        #TODO: All of these but the children tensor should not have a 'actions' dim; the field can be attributed
        # to the parent 
        self.log_pi = torch.full((env.n_envs, self.n_nodes, n_actions), np.nan, device=self.device)
        self.valid = torch.full((env.n_envs, self.n_nodes, n_actions), False, device=self.device, dtype=torch.bool)
        self.n = torch.full((env.n_envs, self.n_nodes, n_actions), 0, device=self.device, dtype=torch.int)
        self.m = torch.full((env.n_envs, self.n_nodes, n_actions), 0, device=self.device, dtype=torch.int)
        self.w = torch.full((env.n_envs, self.n_nodes, n_actions), np.nan, device=self.device)
        self.terminal = torch.full((env.n_envs, self.n_nodes), False, device=self.device, dtype=torch.bool)
        self.rewards = torch.full((env.n_envs, self.n_nodes, self.n_seats), 0., device=self.device, dtype=torch.float)
        self.seats = torch.full((env.n_envs, self.n_nodes), -1, device=self.device, dtype=torch.long)

        self.sim = 0

        # https://github.com/LeelaChessZero/lc0/issues/694
        self.c_puct = c_puct

    def sample(self, envs, nodes):
        pi = self.log_pi[envs, nodes].exp()
        q = self.w[envs, nodes]/self.n[envs, nodes]
        n = self.n[envs, nodes]

        N = n.sum(-1, keepdims=True)

        values = q + self.c_puct*pi*N/(1 + n)
        values[~self.valid[envs, nodes]] = -np.inf
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
        responses = arrdict.arrdict(
            terminal=torch.zeros_like(self.terminal[:, self.sim]),
            rewards=torch.zeros_like(self.rewards[:, self.sim]))
        #TODO: Handle termination, bit of wasted compute right now.
        for a in trace:
            active = a != -1
            dummies = torch.distributions.Categorical(probs=inputs.valid.float()).sample()
            dummies[active] = a
            new_responses, new_inputs = env.step(dummies)

            #TODO: Generalise this
            for k in inputs:
                inputs[k][active] = new_inputs[k][active]
            for k in responses:
                responses[k][active] = new_responses[k][active]

        return responses, inputs

    def backup(self, current, seat, v):
        root_seat = self.seats[:, 0]
        v = v.clone()
        m = torch.ones_like(v, dtype=torch.int)
        current = torch.full_like(self.envs, current)
        #TODO: Need to backup value only for choosing player
        while True:
            active = (self.parents[self.envs, current] != -1) & (root_seat == seat)
            if not active.any():
                break

            parent = self.parents[self.envs[active], current[active]]
            relation = self.relation[self.envs[active], current[active]]

            v[self.terminal[self.envs[active], current[active]]] = 0. 

            v[active] = v[active] + self.rewards[self.envs[active], current[active], root_seat[active]]

            self.m[self.envs[active], parent, relation] += m
            self.n[self.envs[active], parent, relation] += 1
            self.w[self.envs[active], parent, relation] += v

            m[self.terminal[self.envs[active], current[active]]] = 0

            current[active] = parent

    def initialize(self, env, inputs, agent):
        original_state = env.state_dict()

        self.valid[:, self.sim] = inputs.valid
        self.seats[:, self.sim] = inputs.seats
        self.terminal[:, self.sim] = False
        self.rewards[:, self.sim] = 0

        decisions = agent(inputs, value=True)
        self.log_pi[:, self.sim] = decisions.logits
        self.w[:, self.sim] = 0.
        self.n[:, self.sim] = 0 
        self.m[:, self.sim] = 0 

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

        response, inputs = self.play(env, inputs, trace)
        self.valid[:, self.sim] = inputs.valid
        self.seats[:, self.sim] = inputs.seats
        self.terminal[:, self.sim] = response.terminal
        self.rewards[:, self.sim] = response.rewards

        decisions = agent(inputs, value=True)
        self.log_pi[:, self.sim] = decisions.logits
        self.w[:, self.sim] = 0
        self.m[:, self.sim] = 0
        self.n[:, self.sim] = 0 

        self.backup(self.sim, inputs.seats, decisions.v)

        self.sim += 1

        env.load_state_dict(original_state)

    def root(self):
        return arrdict.arrdict(
            p=self.n[:, 0].float()/self.n[:, 0].sum(-1, keepdims=True),
            logits=self.log_pi[:, 0],
            v=self.w[:, 0]/self.m[:, 0])

    def display(self, e=0):
        import networkx as nx
        import matplotlib.pyplot as plt

        edges, labels, edge_vals = [], {}, {}
        qs = (self.w/self.m).where(self.m > 0, torch.zeros_like(self.w))[e].cpu()
        q_min, q_max = np.nanmin(qs), np.nanmax(qs)
        for i, p in enumerate(self.parents[e].cpu()):
            p = int(p)
            if p >= 0:
                r = int(self.relation[e][i].cpu())
                q = float(qs[p, r])
                edge = (p, i)
                edges.append(edge)
                labels[edge] = f'{r}, {q:.1f}'
                edge_vals[edge] = (q - q_min)/(q_max - q_min + 1e-6)
            
        G = nx.from_edgelist(edges)
        colors = ['C1' if t else 'C2' for t in self.terminal.float()[e].cpu().numpy()]
        edge_colors = [edge_vals[e] for e in G.edges()]

        pos = nx.kamada_kawai_layout(G)
        nx.draw(G, pos, node_color=colors, edge_color=edge_colors, width=5)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        nx.draw_networkx_labels(G, pos, labels={i: i for i in range(self.n_nodes)})

        return plt.gca()

def mcts(env, inputs, agent, **kwargs):
    mcts = MCTS(env, **kwargs)

    mcts.initialize(env, inputs, agent)
    for _ in range(mcts.n_nodes-1):
        mcts.simulate(env, inputs, agent)

    return mcts

def test():
    from . import testgames
    env = testgames.InstantReturn()
    agent = RandomAgent(env)
    inputs = env.initialize()

    m = mcts(env, inputs, agent, 7)