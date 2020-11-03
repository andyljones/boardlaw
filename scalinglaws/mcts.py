import torch
import numpy as np
from rebar import arrdict

class MCTS:

    def __init__(self, env, n_nodes, c_puct=2.5):
        self.device = env.device
        self.n_envs = env.n_envs
        self.n_nodes = n_nodes
        self.n_seats = env.n_seats
        assert n_nodes > 1, 'MCTS requires at least two nodes'

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
        self.w = torch.full((env.n_envs, self.n_nodes, n_actions, self.n_seats), np.nan, device=self.device)
        self.terminal = torch.full((env.n_envs, self.n_nodes), False, device=self.device, dtype=torch.bool)
        self.rewards = torch.full((env.n_envs, self.n_nodes, self.n_seats), 0., device=self.device, dtype=torch.float)
        self.seats = torch.full((env.n_envs, self.n_nodes), -1, device=self.device, dtype=torch.long)

        self.sim = 0

        # https://github.com/LeelaChessZero/lc0/issues/694
        self.c_puct = c_puct

    def sample(self, envs, nodes):
        seat = self.seats[envs, nodes]

        pi = self.log_pi[envs, nodes].exp()
        q = self.w[envs, nodes, :, seat]/self.n[envs, nodes]
        n = self.n[envs, nodes]

        N = n.sum(-1, keepdims=True)

        values = q + self.c_puct*pi*N/(1 + n)
        values[~self.valid[envs, nodes]] = -np.inf
        return values.max(-1).indices

    def descend(self):
        trace = []
        current = torch.full_like(self.envs, -1)
        next = torch.full_like(self.envs, 0)
        final_actions = torch.full_like(self.envs, -1)
        while True:
            interior = (next != -1)
            if not interior.any():
                break

            current[interior] = next[interior]

            actions = torch.full_like(self.envs, -1)
            actions[interior] = self.sample(self.envs[interior], current[interior])
            trace.append(actions)

            final_actions[interior] = actions[interior]

            next = next.clone()
            next[interior] = self.children[self.envs[interior], current[interior], actions[interior]]

        return torch.stack(trace), current, final_actions

    def play(self, env, inputs, trace):
        inputs = inputs.clone()
        responses = arrdict.arrdict(
            terminal=torch.zeros_like(self.terminal[:, self.sim]),
            rewards=torch.zeros_like(self.rewards[:, self.sim]))
        #TODO: Handle termination, bit of wasted compute right now.
        for a in trace:
            active = a != -1
            dummies = torch.distributions.Categorical(probs=inputs.valid.float()).sample()
            dummies[active] = a[active]
            new_responses, new_inputs = env.step(dummies)

            #TODO: Generalise this
            for k in inputs:
                inputs[k][active] = new_inputs[k][active]
            for k in responses:
                responses[k][active] = new_responses[k][active]

        return responses, inputs

    def backup(self, current, v):
        v = v.clone()
        current = torch.full_like(self.envs, current)
        while True:
            active = (self.parents[self.envs, current] != -1)
            if not active.any():
                break

            parent = self.parents[self.envs[active], current[active]]
            relation = self.relation[self.envs[active], current[active]]

            t = self.terminal[self.envs[active], current[active]]
            v[self.envs[active][t]] = 0. 

            v[active] += self.rewards[self.envs[active], current[active]]

            self.n[self.envs[active], parent, relation] += 1
            self.w[self.envs[active], parent, relation] += v[active]

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

        self.sim += 1

        env.load_state_dict(original_state)

    def simulate(self, env, inputs, agent):
        if self.sim >= self.n_nodes:
            raise ValueError('Called simulate more times than were declared in the constructor')

        original_state = env.state_dict()

        trace, leaf, final_actions = self.descend()
        self.children[self.envs, leaf, final_actions] = self.sim
        self.parents[self.envs, self.sim] = leaf
        self.relation[self.envs, self.sim] = final_actions

        response, inputs = self.play(env, inputs, trace)
        self.valid[:, self.sim] = inputs.valid
        self.seats[:, self.sim] = inputs.seats
        self.terminal[:, self.sim] = response.terminal
        self.rewards[:, self.sim] = response.rewards

        decisions = agent(inputs, value=True)
        self.log_pi[:, self.sim] = decisions.logits
        self.w[:, self.sim] = 0
        self.n[:, self.sim] = 0 

        self.backup(self.sim, decisions.v)

        self.sim += 1

        env.load_state_dict(original_state)

    def root(self):
        seat = self.seats[:, 0]
        q_seat = self.w[self.envs, 0, :, seat]/self.n[:, 0]
        q = torch.zeros_like(self.w[:, 0])
        q[self.envs, :, seat] = q_seat
        q[self.envs, :, 1-seat] = -q_seat

        p = self.n[:, 0].float()/self.n[:, 0].sum(-1, keepdims=True)

        #TODO: Is this how I should be evaluating root value?
        v = (q*p[..., None]).sum(1)

        return arrdict.arrdict(
            logits=torch.log(p),
            v=v)

    def display(self, e=0):
        import networkx as nx
        import matplotlib.pyplot as plt

        root_seat = self.seats[e, 0]

        edges, labels, edge_vals = [], {}, {}
        qs = (self.w[..., root_seat]/self.n[..., None]).where(self.n > 0, torch.zeros_like(self.w))[e, :, root_seat].cpu()
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

class MCTSAgent:

    def __init__(self, env, agent, **kwargs):
        self.env = env
        self.agent = agent
        self.kwargs = kwargs

    def __call__(self, inputs, value=True):
        r = mcts(self.env, inputs, self.agent, **self.kwargs).root()
        return arrdict.arrdict(
            logits=r.logits,
            v=r.v,
            actions=torch.distributions.Categorical(logits=r.logits).sample())

from . import testgames

def test_two_player():
    env = testgames.FirstWinsSecondLoses()
    agent = testgames.ProxyAgent(env)
    inputs = env.reset()

    m = mcts(env, inputs, agent, n_nodes=3)

    expected = torch.tensor([[+1.]], device=env.device)
    torch.testing.assert_allclose(m.root().v, expected)

def test_depth():
    env = testgames.AllOnes(length=4)
    agent = testgames.ProxyAgent(env)
    inputs = env.reset()

    m = mcts(env, inputs, agent, n_nodes=15)

    expected = torch.tensor([[0., 1/8.]], device=env.device)
    torch.testing.assert_allclose(m.root().v, expected)

def test_full_game():
    from . import hex
    env = hex.Hex(boardsize=3)
    agent = testgames.RandomRolloutAgent(env, 4)
    inputs = env.reset()

    m = mcts(env, inputs, agent, n_nodes=15)

    m.root().v