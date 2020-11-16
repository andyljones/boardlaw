import numpy as np
import torch
from rebar import arrdict

class MCTS:

    def __init__(self, world, n_nodes, c_puct=2.5): 
        self.device = world.device
        self.n_envs = world.n_envs
        self.n_nodes = n_nodes
        self.n_seats = world.n_seats
        assert n_nodes > 1, 'MCTS requires at least two nodes'

        self.envs = torch.arange(world.n_envs, device=self.device)

        n_actions = np.prod(world.action_space)
        self.tree = arrdict.arrdict(
            children=self.envs.new_full((world.n_envs, self.n_nodes, n_actions), -1),
            parents=self.envs.new_full((world.n_envs, self.n_nodes), -1),
            relation=self.envs.new_full((world.n_envs, self.n_nodes), -1))

        self.worlds = arrdict.stack([world for _ in range(self.n_nodes)], 1)
        
        self.transitions = arrdict.arrdict(
            rewards=torch.full((world.n_envs, self.n_nodes, self.n_seats), 0., device=self.device, dtype=torch.float),
            terminal=torch.full((world.n_envs, self.n_nodes), False, device=self.device, dtype=torch.bool))

        self.log_pi = torch.full((world.n_envs, self.n_nodes, n_actions), np.nan, device=self.device)

        self.stats = arrdict.arrdict(
            n=torch.full((world.n_envs, self.n_nodes), 0, device=self.device, dtype=torch.int),
            w=torch.full((world.n_envs, self.n_nodes, self.n_seats), 0., device=self.device))

        self.sim = 0
        self.worlds[:, 0] = world

        # https://github.com/LeelaChessZero/lc0/issues/694
        self.c_puct = c_puct

    def action_stats(self, envs, nodes):
        children = self.tree.children[envs, nodes]
        mask = (children != -1)
        stats = self.stats[envs, children]
        n = stats.n.where(mask, torch.zeros_like(stats.n))
        w = stats.w.where(mask[..., None], torch.zeros_like(stats.w))

        q = w/n[..., None]
        q[n == 0] = 0.

        return q, n

    def sample(self, envs, nodes):
        worlds = self.worlds[envs, nodes]
        q, n = self.action_stats(envs, nodes)

        pi = self.log_pi[envs, nodes].exp()
        q = q[envs, :, worlds.seats.long()]

        N = n.sum(-1, keepdims=True)

        values = q + self.c_puct*pi*N/(1 + n)
        values[~worlds.valid] = -np.inf
        return values.max(-1).indices

    def initialize(self, agent):
        decisions = agent(self.worlds[:, 0], value=True)
        self.log_pi[:, self.sim] = decisions.logits

        #TODO: Should this be v or zero?
        self.stats.w[:, self.sim] = decisions.v
        self.stats.n[:, self.sim] = 1 

        self.sim += 1

    def descend(self):
        current = torch.full_like(self.envs, 0)
        actions = torch.full_like(self.envs, -1)
        leaves = torch.full_like(self.envs, 0)

        while True:
            interior = ~torch.isnan(self.log_pi[self.envs, current]).any(-1)
            terminal = self.transitions.terminal[self.envs, current]
            active = interior & ~terminal
            if not active.any():
                break

            actions[active] = self.sample(self.envs[active], current[active])
            leaves[active] = current[active]
            current[active] = self.tree.children[self.envs[active], current[active], actions[active]]
        
        return leaves, actions

    def backup(self, v):
        v = v.clone()
        current = torch.full_like(self.envs, self.sim)
        while True:
            active = (current != -1)
            if not active.any():
                break

            self.stats.n[self.envs[active], current[active]] += 1
            self.stats.w[self.envs[active], current[active]] += v[active]

            t = self.transitions.terminal[self.envs[active], current[active]]
            v[self.envs[active][t]] = 0. 
            v[active] += self.transitions.rewards[self.envs[active], current[active]]
        
            current[active] = self.tree.parents[self.envs[active], current[active]]

    def simulate(self, agent):
        if self.sim >= self.n_nodes:
            raise ValueError('Called simulate more times than were declared in the constructor')

        leaves, actions = self.descend()
        self.tree.children[self.envs, leaves, actions] = self.sim
        self.tree.parents[:, self.sim] = leaves
        self.tree.relation[:, self.sim] = actions

        old_world = self.worlds[self.envs, leaves]
        world, transition = old_world.step(actions)

        self.worlds[:, self.sim] = world
        self.transitions[:, self.sim] = transition

        decisions = agent(world, value=True)
        self.log_pi[:, self.sim] = decisions.logits

        self.backup(decisions.v)

        self.sim += 1

    def root(self):
        q, n = self.action_stats(self.envs, torch.zeros_like(self.envs))
        p = n.float()/n.sum(-1, keepdims=True)

        #TODO: Is this how I should be evaluating root value?
        # Not actually used in AlphaZero at all, but it's nice to have around for validation
        v = (q*p[..., None]).sum(1)

        return arrdict.arrdict(
            logits=torch.log(p),
            v=v)

    def display(self, e=0):
        import networkx as nx
        import matplotlib.pyplot as plt

        root_seat = self.worlds[:, 0].seats[e]

        edges, labels, edge_vals = [], {}, {}
        ws = self.stats.w[e, ..., root_seat]
        ns = self.stats.n[e]
        qs = (ws/ns).where(ns > 0, torch.zeros_like(ws)).cpu()
        q_min, q_max = np.nanmin(qs), np.nanmax(qs)
        for i, p in enumerate(self.tree.parents[e].cpu()):
            p = int(p)
            if p >= 0:
                r = int(self.tree.relation[e, i].cpu())
                q = float(qs[i])
                n = int(ns[i])
                edge = (p, i)
                edges.append(edge)
                labels[edge] = f'{r}\n{q:.2f}, {n}'
                edge_vals[edge] = (q - q_min)/(q_max - q_min + 1e-6)
            
        G = nx.from_edgelist(edges)
        colors = ['C1' if t else 'C2' for t in self.transitions.terminal.float()[e].cpu().numpy()]
        edge_colors = [edge_vals[e] for e in G.edges()]

        pos = nx.kamada_kawai_layout(G)
        nx.draw(G, pos, node_color=colors, edge_color=edge_colors, width=5)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size='x-small')
        nx.draw_networkx_labels(G, pos, labels={i: i for i in range(self.n_nodes)})

        return plt.gca()

def mcts(world, agent, **kwargs):
    mcts = MCTS(world, **kwargs)

    mcts.initialize(agent)
    for _ in range(mcts.n_nodes-1):
        mcts.simulate(agent)

    return mcts

class MCTSAgent:

    def __init__(self, evaluator, **kwargs):
        self.evaluator = evaluator
        self.kwargs = kwargs

    def __call__(self, world, value=True):
        m = mcts(world, self.evaluator, **self.kwargs)
        r = m.root()
        return arrdict.arrdict(
            logits=r.logits,
            v=r.v,
            actions=torch.distributions.Categorical(logits=r.logits).sample())

from . import validation, analysis

def test_trivial():
    world = validation.InstantWin(envs=torch.arange(1, device='cpu'))
    agent = validation.ProxyAgent()

    m = mcts(world, agent, n_nodes=3)

    expected = torch.tensor([[+1.]], device=world.device)
    torch.testing.assert_allclose(m.root().v, expected)

def full_game_mcts(s, n_nodes, n_rollouts, **kwargs):
    from . import hex
    world = hex.from_string(s, device='cpu')
    agent = validation.RandomRolloutAgent(n_rollouts)
    return mcts(world, agent, n_nodes=n_nodes, **kwargs)

def test_planted_game():
    black_wins = """
    bwb
    wbw
    ...
    """
    m = full_game_mcts(black_wins, 17, 1)
    expected = torch.tensor([[+1., -1.]], device=m.device)
    torch.testing.assert_allclose(m.root().v, expected)

    white_wins = """
    wb.
    bw.
    wbb
    """
    m = full_game_mcts(white_wins, 4, 1)
    expected = torch.tensor([[-1., +1.]], device=m.device)
    torch.testing.assert_allclose(m.root().v, expected)

    # Hard to validate the logits
    competitive = """
    wb.
    bw.
    wb.
    """
    m = full_game_mcts(competitive, 31, 4, c_puct=100.)
    expected = torch.tensor([[-1/3., +1/3.]], device=m.device)
    assert ((m.root().v - expected).abs() < 1/3).all()