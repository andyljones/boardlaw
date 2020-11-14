import torch
import numpy as np
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
        self.children = self.envs.new_full((world.n_envs, self.n_nodes, n_actions), -1)
        self.parents = self.envs.new_full((world.n_envs, self.n_nodes), -1)
        self.relation = self.envs.new_full((world.n_envs, self.n_nodes), -1)

        #TODO: All of these but the children tensor should not have a 'actions' dim; the field can be attributed
        # to the parent 
        self.log_pi = torch.full((world.n_envs, self.n_nodes, n_actions), np.nan, device=self.device)
        self.valid = torch.full((world.n_envs, self.n_nodes, n_actions), False, device=self.device, dtype=torch.bool)
        self.n = torch.full((world.n_envs, self.n_nodes, n_actions), 0, device=self.device, dtype=torch.int)
        self.w = torch.full((world.n_envs, self.n_nodes, n_actions, self.n_seats), np.nan, device=self.device)
        self.terminal = torch.full((world.n_envs, self.n_nodes), False, device=self.device, dtype=torch.bool)
        self.rewards = torch.full((world.n_envs, self.n_nodes, self.n_seats), 0., device=self.device, dtype=torch.float)
        self.seats = torch.full((world.n_envs, self.n_nodes), -1, device=self.device, dtype=torch.long)

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

    def play(self, world, trace):
        #TODO: There's a bunch of weirdness here around envs that terminate early. Would
        # be a lot better to implement maskable envs.
        world = world.clone()
        transition = arrdict.arrdict(
            terminal=torch.zeros_like(self.terminal[:, self.sim]),
            rewards=torch.zeros_like(self.rewards[:, self.sim]))
        for a in trace:
            active = a != -1
            world[active], transition[active] = world[active].step(a[active])

        return world, transition

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

    def initialize(self, world, agent):
        self.valid[:, self.sim] = world.valid
        self.seats[:, self.sim] = world.seats
        self.terminal[:, self.sim] = False
        self.rewards[:, self.sim] = 0

        decisions = agent(world, value=True)
        self.log_pi[:, self.sim] = decisions.logits
        self.w[:, self.sim] = 0.
        self.n[:, self.sim] = 0 

        self.sim += 1

    def simulate(self, world, agent):
        if self.sim >= self.n_nodes:
            raise ValueError('Called simulate more times than were declared in the constructor')

        trace, leaf, final_actions = self.descend()
        self.children[self.envs, leaf, final_actions] = self.sim
        self.parents[self.envs, self.sim] = leaf
        self.relation[self.envs, self.sim] = final_actions

        world, transition = self.play(world, trace)

        self.valid[:, self.sim] = world.valid
        self.seats[:, self.sim] = world.seats
        self.terminal[:, self.sim] = transition.terminal
        self.rewards[:, self.sim] = transition.rewards

        decisions = agent(world, value=True)
        self.log_pi[:, self.sim] = decisions.logits
        self.w[:, self.sim] = 0
        self.n[:, self.sim] = 0 

        self.backup(self.sim, decisions.v)

        self.sim += 1

    def root(self):
        p = self.n[:, 0].float()/self.n[:, 0].sum(-1, keepdims=True)

        q = self.w[:, 0, :]/self.n[:, 0, ..., None]
        q[p == 0] = 0.

        #TODO: Is this how I should be evaluating root value?
        # Not actually used in AlphaZero at all, but it's nice to have around for validation
        v = (q*p[..., None]).sum(1)

        return arrdict.arrdict(
            logits=torch.log(p),
            v=v)

    def display(self, e=0):
        import networkx as nx
        import matplotlib.pyplot as plt

        root_seat = self.seats[e, 0]

        edges, labels, edge_vals = [], {}, {}
        ws = self.w[e, ..., root_seat]
        ns = self.n[e]
        qs = (ws/ns).where(ns > 0, torch.zeros_like(ws)).cpu()
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

def mcts(world, agent, **kwargs):
    mcts = MCTS(world, **kwargs)

    mcts.initialize(world, agent)
    for _ in range(mcts.n_nodes-1):
        mcts.simulate(world, agent)

    return mcts

class MCTSAgent:

    def __init__(self, env, evaluator, **kwargs):
        self.env = env
        self.evaluator = evaluator
        self.kwargs = kwargs

    def __call__(self, inputs, value=True):
        m = mcts(self.env, inputs, self.evaluator, **self.kwargs)
        r = m.root()
        return arrdict.arrdict(
            logits=r.logits,
            v=r.v,
            actions=torch.distributions.Categorical(logits=r.logits).sample())

from . import validation 

def test_trivial():
    env = validation.InstantWin(device='cpu')
    agent = validation.ProxyAgent()

    inputs = env.reset()

    m = mcts(env, inputs, agent, n_nodes=3)

    expected = torch.tensor([[+1.]], device=env.device)
    torch.testing.assert_allclose(m.root().v, expected)

def test_two_player():
    env = validation.FirstWinsSecondLoses(device='cpu')
    agent = validation.ProxyAgent()
    inputs = env.reset()

    m = mcts(env, inputs, agent, n_nodes=3)

    expected = torch.tensor([[+1., -1.]], device=env.device)
    torch.testing.assert_allclose(m.root().v, expected)

def test_depth():
    env = validation.AllOnes(length=4, device='cpu')
    agent = validation.ProxyAgent()
    inputs = env.reset()

    m = mcts(env, inputs, agent, n_nodes=15)

    expected = torch.tensor([[1/8.]], device=env.device)
    torch.testing.assert_allclose(m.root().v, expected)

def test_multienv():
    # Need to use a fairly complex env here to make sure we've not got 
    # any singleton dims hanging around internally. They can really ruin
    # a tester's day. 
    env = validation.AllOnes(n_envs=2, length=3)
    agent = validation.ProxyAgent()
    inputs = env.reset()

    m = mcts(env, inputs, agent, n_nodes=15)

    expected = torch.tensor([[1/8.], [1/8.]], device=env.device)
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
    m = full_game_mcts(competitive, 31, 1, c_puct=1.)
    expected = torch.tensor([[-1., +1.]], device=m.device)
    torch.testing.assert_allclose(m.root().v, expected)
    m.root().logits

def test_full_game():
    env = hex.Hex(boardsize=3, device='cpu')
    black = MCTSAgent(env, validation.RandomRolloutAgent(env, 4), n_nodes=16, c_puct=.5)
    white = validation.RandomAgent(env)
    trace, inputs = validation.rollout(env, [black, white], 128)
    trace.responses.rewards.sum(0)/trace.responses.rewards.abs().sum(0)

def benchmark():
    import pandas as pd
    import aljpy
    import matplotlib.pyplot as plt

    results = []
    for n in np.logspace(0, 14, 15, base=2, dtype=int):
        torch.cuda.synchronize()
        with aljpy.timer() as timer:
            env = hex.Hex(n_envs=n, boardsize=3, device='cuda')
            black = MCTSAgent(env, validation.RandomAgent(env), n_nodes=16, c_puct=.5)
            white = validation.RandomAgent(env)
            trace, inputs = validation.rollout(env, [black, white], 16)
            torch.cuda.synchronize()
        results.append({'n_envs': n, 'runtime': timer.time(), 'samples': trace.inputs.seats.nelement()})
        print(results[-1])
    df = pd.DataFrame(results)
        
    with plt.style.context('seaborn-poster'):
        ax = df.plot.scatter('n_envs', 'runtime', zorder=2)
        ax.set_xscale('log', base=2)
        ax.set_xlim(1, 2**14)
        ax.set_title('scaling of runtime w/ env count')
        ax.grid(True, zorder=1, alpha=.25)