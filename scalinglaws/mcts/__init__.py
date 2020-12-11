from contextlib import contextmanager
from multiprocessing import Value
import numpy as np
import torch
from rebar import arrdict
from . import search
import logging

log = logging.getLogger(__name__)

def dirichlet_noise(logits, valid, alpha=None, eps=.25):
    alpha = alpha or 10/logits.size(-1)
    alpha = torch.full((valid.shape[-1],), alpha, dtype=torch.float, device=logits.device)
    dist = torch.distributions.Dirichlet(alpha)
    
    draw = dist.sample(logits.shape[:-1])

    # This gives us a Dirichlet draw over the valid values
    draw[~valid] = 0.
    draw = draw/draw.sum(-1, keepdims=True)

    return (logits.exp()*(1 - eps) + draw*eps).log()


CACHE = None

class MCTS:

    def __init__(self, world, n_nodes, c_puct=2.5, noise_eps=.05):
        self.device = world.device
        self.n_envs = world.n_envs
        self.n_nodes = n_nodes
        self.n_seats = world.n_seats
        self.noise_eps = noise_eps
        assert n_nodes > 1, 'MCTS requires at least two nodes'

        self.envs = torch.arange(world.n_envs, device=self.device)

        self.n_actions = np.prod(world.action_space)
        self.tree = arrdict.arrdict(
            children=self.envs.new_full((world.n_envs, self.n_nodes, self.n_actions), -1),
            parents=self.envs.new_full((world.n_envs, self.n_nodes), -1),
            relation=self.envs.new_full((world.n_envs, self.n_nodes), -1))

        self.worlds = arrdict.stack([world for _ in range(self.n_nodes)], 1)
        
        self.transitions = arrdict.arrdict(
            rewards=torch.full((world.n_envs, self.n_nodes, self.n_seats), 0., device=self.device, dtype=torch.float),
            terminal=torch.full((world.n_envs, self.n_nodes), False, device=self.device, dtype=torch.bool))

        self.decisions = arrdict.arrdict(
            logits=torch.full((world.n_envs, self.n_nodes, self.n_actions), np.nan, device=self.device),
            v=torch.full((world.n_envs, self.n_nodes, self.n_seats), np.nan, device=self.device))

        self.stats = arrdict.arrdict(
            n=torch.full((world.n_envs, self.n_nodes), 0, device=self.device, dtype=torch.int),
            w=torch.full((world.n_envs, self.n_nodes, self.n_seats), 0., device=self.device))

        self.sim = torch.tensor(0, device=self.device, dtype=torch.long)
        self.worlds[:, 0] = world

        # https://github.com/LeelaChessZero/lc0/issues/694
        # Larger c_puct -> greater regularization
        self.c_puct = c_puct

    def initialize(self, evaluator):
        world = self.worlds[:, 0]
        decisions = evaluator(world, value=True)
        self.decisions.logits[:, self.sim] = dirichlet_noise(decisions.logits, world.valid, eps=self.noise_eps)
        self.decisions.v[:, 0] = decisions.v

        self.sim += 1

    def action_stats(self, envs, nodes):
        children = self.tree.children[envs, nodes]
        mask = (children != -1)
        stats = self.stats[envs[:, None].expand_as(children), children]
        n = stats.n.where(mask, torch.zeros_like(stats.n))
        w = stats.w.where(mask[..., None], torch.zeros_like(stats.w))

        q = w/n[..., None]

        # Q scaling + pessimistic initialization
        q[n == 0] = 0 
        q = (q - q.min())/(q.max() - q.min() + 1e-6)
        q[n == 0] = 0 

        return q, n

    def policy(self, envs, nodes):
        pi = self.decisions.logits[envs, nodes].exp()

        # https://arxiv.org/pdf/2007.12509.pdf
        worlds = self.worlds[envs, nodes]
        q, n = self.action_stats(envs, nodes)

        seats = worlds.seats[:, None, None].expand(-1, q.size(1), -1)
        q = q.gather(2, seats.long()).squeeze(-1)

        # N == 0 leads to nans, so let's clamp it at 1
        N = n.sum(-1).clamp(1, None)
        lambda_n = self.c_puct*N/(self.n_actions + N)

        soln = search.solve_policy(pi, q, lambda_n)

        return soln.policy

    def sample(self, env, nodes):
        probs = self.policy(env, nodes)
        return torch.distributions.Categorical(probs=probs).sample()

    def _descend_state_collect(self):
        print(CACHE)
        if CACHE is not None:
            state = arrdict.arrdict(
                logits=self.decisions.logits,
                seats=self.worlds.seats,
                terminal=self.transitions.terminal,
                children=self.tree.children,
                w=self.stats.w,
                n=self.stats.n,
                c_puct=torch.as_tensor(self.c_puct))

            CACHE.append(state.clone().detach().cpu())

    def descend(self):
        # What does this use?
        # * descent: envs, logits, terminal, children
        # * policy: logits, seats, c_puct, n_actions
        # * action_stats: children, n, w
        # 
        # So all together:
        # * substantial: logits, terminal, children, seats, n, w
        # * trivial: envs, n, w

        self._descend_state_collect()

        current = torch.full_like(self.envs, 0)
        actions = torch.full_like(self.envs, -1)
        parents = torch.full_like(self.envs, 0)

        while True:
            interior = ~torch.isnan(self.decisions.logits[self.envs, current]).any(-1)
            terminal = self.transitions.terminal[self.envs, current]
            active = interior & ~terminal
            if not active.any():
                break

            e, c = self.envs[active], current[active]
            sampled = self.sample(e, c)
            actions[active] = sampled
            parents[active] = c
            current[active] = self.tree.children[e, c, sampled]

        return parents, actions

    def backup(self, leaves):
        current = leaves.clone()
        v = self.decisions.v[self.envs, leaves]
        while True:
            active = (current != -1)
            if not active.any():
                break

            e, c = self.envs[active], current[active]
            
            t = self.transitions.terminal[e, c]
            v[e[t]] = 0. 
            v[active] += self.transitions.rewards[e, c]

            self.stats.n[e, c] += 1
            self.stats.w[e, c] += v[active]
        
            current[active] = self.tree.parents[e, c]

    def simulate(self, evaluator):
        if self.sim >= self.n_nodes:
            raise ValueError('Called simulate more times than were declared in the constructor')

        parents, actions = self.descend()

        # If the transition is terminal - and so we stopped our descent early
        # we don't want to end up creating a new node. 
        leaves = self.tree.children[self.envs, parents, actions]
        leaves[leaves == -1] = self.sim

        self.tree.children[self.envs, parents, actions] = leaves
        self.tree.parents[self.envs, leaves] = parents
        self.tree.relation[self.envs, leaves] = actions

        old_world = self.worlds[self.envs, parents]
        world, transition = old_world.step(actions)

        self.worlds[self.envs, leaves] = world
        self.transitions[self.envs, leaves] = transition

        with torch.no_grad():
            decisions = evaluator(world, value=True)
        self.decisions.logits[self.envs, leaves] = decisions.logits
        self.decisions.v[self.envs, leaves] = decisions.v

        self.backup(leaves)

        self.sim += 1

    def root(self):
        root = torch.zeros_like(self.envs)
        q, n = self.action_stats(self.envs, root)
        p = self.policy(self.envs, root)

        return arrdict.arrdict(
            logits=torch.log(p),
            v=self.decisions.v[:, 0])
    
    def n_leaves(self):
        return ((self.tree.children == -1).all(-1) & (self.tree.parents != -1)).sum(-1)

    def display(self, e=0):
        import networkx as nx
        import matplotlib.pyplot as plt

        root_seat = self.worlds[:, 0].seats[e]

        ws = self.stats.w[e, ..., root_seat]
        ns = self.stats.n[e]
        qs = (ws/ns).where(ns > 0, torch.zeros_like(ws)).cpu()
        q_min, q_max = np.nanmin(qs), np.nanmax(qs)

        nodes, edges = {}, {}
        for i in range(self.sim):
            p = int(self.tree.parents[e, i].cpu())
            if (i == 0) or (p >= 0):
                t = self.transitions.terminal[e, i].cpu().numpy()
                if i == 0:
                    color = 'C0'
                elif t:
                    color = 'C3'
                else:
                    color = 'C2'
                nodes[i] = {
                    'label': f'{i}', 
                    'color': color}
            
            if p >= 0:
                r = int(self.tree.relation[e, i].cpu())
                q, n = float(qs[i]), int(ns[i])
                edges[(p, i)] = {
                    'label': f'{r}\n{q:.2f}, {n}',
                    'color':  (q - q_min)/(q_max - q_min + 1e-6)}

        G = nx.from_edgelist(edges)

        pos = nx.kamada_kawai_layout(G)
        nx.draw(G, pos, 
            node_color=[nodes[i]['color'] for i in G.nodes()],
            edge_color=[edges[e]['color'] for e in G.edges()], width=5)
        nx.draw_networkx_edge_labels(G, pos, font_size='x-small',
            edge_labels={e: d['label'] for e, d in edges.items()})
        nx.draw_networkx_labels(G, pos, 
            labels={i: d['label'] for i, d in nodes.items()})

        return plt.gca()

def mcts(worlds, evaluator, **kwargs):
    mcts = MCTS(worlds, **kwargs)

    mcts.initialize(evaluator)
    for _ in range(mcts.n_nodes-1):
        mcts.simulate(evaluator)

    return mcts

class MCTSAgent:

    def __init__(self, evaluator, **kwargs):
        self.evaluator = evaluator
        self.kwargs = kwargs

    def __call__(self, world, value=True, **kwargs):
        m = mcts(world, self.evaluator, **self.kwargs, **kwargs)
        r = m.root()
        return arrdict.arrdict(
            logits=r.logits,
            n_sims=torch.full_like(m.envs, m.sim+1),
            n_leaves=m.n_leaves(),
            v=r.v,
            actions=torch.distributions.Categorical(logits=r.logits).sample())

    @contextmanager
    def no_noise(self):
        kwargs = self.kwargs
        self.kwargs = {**kwargs, 'noise_eps': 0.}
        yield
        self.kwargs = kwargs

    def load_state_dict(self, sd):
        #TODO: Systematize this
        evaluator = {k[10:]: v for k, v in sd.items() if k.startswith('evaluator.')}
        kwargs = {k[7:]: v for k, v in sd.items() if k.startswith('kwargs.')}
        self.evaluator.load_state_dict(evaluator)
        self.kwargs.update(kwargs)

    def state_dict(self):
        evaluator = {f'evaluator.{k}': v for k, v in self.evaluator.state_dict().items()}
        kwargs = {f'kwargs.{k}': v for k, v in self.kwargs.items()}
        return {**evaluator, **kwargs}


from .. import validation, analysis

#TODO: The 'v' all need to be rewritten to test something else.
def test_trivial():
    world = validation.Win.initial(device='cpu')
    agent = validation.ProxyAgent()

    m = mcts(world, agent, n_nodes=3)

    expected = torch.tensor([[+1.]], device=world.device)
    torch.testing.assert_allclose(m.root().v, expected)

def test_two_player():
    world = validation.WinnerLoser.initial(device='cpu')
    agent = validation.ProxyAgent()

    m = mcts(world, agent, n_nodes=3)

    expected = torch.tensor([[+1., -1.]], device=world.device)
    torch.testing.assert_allclose(m.root().v, expected)

def test_depth():
    world = validation.All.initial(length=3, device='cpu')
    agent = validation.ProxyAgent()

    m = mcts(world, agent, n_nodes=15)

    expected = torch.tensor([[1/8.]], device=world.device)
    torch.testing.assert_allclose(m.root().v, expected)

def test_multienv():
    # Need to use a fairly complex env here to make sure we've not got 
    # any singleton dims hanging around internally. They can really ruin
    # a tester's day. 
    world = validation.All.initial(n_envs=2, length=3)
    agent = validation.ProxyAgent()

    m = mcts(world, agent, n_nodes=15)

    expected = torch.tensor([[1/8.], [1/8.]], device=world.device)
    torch.testing.assert_allclose(m.root().v, expected)

def full_game_mcts(s, n_nodes, n_rollouts, **kwargs):
    from . import hex
    world = hex.from_string(s, device='cpu')
    agent = validation.MonteCarloAgent(n_rollouts)
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

def test_full_game():
    from . import hex
    world = hex.Hex.initial(1, boardsize=3, device='cpu')
    black = MCTSAgent(validation.RandomRolloutAgent(4), n_nodes=16, c_puct=.5)
    white = validation.RandomAgent()
    trace = analysis.rollout(world, [black, white], 128)
    winrates = trace.trans.rewards.sum(0).sum(0)/trace.trans.terminal.sum(0).sum(0)

def benchmark_mcts(T=16):
    import pandas as pd
    import aljpy
    import matplotlib.pyplot as plt
    from . import hex

    results = []
    for n in np.logspace(0, 14, 15, base=2, dtype=int):
        env = hex.Hex.initial(n_envs=n, boardsize=3, device='cuda')
        black = MCTSAgent(validation.RandomAgent(), n_nodes=16)
        white = validation.RandomAgent()

        torch.cuda.synchronize()
        with aljpy.timer() as timer:
            trace = analysis.rollout(env, [black, white], 16)
            torch.cuda.synchronize()
        results.append({'n_envs': n, 'runtime': timer.time(), 'samples': T*n})
        print(results[-1])
    df = pd.DataFrame(results)
        
    with plt.style.context('seaborn-poster'):
        ax = df.plot.scatter('n_envs', 'runtime', zorder=2)
        ax.set_xscale('log', base=2)
        ax.set_xlim(1, 2**14)
        ax.set_title('scaling of runtime w/ env count')
        ax.grid(True, zorder=1, alpha=.25)