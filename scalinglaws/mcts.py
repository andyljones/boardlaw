from multiprocessing import Value
import numpy as np
import torch
from rebar import arrdict
import logging

log = logging.getLogger(__name__)

def safe_div(x, y):
    r = x/y
    r[x == 0] = 0
    return r

def newton_search(f, grad, x0, tol=1e-3):
    # Some guidance on what's going on here:
    # * While the Regularized MCTS paper recommends binary search, it turns out to be pretty slow and - thanks
    #   to the numerical errors that show up when you run this whole thing in float32 - tricky to implemnt.
    # * What works better is to exploit the geometry of the problem. The error function is convex and 
    #   descends from an asymptote somewhere to the left of x0. 
    # * By taking Newton steps from x0, we head right and so don't run into any more asymptotes.
    # * More, because the error is convex, Newton's is guaranteed to undershoot the solution.
    # * The only thing we need to be careful about is when we start near the asymptote. In that case the gradient
    #   is really large and it's possible that - again, thanks to numerical issues - our steps 
    #   *won't actually change the error*. I couldn't think of a good solution to this, but so far in practice it
    #   turns out 'just giving up' works pretty well. It's a rare occurance - 1 in 40k samples of the benchmark
    #   run I did - and 'just give up' only missed the specified error tol by a small amount.
    x = x0.clone()
    y = torch.zeros_like(x)
    while True:
        y_new = f(x)
        done = (y_new.abs() < tol) | (y == y_new)
        if done.all():
            return x
        y = y_new

        x[~done] = (x - y/grad(x))[~done]

def solve_policy(pi, q, lambda_n):
    assert (lambda_n > 0).all(), 'Don\'t currently support zero lambda_n'
    alpha_min = (q + lambda_n[:, None]*pi).max(-1).values

    policy = lambda alpha: safe_div(lambda_n[:, None]*pi, alpha[:, None] - q)
    error = lambda alpha: policy(alpha).sum(-1) - 1
    grad = lambda alpha: -safe_div(lambda_n[:, None]*pi, (alpha[:, None] - q).pow(2)).sum(-1)

    alpha_star = newton_search(error, grad, alpha_min)

    p = policy(alpha_star)

    return arrdict.arrdict(
        policy=p,
        alpha_min=alpha_min, 
        alpha_star=alpha_star,
        error=p.sum(-1) - 1)

def test_policy():
    # Case when the root is at the lower bound
    pi = torch.tensor([[.999, .001]])
    q = torch.tensor([[0., 1.]])
    lambda_n = torch.tensor([[.1]])
    soln = solve_policy(pi, q, lambda_n)
    torch.testing.assert_allclose(soln.alpha_star, torch.tensor([[1.]]), rtol=.001, atol=.001)

    # Case when the root is at the upper bound
    pi = torch.tensor([[.5, .5]])
    q = torch.tensor([[1., 1.]])
    lambda_n = torch.tensor([[.1]])
    soln = solve_policy(pi, q, lambda_n)
    torch.testing.assert_allclose(soln.alpha_star, torch.tensor([[1.1]]), rtol=.001, atol=.001)

    # Case when the root is at the upper bound
    pi = torch.tensor([[.25, .75]])
    q = torch.tensor([[1., .25]])
    lambda_n = torch.tensor([[.5]])
    soln = solve_policy(pi, q, lambda_n)
    torch.testing.assert_allclose(soln.alpha_star, torch.tensor([[1.205]]), rtol=.001, atol=.001)
    
def benchmark_search():
    import aljpy
    Ds = torch.load('output/search/benchmark.pkl')
    ds, solns = [], []
    torch.cuda.synchronize()
    with aljpy.timer() as timer:
        for i, d in enumerate(Ds):
            ds.append(arrdict.arrdict(d))
            solns.append(solve_policy(**d))
        torch.cuda.synchronize()
    print(f'{1000*timer.time():.0f}ms')
    solns = arrdict.cat(solns)
    ds = arrdict.cat(ds)


class MCTS:

    def __init__(self, world, n_nodes, c_puct=2.5): 
        self.device = world.device
        self.n_envs = world.n_envs
        self.n_nodes = n_nodes
        self.n_seats = world.n_seats
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

        self.log_pi = torch.full((world.n_envs, self.n_nodes, self.n_actions), np.nan, device=self.device)

        self.stats = arrdict.arrdict(
            n=torch.full((world.n_envs, self.n_nodes), 0, device=self.device, dtype=torch.int),
            w=torch.full((world.n_envs, self.n_nodes, self.n_seats), 0., device=self.device))

        self.sim = 0
        self.worlds[:, 0] = world

        # https://github.com/LeelaChessZero/lc0/issues/694
        self.c_puct = c_puct
        assert self.c_puct > 0, 'Zero c_puct is not currently supported, as it\'d start the search at a singularity'

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
        pi = self.log_pi[envs, nodes].exp()

        # https://arxiv.org/pdf/2007.12509.pdf
        worlds = self.worlds[envs, nodes]
        q, n = self.action_stats(envs, nodes)

        seats = worlds.seats[:, None, None].expand(-1, q.size(1), -1)
        q = q.gather(2, seats.long()).squeeze(-1)

        # N == 0 leads to nans, so let's clamp it at 1
        N = n.sum(-1).clamp(1, None)
        lambda_n = self.c_puct*N/(self.n_actions + N)

        soln = solve_policy(pi, q, lambda_n)

        return soln.policy

    def sample(self, env, nodes):
        probs = self.policy(env, nodes)
        return torch.distributions.Categorical(probs=probs).sample()

    def initialize(self, agent):
        decisions = agent(self.worlds[:, 0], value=True)
        self.log_pi[:, self.sim] = decisions.logits

        self.sim += 1

    def descend(self):
        current = torch.full_like(self.envs, 0)
        actions = torch.full_like(self.envs, -1)
        parents = torch.full_like(self.envs, 0)

        while True:
            interior = ~torch.isnan(self.log_pi[self.envs, current]).any(-1)
            terminal = self.transitions.terminal[self.envs, current]
            active = interior & ~terminal
            if not active.any():
                break

            actions[active] = self.sample(self.envs[active], current[active])
            parents[active] = current[active]
            current[active] = self.tree.children[self.envs[active], current[active], actions[active]]

        return parents, actions

    def backup(self, leaves, v):
        current, v = leaves.clone(), v.clone()
        while True:
            active = (current != -1)
            if not active.any():
                break

            t = self.transitions.terminal[self.envs[active], current[active]]
            v[self.envs[active][t]] = 0. 
            v[active] += self.transitions.rewards[self.envs[active], current[active]]

            self.stats.n[self.envs[active], current[active]] += 1
            self.stats.w[self.envs[active], current[active]] += v[active]
        
            current[active] = self.tree.parents[self.envs[active], current[active]]

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
        self.log_pi[self.envs, leaves] = decisions.logits

        self.backup(leaves, decisions.v)

        self.sim += 1

    def root(self):
        root = torch.zeros_like(self.envs)
        q, n = self.action_stats(self.envs, root)
        p = self.policy(self.envs, root)

        #TODO: Is this how I should be evaluating root value?
        # Not actually used in AlphaZero at all, but it's nice to have around for validation
        v = (q*p[..., None]).sum(-2)

        return arrdict.arrdict(
            logits=torch.log(p),
            v=v)

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
    world = validation.InstantWin.initial(device='cpu')
    agent = validation.ProxyAgent()

    m = mcts(world, agent, n_nodes=3)

    expected = torch.tensor([[+1.]], device=world.device)
    torch.testing.assert_allclose(m.root().v, expected)

def test_two_player():
    world = validation.FirstWinsSecondLoses.initial(device='cpu')
    agent = validation.ProxyAgent()

    m = mcts(world, agent, n_nodes=3)

    expected = torch.tensor([[+1., -1.]], device=world.device)
    torch.testing.assert_allclose(m.root().v, expected)

def test_depth():
    world = validation.AllOnes.initial(length=3, device='cpu')
    agent = validation.ProxyAgent()

    m = mcts(world, agent, n_nodes=15)

    expected = torch.tensor([[1/8.]], device=world.device)
    torch.testing.assert_allclose(m.root().v, expected)

def test_multienv():
    # Need to use a fairly complex env here to make sure we've not got 
    # any singleton dims hanging around internally. They can really ruin
    # a tester's day. 
    world = validation.AllOnes.initial(n_envs=2, length=3)
    agent = validation.ProxyAgent()

    m = mcts(world, agent, n_nodes=15)

    expected = torch.tensor([[1/8.], [1/8.]], device=world.device)
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