from contextlib import contextmanager
from multiprocessing import Value
import numpy as np
import torch
import torch.distributions
from rebar import arrdict
from . import cuda
import logging
from rebar import profiling

log = logging.getLogger(__name__)

def dirichlet_noise(logits, valid, eps, alpha=None):
    alpha = alpha or 10/logits.size(-1)
    alpha = torch.full((valid.shape[-1],), alpha, dtype=torch.float, device=logits.device)
    dist = torch.distributions.Dirichlet(alpha)
    
    draw = dist.sample(logits.shape[:-1])

    # This gives us a Dirichlet draw over the valid values
    draw[~valid] = 0.
    draw = draw/draw.sum(-1, keepdims=True)

    return (logits.exp()*(1 - eps) + draw*eps).log()

class MCTS:

    def __init__(self, world, n_nodes, noise_eps, c_puct=2.5):
        """
        c_puct high: concentrates on prior
        c_puct low: concentrates on value
        """
        self.device = world.device
        self.n_envs = world.n_envs
        self.n_nodes = n_nodes
        self.n_seats = world.n_seats
        self.noise_eps = noise_eps
        assert n_nodes > 1, 'MCTS requires at least two nodes'

        self.envs = torch.arange(world.n_envs, device=self.device)

        self.n_actions = np.prod(world.action_space)
        self.tree = arrdict.arrdict(
            children=self.envs.new_full((world.n_envs, self.n_nodes, self.n_actions), -1, dtype=torch.short),
            parents=self.envs.new_full((world.n_envs, self.n_nodes), -1, dtype=torch.short),
            relation=self.envs.new_full((world.n_envs, self.n_nodes), -1, dtype=torch.short))

        self.worlds = arrdict.stack([world for _ in range(self.n_nodes)], 1)
        
        self.transitions = arrdict.arrdict(
            rewards=torch.full((world.n_envs, self.n_nodes, self.n_seats), 0., device=self.device, dtype=torch.half),
            terminal=torch.full((world.n_envs, self.n_nodes), False, device=self.device, dtype=torch.bool))

        self.decisions = arrdict.arrdict(
            logits=torch.full((world.n_envs, self.n_nodes, self.n_actions), np.nan, device=self.device, dtype=torch.half),
            v=torch.full((world.n_envs, self.n_nodes, self.n_seats), np.nan, device=self.device, dtype=torch.half))

        self.stats = arrdict.arrdict(
            n=torch.full((world.n_envs, self.n_nodes), 0, device=self.device, dtype=torch.short),
            w=torch.full((world.n_envs, self.n_nodes, self.n_seats), 0., device=self.device, dtype=torch.half))

        self.sim = torch.tensor(0, device=self.device, dtype=torch.long)
        self.worlds[:, 0] = world

        # https://github.com/LeelaChessZero/lc0/issues/694
        # Larger c_puct -> greater regularization
        self.c_puct = torch.full((world.n_envs,), c_puct, device=self.device, dtype=torch.half)

    def initialize(self, network):
        world = self.worlds[:, 0]
        with torch.no_grad():
            decisions = network(world)
        self.decisions.logits[:, self.sim] = dirichlet_noise(decisions.logits, world.valid, self.noise_eps)
        self.decisions.v[:, 0] = decisions.v

        self.sim += 1

    def _cuda(self):
        return cuda.mcts(
            self.decisions.logits, 
            self.stats.w, 
            self.stats.n, 
            self.c_puct, 
            self.worlds.seats, 
            self.transitions.terminal, 
            self.tree.children)

    @profiling.nvtx
    def descend(self):
        result = cuda.descend(self._cuda())
        return result.parents.long(), result.actions.long()

    @profiling.nvtx
    def backup(self, leaves):
        bk = cuda.Backup(
            v=self.decisions.v,
            w=self.stats.w,
            n=self.stats.n,
            rewards=self.transitions.rewards,
            parents=self.tree.parents,
            terminal=self.transitions.terminal)
        cuda.backup(bk, leaves.short())

    @profiling.nvtx
    def simulate(self, network):
        if self.sim >= self.n_nodes:
            raise ValueError('Called simulate more times than were declared in the constructor')

        parents, actions = self.descend()

        # If the transition is terminal - and so we stopped our descent early
        # we don't want to end up creating a new node. 
        leaves = self.tree.children[self.envs, parents, actions].long()
        leaves[leaves == -1] = self.sim

        self.tree.children[self.envs, parents, actions] = leaves.short()
        self.tree.parents[self.envs, leaves] = parents.short()
        self.tree.relation[self.envs, leaves] = actions.short()

        old_world = self.worlds[self.envs, parents]
        world, transition = old_world.step(actions)

        self.worlds[self.envs, leaves] = world
        self.transitions.rewards[self.envs, leaves] = transition.rewards.half()
        self.transitions.terminal[self.envs, leaves] = transition.terminal

        with torch.no_grad(), torch.cuda.amp.autocast():
            decisions = network(world)
        self.decisions.logits[self.envs, leaves] = decisions.logits.half()
        self.decisions.v[self.envs, leaves] = decisions.v.half()

        self.backup(leaves)

        self.sim += 1

    @profiling.nvtx
    def root(self):
        #TODO: Bit of a hack this 
        return arrdict.arrdict(
            logits=cuda.root(self._cuda()).log(),
            prior=self.decisions.logits[:, 0], # useful for monitoring KL div
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

def mcts(worlds, network, **kwargs):
    mcts = MCTS(worlds, **kwargs)

    mcts.initialize(network)
    for _ in range(mcts.n_nodes-1):
        mcts.simulate(network)

    return mcts

class MCTSAgent:

    def __init__(self, network, noise_eps=.05, **kwargs):
        self.network = network
        self.kwargs = kwargs
        self.noise_eps = noise_eps

    @profiling.nvtx
    def __call__(self, world, value=True, eval=False, **kwargs):
        noise_eps = 0. if eval else self.noise_eps 
        m = mcts(world, self.network, noise_eps=noise_eps, **{**self.kwargs, **kwargs})
        r = m.root()

        # Need to go back to float here because `sample` doesn't like halves
        actions = r.logits.argmax(-1) if eval else torch.distributions.Categorical(logits=r.logits.float()).sample()

        return arrdict.arrdict(
            logits=r.logits,
            prior=r.prior,
            n_sims=torch.full_like(m.envs, m.sim+1),
            n_leaves=m.n_leaves(),
            v=r.v,
            actions=actions).clone()

    def load_state_dict(self, sd):
        #TODO: Systematize this
        network = {k[10:]: v for k, v in sd.items() if k.startswith('network.')}
        kwargs = {k[7:]: v for k, v in sd.items() if k.startswith('kwargs.')}
        self.network.load_state_dict(network)
        self.kwargs.update(kwargs)
        self.noise_eps = sd['noise_eps']

    def state_dict(self):
        network = {f'network.{k}': v for k, v in self.network.state_dict().items()}
        kwargs = {f'kwargs.{k}': v for k, v in self.kwargs.items()}
        return {**network, **kwargs, 'noise_eps': self.noise_eps}

