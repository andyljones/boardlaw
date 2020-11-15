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
            w=torch.full((world.n_envs, self.n_nodes, self.n_seats), np.nan, device=self.device))

        self.sim = 0

        # https://github.com/LeelaChessZero/lc0/issues/694
        self.c_puct = c_puct

    def sample(self, envs, nodes):
        worlds = self.worlds[envs, nodes]

        pi = self.log_pi[envs, nodes].exp()
        n = self.stats.n[envs, nodes]
        q = self.stats.w[envs, nodes, :, worlds.seats]/n

        N = n.sum(-1, keepdims=True)

        values = q + self.c_puct*pi*N/(1 + n)
        values[~worlds.valid[envs, nodes]] = -np.inf
        return values.max(-1).indices

    def descend(self, world):
        current = torch.full_like(self.envs, -1)
        next = torch.full_like(self.envs, 0)

        actions = torch.full_like(self.envs, -1)
        world = world.clone()
        transition = arrdict.arrdict(
            terminal=torch.zeros_like(self.terminal[:, self.sim]),
            rewards=torch.zeros_like(self.rewards[:, self.sim]))

        #TODO: Handle termination properly
        while True:
            interior = (next != -1)
            if not interior.any():
                break

            current[interior] = next[interior]

            choice = self.sample(self.envs[interior], current[interior])
            actions[interior] = choice

            world[interior], transition[interior] = world[interior].step(choice)

            next = next.clone()
            next[interior] = self.children[self.envs[interior], current[interior], choice]

        return current, actions, transition, worl