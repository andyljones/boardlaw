import numpy as np
import torch
from rebar import arrdict

def scatter_inc_(totals, indices):
    assert indices.ndim == 2
    rows, cols = indices.T

    width = totals.shape[1]
    raveled = rows + width*cols

    ones = totals.new_ones((len(rows),))
    totals.view(-1).scatter_add_(0, raveled, ones)

def scatter_inc_symmetric_(totals, indices):
    scatter_inc_(totals, indices)
    scatter_inc_(totals, indices.flip(1))


class Tracker:

    def __init__(self, n_envs, n_envs_per, names, device='cuda'):
        assert n_envs % n_envs_per == 0
        self.n_envs = n_envs
        self.n_envs_per = n_envs_per
        self.names = list(names)

        # Counts games that are either in-progress or that have been completed
        self.games = torch.zeros((len(names), len(names)), dtype=torch.int, device=device)

        self.live = torch.full((n_envs, 2), -1, device=device)

    def _live_counts(self):
        counts = torch.zeros_like(self.games)
        scatter_inc_(counts, self.live[(self.live > -1).all(-1)])
        return counts

    def update(self, terminal, mask):
        # Kill off the finished games
        masked = torch.zeros_like(mask)
        masked[mask] = terminal
        self.live[masked] = -1

    def finished(self):
        return (self.games == self.n_envs_per).all()

    def suggest(self, seats):
        # Figure out how the -1s in live should be repopulated
        while True:
            available = (self.live == -1).any(-1)
            remaining = (self.games < self.n_envs_per)
            if not (available.any() and remaining.any()):
                break

            counts = self._live_counts()
            counts = counts.sum(0, keepdim=True) * counts.sum(1, keepdim=True) 
            counts = counts + counts.T
            goodness = (2*remaining.float() - 1)*counts

            choice = goodness.argmax()
            choice = (choice // len(self.names), choice % len(self.names))

            allocation = available.nonzero(as_tuple=False)[:self.n_envs_per]
            self.live[allocation] = torch.as_tensor(choice, device=allocation.device)

            self.games[choice] += len(allocation)

        # Suggest the most 'popular' agent  
        active = self.live.gather(1, seats[:, None]).squeeze(1)
        totals = torch.zeros_like(self.games[0])
        totals.scatter_add_(0, active, totals.new_ones(active.shape))

        suggestion = totals.argmax()
        mask = (active == suggestion)

        return self.names[int(suggestion)], mask

class MultiEvaluator:
    # Idea: keep lots and lots of envs in memory at once, play 
    # every agent against every agent simultaneously
    
    def __init__(self, worlds, agents):
        pass
    pass

class MockAgent:

    def __init__(self, id):
        self.id = id

    def __call__(self, world):
        id = torch.full((world.n_envs,), self.id, device=world.device, dtype=torch.long)
        actions = torch.empty((world.n_envs, 0), device=world.device, dtype=torch.long)
        return arrdict.arrdict(id=id, actions=actions)

class MockGame(arrdict.namedarrtuple(fields=('count', 'max_count'))):

    @classmethod
    def initial(cls, n_envs=1, max_count=4, device='cuda'):
        return cls(
            max_count=torch.full((n_envs,), max_count, dtype=torch.long, device=device),
            count=torch.full((n_envs,), 0, dtype=torch.long, device=device))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.count, torch.Tensor):
            return 

        self.n_envs = self.count.shape[0]
        self.device = self.count.device 
        self.n_seats = 2

        self.valid = torch.ones(self.count.shape + (2,), dtype=torch.bool, device=self.device)

    @property
    def seats(self):
        return self.count % self.n_seats

    def step(self, actions):
        count = self.count + 1
        terminal = (count == self.max_count)
        transition = arrdict.arrdict(terminal=terminal)

        count[terminal] = 0
        
        world = type(self)(
            count=count,
            max_count=self.max_count.clone())
        return world, transition

def test_tracker():
    n_envs = 8
    n_envs_per = 2

    worlds = MockGame.initial(n_envs)
    agents = {i: MockAgent(i) for i in range(8)}

    tracker = Tracker(n_envs, n_envs_per, agents, worlds.device)

    while True:
        name, mask = tracker.suggest(worlds.seats)
        
        decisions = agents[name](worlds)
        worlds[mask], transitions = worlds[mask].step(decisions)

        tracker.update(transitions.terminal)