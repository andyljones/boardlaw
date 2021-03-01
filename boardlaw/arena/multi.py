import time
import pandas as pd
import numpy as np
import torch
from rebar import arrdict, dotdict
from logging import getLogger

log = getLogger(__name__)

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

    def __init__(self, n_envs, n_envs_per, names, device='cuda', verbose=False):
        assert n_envs % n_envs_per == 0
        self.n_envs = n_envs
        self.n_envs_per = n_envs_per
        self.names = list(names)

        # Counts games that are either in-progress or that have been completed
        # self.games = torch.zeros((len(names), len(names)), dtype=torch.int, device=device)
        self.games = n_envs_per*torch.eye(len(names), dtype=torch.int, device=device)

        self.live = torch.full((n_envs, 2), -1, device=device)
        self.verbose = verbose

    def _live_counts(self):
        counts = torch.zeros_like(self.games)
        scatter_inc_(counts, self.live[(self.live > -1).all(-1)])
        return counts

    def update(self, terminal, mask):
        # Kill off the finished games
        masked = torch.zeros_like(mask)
        masked[mask] = terminal

        terminated = self.live[masked]

        self.live[masked] = -1
        if self.verbose:
            log.debug(f'Marked as terminated: {list(masked.cpu().int().numpy())}')

        return terminated

    def finished(self):
        return (self.games == self.n_envs_per).all() & (self.live == -1).all()

    def suggest(self, seats):
        # Figure out how the -1s in live should be repopulated

        while True:
            available = (self.live == -1).any(-1)
            remaining = (self.games < self.n_envs_per)
            if not (available.any() and remaining.any()):
                break

            counts = self._live_counts() + 1
            counts = counts.sum(0, keepdim=True) * counts.sum(1, keepdim=True) 
            counts = counts + counts.T
            goodness = (2*remaining.float() - 1)*counts

            choice = goodness.argmax()
            choice = (choice // len(self.names), choice % len(self.names))

            residual = self.n_envs_per - self.games[choice]
            allocation = available.nonzero(as_tuple=False)[:residual]
            self.live[allocation] = torch.as_tensor(choice, device=allocation.device)

            self.games[choice] += len(allocation)
            if self.verbose:
                log.debug(f'Matching {list(map(int, choice))} on envs {list(allocation.flatten().int().cpu().numpy())}')

        # Suggest the most 'popular' agent  
        active = self.live.gather(1, seats.long()[:, None]).squeeze(1)
        live_active = active[active > -1]
        totals = torch.zeros_like(self.games[0])
        totals.scatter_add_(0, live_active, totals.new_ones(live_active.shape))

        suggestion = totals.argmax()
        mask = (active == suggestion)
        name = self.names[int(suggestion)]

        if self.verbose:
            log.debug(f'Suggesting agent {name} with mask {list(mask.int().cpu().numpy())}')

        return name, mask, self.live.clone()

class MultiEvaluator:
    # Idea: keep lots and lots of envs in memory at once, play 
    # every agent against every agent simultaneously
    
    def __init__(self, worlds, agents, n_envs_per=1024):
        self.worlds = worlds
        self.agents = agents
        self.tracker = Tracker(worlds.n_envs, n_envs_per, list(agents))

        self.wins = torch.zeros((worlds.n_envs, worlds.n_seats), dtype=torch.int, device=worlds.device)
        self.moves = torch.zeros((worlds.n_envs,), dtype=torch.int, device=worlds.device)
        self.times = torch.zeros((worlds.n_envs,), dtype=torch.float, device=worlds.device)

    def record(self, mask, transitions, live, start, end):
        self.wins[mask] += (transitions.rewards == 1).int()
        self.moves[mask] += 1
        self.times[mask] += (end - start)/mask.sum()

        masked = torch.zeros_like(mask)
        masked[mask] = transitions.terminal

        # results = []
        # for idx in masked.nonzero(as_tuple=False).squeeze(-1):
        #     names = tuple(self.tracker.names[l] for l in live[idx])
        #     results.append(dotdict.dotdict(
        #                 names=names,
        #                 wins=tuple(map(float, self.wins[idx])),
        #                 moves=float(self.moves[idx].sum()),
        #                 games=float(self.wins[idx].sum()),
        #                 times=float(self.times[idx].sum()),
        #                 boardsize=self.worlds.boardsize))

        self.wins[masked] = 0.
        self.moves[masked] = 0.
        self.times[masked] = 0.

        return masked.sum()

    def step(self):
        name, mask, live = self.tracker.suggest(self.worlds.seats)
        
        start = time.time()
        decisions = self.agents[name](self.worlds[mask])
        self.worlds[mask], transitions = self.worlds[mask].step(decisions.actions)
        end = time.time()

        self.tracker.update(transitions.terminal, mask)

        results = self.record(mask, transitions, live, start, end)
        return results

class MockAgent:

    def __init__(self, id):
        self.id = id

    def __call__(self, world):
        id = torch.full((world.n_envs,), self.id, device=world.device, dtype=torch.long)
        return arrdict.arrdict(actions=id)

class MockGame(arrdict.namedarrtuple(fields=('count', 'history'))):

    @classmethod
    def initial(cls, n_envs=1, length=4, device='cuda'):
        return cls(
            history=torch.full((n_envs, length), -1, dtype=torch.long, device=device),
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
        history = self.history.clone()
        history.scatter_(1, self.count[:, None], actions[:, None])

        count = self.count + 1
        terminal = (count == self.history.shape[1])
        transition = arrdict.arrdict(terminal=terminal)

        count[terminal] = 0
        
        world = type(self)(count=count, history=history)

        return world, transition, list(history[terminal])

def test_tracker():
    n_envs = 32
    n_envs_per = 4
    length = 8

    worlds = MockGame.initial(n_envs, device='cpu', length=length)
    agents = {i: MockAgent(i) for i in range(16)}

    tracker = Tracker(n_envs, n_envs_per, agents, worlds.device)

    hists = []
    while not tracker.finished():
        name, mask, _ = tracker.suggest(worlds.seats)
        
        decisions = agents[name](worlds)
        worlds[mask], transitions, hist = worlds[mask].step(decisions.actions)
        hists.extend(hist)

        tracker.update(transitions.terminal, mask)
    hists = torch.stack(hists).cpu().numpy()

    from collections import defaultdict
    counts = defaultdict(lambda: 0)
    for h in hists:
        assert len(set(h)) <= 2
        counts[tuple(h[:2])] += 1

    assert len(counts) == len(agents)*(len(agents)-1)
    assert set(counts.values()) == {n_envs_per}

def test_evaluator():
    from pavlov import runs
    from boardlaw.arena import common

    df = runs.pandas(description='cat/nodes')
    worlds = common.worlds(df.index[0], 128*1024, device='cuda')
    agents = {r: common.agent(r, 1, worlds.device) for r in df.index}

    evaluator = MultiEvaluator(worlds, agents, 1024)

    import time 
    from IPython import display

    games = 0
    start = time.time()
    for i in range(1000):
        results = evaluator.step()
        for r in results:
            games += 1
        end = time.time()
        display.clear_output(wait=True)
        print(f'{i}: {moves/(end - start)/60:.0f}')
