import time
import pandas as pd
import numpy as np
import torch
from rebar import arrdict, dotdict
from logging import getLogger

log = getLogger(__name__)

def scatter_add_(totals, indices, vals=None):
    assert indices.ndim == 2
    rows, cols = indices.T

    width = totals.shape[1]
    raveled = rows + width*cols

    if vals is None:
        vals = totals.new_ones((len(rows),))
    if isinstance(vals, (int, float)):
        vals = totals.new_full((len(rows),), vals)

    totals.view(-1).scatter_add_(0, raveled, vals)

class Tracker:

    def __init__(self, n_envs, n_envs_per, names, max_simultaneous=32*1024, device='cuda', verbose=False):
        assert n_envs % n_envs_per == 0
        self.n_envs = n_envs
        self.n_envs_per = n_envs_per
        self.max_simultaneous = max_simultaneous
        self.names = np.random.permutation(list(names))

        # Counts games that are either in-progress or that have been completed
        # self.games = torch.zeros((len(names), len(names)), dtype=torch.int, device=device)
        self.games = n_envs_per*torch.eye(len(names), dtype=torch.int, device=device)

        self.live = torch.full((n_envs, 2), -1, device=device)
        self.verbose = verbose

    def progress(self):
        diag = len(self.names)*self.n_envs_per
        done = self.games.sum() - diag
        total = self.games.nelement()*self.n_envs_per - diag
        return done/total

    def _live_counts(self):
        counts = torch.zeros_like(self.games)
        scatter_add_(counts, self.live[(self.live > -1).all(-1)])
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
        mask = mask & (mask.cumsum(0) < self.max_simultaneous)
        name = self.names[int(suggestion)]

        if self.verbose:
            log.debug(f'Suggesting agent {name} with mask {list(mask.int().cpu().numpy())}')

        return name, mask, self.live[mask]

class MultiEvaluator:
    # Idea: keep lots and lots of envs in memory at once, play 
    # every agent against every agent simultaneously
    
    def __init__(self, worlds, agents, n_envs_per=1024):
        self.worlds = worlds
        self.agents = agents
        self.tracker = Tracker(worlds.n_envs, n_envs_per, list(agents))

        n_agents = len(agents)
        self.stats = arrdict.arrdict(
            indices=torch.stack(torch.meshgrid(torch.arange(n_agents), torch.arange(n_agents)), -1),
            wins=torch.zeros((n_agents, n_agents, worlds.n_seats), dtype=torch.int),
            moves=torch.zeros((n_agents, n_agents,), dtype=torch.int),
            times=torch.zeros((n_agents, n_agents), dtype=torch.float)).to(worlds.device)

    def finished(self):
        return self.tracker.finished()

    def record(self, transitions, live, start, end):
        #TODO: Figure out how to get scatter_add_ to work on vector-valued vals
        wins = (transitions.rewards == 1).int()
        scatter_add_(self.stats.wins[:, :, 0], live, wins[:, 0])
        scatter_add_(self.stats.wins[:, :, 1], live, wins[:, 1])
        scatter_add_(self.stats.moves, live, 1) 
        scatter_add_(self.stats.times, live, (end - start)/transitions.terminal.size(0))

        done = self.stats.wins.sum(-1) == self.tracker.n_envs_per
        stats = self.stats[done].cpu()
        results = []
        for idx in range(stats.indices.size(0)):
            item = stats[idx]
            names = tuple(self.tracker.names[i] for i in item.indices)
            results.append(dotdict.dotdict(
                        names=names,
                        wins=tuple(map(float, item.wins)),
                        moves=float(item.moves),
                        games=float(sum(item.wins)),
                        times=float(item.times),
                        boardsize=self.worlds.boardsize))
        
        self.stats.wins[done] = -1
        self.stats.moves[done] = -1
        self.stats.times[done] = -1

        return results

    def step(self):
        name, mask, live = self.tracker.suggest(self.worlds.seats)
        
        start = time.time()
        decisions = self.agents[name](self.worlds[mask])
        self.worlds[mask], transitions = self.worlds[mask].step(decisions.actions)
        end = time.time()

        self.tracker.update(transitions.terminal, mask)

        results = self.record(transitions, live, start, end)
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

    start = time.time()
    results = []
    games = 0
    while not evaluator.finished():
        rs = evaluator.step()
        end = time.time()
        
        results.extend(rs)
        games += sum(sum(r.wins) for r in rs)
        
        display.clear_output(wait=True)
        print(f'{games/(end - start)/60:.0f}')