import matplotlib.pyplot as plt
import numpy as np
import torch
from logging import getLogger
from torch import nn
import torch.cuda
from torch.nn import functional as F
from rebar import dotdict, arrdict, profiling

log = getLogger(__name__)

def clone(x):
    if isinstance(x, dict):
        return {k: clone(v) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.clone().detach()
    else:
        return x

def assemble(agentfunc, state_dict):
    new = agentfunc().network
    new.load_state_dict(state_dict)
    return new

class Splitter(nn.Module):

    def __init__(self, names, slices, field):
        super().__init__()

        self.network = None

        self.names = names
        self.slices = slices
        self.field = nn.ModuleList(field)

        self.streams = [torch.cuda.Stream() for _ in range(2)]

    @profiling.nvtx
    def forward(self, worlds):
        split = min([s.start for s in self.slices], default=worlds.n_envs)

        parts = []

        torch.cuda.synchronize()
        with torch.cuda.stream(self.streams[0]):
            s = slice(0, split)
            parts.append(self.network(worlds[s]))

        if split < worlds.n_envs:
            chunk = (worlds.n_envs - split)//len(self.field)
            assert split + chunk*len(self.field) == worlds.n_envs
            with torch.cuda.stream(self.streams[1]):
                for s, opponent in zip(self.slices, self.field): 
                    parts.append(opponent(worlds[s]))

        torch.cuda.synchronize()
        return arrdict.from_dicts(arrdict.cat(parts))

    def state_dict(self):
        return self.network.state_dict()

    def load_state_dict(self, sd):
        self.network.load_state_dict(sd)

def splitter(stable, agentfunc, n_envs, n_fielded, prime_frac):

    if n_fielded:
        n_prime_envs = int(prime_frac*n_envs)
        n_oppo_envs = int((1 - prime_frac)*n_envs//n_fielded)
    else:
        n_prime_envs = n_envs
        n_oppo_envs = 0
    assert n_prime_envs + n_oppo_envs*n_fielded == n_envs

    names, slices, field = [], [], []
    for i in range(n_fielded):
        name, sd = stable.draw()
        names.append(name)

        start = n_prime_envs + i*n_oppo_envs
        end = n_prime_envs + (i+1)*n_oppo_envs
        slices.append(slice(start, end))

        network = agentfunc().network
        network.load_state_dict(sd)
        field.append(network)

    return Splitter(names, slices, field)

class Stable:

    def __init__(self, default, n_stabled, stable_interval, verbose=True):
        #TODO: This is a bit lazy; doesn't deal with games that have multiple networks playing in them,
        # and doesn't deal with networks that get evicted from the stable while they're still on the field.
        self.n_stabled = n_stabled
        self.stable_interval = stable_interval
        self.verbose = verbose

        self.names = [-1]
        self.stable = [clone(default.state_dict())]

        self.losses = np.full((n_stabled,), 1)
        self.wins = np.full((n_stabled,), 1)

        self.step = 0

    def update_stats(self, splitter, league_seat, transition):
        rewards = transition.rewards.gather(1, league_seat[:, None].long()).squeeze(-1).cpu().numpy()
        for n, s in zip(splitter.names, splitter.slices):
            if n in self.names:
                i = self.names.index(n)
                self.wins[i] += (rewards[s] == 1).sum()
                self.losses[i] += (rewards[s] == -1).sum()

    def update_stable(self, network):
        rates = (self.wins/(self.wins + self.losses))[:len(self.stable)]

        if self.step % self.stable_interval == 0:
            if len(self.stable) == self.n_stabled:
                old = rates.argmin()
                self.log(f'Network #{self.step} stabled; #{self.names[old]} removed')
                self.names[old] = self.step
                self.stable[old] = clone(network.state_dict())
                self.losses[old] = 0
                self.wins[old] = 1
            else:
                self.names.append(self.step)
                self.stable.append(clone(network.state_dict()))
                self.log(f'Network #{self.step} stabled')

        self.step += 1

    def draw(self):
        rates = (self.wins/(self.wins + self.losses))[:len(self.stable)]
        dist = np.exp(rates)/np.exp(rates).sum()

        i = np.random.choice(np.arange(len(dist)), p=dist)
        return self.names[i], self.stable[i]

    def log(self, m):
        if self.verbose:
            log.info(m)

class Field:

    def __init__(self, n_fielded, verbose=True):
        self.n_fielded = n_fielded
        self.games = np.zeros((n_fielded,))
        self.verbose = verbose

    def update_stats(self, splitter, transition):
        transition = transition.detach().cpu().numpy()
        for i, s in enumerate(splitter.slices):
            self.games[i] += transition.terminal[s].sum()

    def update_field(self, splitter, stable):
        # Figure out who's been playing too long. Stagger it a bit so they don't all change at once
        threshold = np.linspace(1.5, 2.5, self.n_fielded)
        for i, (n, s) in enumerate(zip(splitter.names, splitter.slices)):
            replace = self.games[i] >= threshold[i]*(s.stop - s.start)
        # Swap out any over the limit
            if replace:
                name, sd = stable.draw()
                splitter.field[i].load_state_dict(sd)
                splitter.names[i] = name

                if self.verbose:
                    log.info(f'New opponent is #{name}')

                self.games[i] = 0

class League:

    def __init__(self, agentfunc, n_envs, 
            n_fielded=4, n_stabled=16, prime_frac=3/4, 
            stable_interval=100, device='cuda', verbose=True):

        self.n_envs = n_envs
        self.n_opponents = n_fielded
        self.n_stabled = n_stabled
        self.stable_interval = stable_interval
        self.device = device

        self.stable = Stable(agentfunc().network, n_stabled, stable_interval, verbose=verbose)
        self.field = Field(n_fielded)
        self.splitter = splitter(self.stable, agentfunc, n_envs, n_fielded, prime_frac, verbose=verbose)

        self.update_mask(False)

    def is_league(self, agent):
        return agent.network == self.splitter

    def update_mask(self, is_league):
        is_prime = torch.full((self.n_envs,), True, device=self.device)
        if is_league:
            for s in self.splitter.slices:
                is_prime[s] = False
        self.is_prime = is_prime

    def update(self, agent, seats, transition):
        # Toggle whether the network is running only the prime copy, or many.
        league_stepped = self.is_league(agent)
        if league_stepped:
            league_seat = seats
            agent.network = self.splitter.network
        else:
            league_seat = 1 - seats
            self.splitter.network = agent.network
            agent.network = self.splitter

        self.stable.update_stats(self.splitter, league_seat, transition)
        self.field.update_stats(self.splitter, transition)
        self.field.update_field(self.splitter, self.stable)
        self.stable.update_stable(agent.network)

        self.update_mask(self.is_league(agent))

### TESTS ###

class MockWorlds(arrdict.namedarrtuple(fields=('seats', 'cumulator', 'lengths'))):
    """One-step one-seat win (+1)"""

    @classmethod
    def initial(cls, n_envs=1, device='cpu'):
        return cls(
            seats=torch.full((n_envs,), 0, device=device),
            lengths=torch.full((n_envs,), 1, device=device),
            cumulator=torch.full((n_envs, 2), 0., device=device))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.seats, torch.Tensor):
            return 

        self.n_envs = len(self.seats)
        self.device = self.seats.device
        self.n_seats = 2

    def step(self, skills):
        terminal = self.lengths == 4
        new_lengths = self.lengths + 1
        new_lengths[terminal] = 1

        new_cumulator = self.cumulator.scatter_add(1, self.seats[:, None], skills[:, None].float())

        rates = torch.sigmoid((new_cumulator[:, 0] - new_cumulator[:, 1])/4)
        rewards = terminal * (2*(torch.rand((self.n_envs,),) <= rates) - 1)
        rewards = torch.stack([rewards, -rewards], -1)

        new_cumulator[terminal] = 0

        new_seats = 1 - self.seats
        new_seats[terminal] = 0

        trans = arrdict.arrdict(terminal=terminal, rewards=rewards)
        return type(self)(seats=new_seats, lengths=new_lengths, cumulator=new_cumulator), trans

class MockNetwork(nn.Module):

    def __init__(self, skill):
        super().__init__()
        self.skill = skill

    def forward(self, worlds):
        return torch.full_like(worlds.seats, self.skill)

    def state_dict(self):
        return self.skill

    def load_state_dict(self, sd):
        self.skill = sd

class MockAgent:

    def __init__(self, skill=0):
        self.network = MockNetwork(skill)

    def __call__(self, worlds):
        return self.network(worlds)

def test_world():
    worlds = MockWorlds.initial(128)
    fst = MockAgent(0)
    snd = MockAgent(10)

    rewards = []
    for _ in range(8):
        for agent in (fst, snd):
            skills = agent(worlds)
            worlds, transitions = worlds.step(skills)
            rewards.append(transitions.rewards)
    wins = (torch.stack(rewards) == 1).sum(0).sum(0)
    assert wins[0] < wins[1]

    worlds = MockWorlds.initial(128)
    fst = MockAgent(10)
    snd = MockAgent(0)

    rewards = []
    for _ in range(8):
        for agent in (fst, snd):
            skills = agent(worlds)
            worlds, transitions = worlds.step(skills)
            rewards.append(transitions.rewards)
    wins = (torch.stack(rewards) == 1).sum(0).sum(0)
    assert wins[0] > wins[1]

def test_stable():
    agent = MockAgent(0)
    stable = Stable(agent, 2, 1, verbose=False)

    splitter = Splitter([-1], [slice(0, 1)], [agent])

    stable.update_stats()

    pass


def demo():
    T = 1000
    n_env = 128
    n_stabled = 8
    worlds = MockWorlds.initial(n_env)

    league = League(MockAgent, worlds.n_envs, n_stabled=n_stabled, stable_interval=8, verbose=False)

    agent = MockAgent(0)
    network = agent.network

    fielded = np.zeros((T, T))
    stabled = np.zeros((T, T))
    for t in range(15):
        skills = agent(worlds)
        worlds, transitions = worlds.step(skills)
        league.update(agent, worlds.seats, transitions)

        network.skill = 10. if t < 10 else 1.

        for n, s in zip(league.splitter.names, league.splitter.slices):
            fielded[t, n] = s.stop - s.start
        
        for n in league.stable.names:
            stabled[t, n] = 1.

    parts = dotdict.dotdict(
        agent=agent,
        league=league,
        worlds=worlds)
    trace = dotdict.dotdict(
        fielded=fielded,
        stabled=stabled)

    return parts, trace

def plot_trace(trace):
    fig, (l, r) = plt.subplots(1, 2)

    l.imshow(trace.stabled)
    l.set_title('stabled')

    r.imshow(trace.fielded)
    r.set_title('fielded')

    fig.set_size_inches(18, 9)


