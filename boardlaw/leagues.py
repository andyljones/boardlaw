"""
* Get a stream of agents
* Get a stream of wins/losses
* Emit which agent to play in each step
* Use emissions and wins/losses to figure out each agent's 'responsibility' 
  for winning/losing a game.
* Use these responsibilities to figure out pairwise winrates
  * Convolve responsibilities with a blurring kernel, so that similar agents
    can have their winrates refined
  * Will need to extend responsibilities tracker into the future, or have 
    new agents inherit some fraction of the old agents' winrates
* Use pairwise winrates and previous emission to figure out who to emit next
  * Want to emit agents that'll win against the latest agent
  * Want to emit agents that're dissimilar to recently emitted agents
  * Want to bias strongly towards emitting the latest agent
"""
import numpy as np
import torch
from logging import getLogger
from torch import nn
import torch.cuda
from torch.nn import functional as F
from rebar import arrdict, profiling

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

class LeagueNetwork(nn.Module):

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

def league_network(stable, agentfunc, n_envs, n_fielded, prime_frac):

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

    return LeagueNetwork(names, slices, field)

class Stable:

    def __init__(self, default, n_stabled, stable_interval):
        #TODO: This is a bit lazy; doesn't deal with games that have multiple networks playing in them,
        # and doesn't deal with networks that get evicted from the stable while they're still on the field.
        self.n_stabled = n_stabled

        self.stable_interval = stable_interval

        self.names = [-1]
        self.stable = [default]

        self.losses = np.full((n_stabled,), 1)
        self.wins = np.full((n_stabled,), 1)

        self.step = 0

    def update_stats(self, league_eval, league_seat, transition):
        rewards = transition.rewards.gather(1, league_seat[:, None].long()).squeeze(-1).cpu().numpy()
        for n, s in zip(league_eval.names, league_eval.slices):
            if n in self.names:
                i = self.names.index(n)
                self.wins[i] += (rewards[s] == 1).sum()
                self.losses[i] += (rewards[s] == -1).sum()

    def update_stable(self, network):
        if self.step % self.stable_interval == 0:
            if len(self.stable) == self.n_stabled:
                old = np.random.randint(len(self.names))
                self.names[old] = self.step
                self.stable[old] = clone(network.state_dict())
                self.losses[old] = 0
                self.wins[old] = 1
                log.info(f'Network #{self.step} stabled; #{old} removed')
            else:
                self.names.append(self.step)
                self.stable.append(clone(network.state_dict()))
                log.info(f'Network #{self.step} stabled')

        self.step += 1

    def draw(self):
        rates = (self.wins/(self.wins + self.losses))[:len(self.stable)]
        dist = np.exp(rates)/np.exp(rates).sum()

        i = np.random.choice(np.arange(len(dist)), p=dist)
        return self.names[i], self.stable[i]

class Field:

    def __init__(self, n_fielded):
        self.n_fielded = n_fielded
        self.games = np.zeros((n_fielded,))

    def update_stats(self, league_eval, transition):
        transition = transition.detach().cpu().numpy()
        for i, s in enumerate(league_eval.slices):
            self.games[i] += transition.terminal[s].sum()

    def update_field(self, league_eval, stable):
        # Figure out who's been playing too long. Stagger it a bit so they don't all change at once
        threshold = np.linspace(1.5, 2.5, self.n_fielded)
        for i, (n, s) in enumerate(zip(league_eval.names, league_eval.slices)):
            replace = self.games[i] >= threshold[i]*(s.stop - s.start)
        # Swap out any over the limit
            if replace:
                name, sd = stable.draw()
                league_eval.field[i].load_state_dict(sd)

                log.info(f'New opponent is #{name}')

                self.games[i] = 0

class League:

    def __init__(self, agentfunc, n_envs, 
            n_fielded=4, n_stabled=10, prime_frac=3/4, 
            stable_interval=100, device='cuda'):

        self.n_envs = n_envs
        self.n_opponents = n_fielded
        self.n_stabled = n_stabled
        self.stable_interval = stable_interval
        self.device = device

        self.stable = Stable(agentfunc().network.state_dict(), n_stabled, stable_interval)
        self.field = Field(n_fielded)
        self.league_eval = league_network(self.stable, agentfunc, n_envs, n_fielded, prime_frac)

        self.update_mask(False)

    def is_league(self, agent):
        return agent.network == self.league_eval

    def update_mask(self, is_league):
        is_prime = torch.full((self.n_envs,), True, device=self.device)
        if is_league:
            for s in self.league_eval.slices:
                is_prime[s] = False
        self.is_prime = is_prime

    def update(self, agent, seats, transition):
        # Toggle whether the network is running only the prime copy, or many.
        league_stepped = self.is_league(agent)
        if league_stepped:
            league_seat = seats
            agent.network = self.league_eval.network
        else:
            league_seat = 1 - seats
            self.league_eval.network = agent.network
            agent.network = self.league_eval

        self.stable.update_stats(self.league_eval, league_seat, transition)
        self.field.update_stats(self.league_eval, transition)
        self.field.update_field(self.league_eval, self.stable)
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
            cumulator=torch.full((n_envs,), 0., device=device))

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

        new_cumulator = self.cumulator - 2*(self.seats - .5)*skills

        rates = torch.sigmoid(new_cumulator/4)
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

def demo():
    worlds = MockWorlds.initial(128)

    league = League(MockAgent, worlds.n_envs, stable_interval=8)

    agent = MockAgent(0)
    network = agent.network

    results = []
    for t in range(1000):
        skills = agent(worlds)
        worlds, transitions = worlds.step(skills)
        league.update(agent, worlds.seats, transitions)

        network.skill = t**.5

        names = np.zeros(worlds.n_envs)
        for n, s in zip(league.league_eval.names, league.league_eval.slices):
            names[s] = n
        results.append(names)

    results = np.stack(results)

    return league, results

def plot_results(results):
    import matplotlib.pyplot as plt
    for i in range(88, 128, 8):
        y = results[1::2, i].cpu()
        x = np.arange(len(y))
        plt.scatter(x, y, marker='.')


