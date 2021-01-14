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
    new = agentfunc().evaluator
    new.load_state_dict(state_dict)
    return new

class LeagueEvaluator(nn.Module):

    def __init__(self, slices, opponents):
        super().__init__()

        self.evaluator = None

        self.slices = slices
        self.opponents = nn.ModuleList(opponents)

        self.streams = [torch.cuda.Stream() for _ in range(2)]

    @profiling.nvtx
    def forward(self, worlds):
        split = min([s.start for s in self.slices], default=worlds.n_envs)

        parts = []

        torch.cuda.synchronize()
        with torch.cuda.stream(self.streams[0]):
            s = slice(0, split)
            parts.append(self.evaluator(worlds[s]))

        if split < worlds.n_envs:
            chunk = (worlds.n_envs - split)//len(self.opponents)
            assert split + chunk*len(self.opponents) == worlds.n_envs
            with torch.cuda.stream(self.streams[1]):
                for s, opponent in zip(self.slices, self.opponents): 
                    parts.append(opponent(worlds[s]))

        torch.cuda.synchronize()
        return arrdict.from_dicts(arrdict.cat(parts))

    def state_dict(self):
        return self.evaluator.state_dict()

    def load_state_dict(self, sd):
        self.evaluator.load_state_dict(sd)


class League:

    def __init__(self, agentfunc, n_envs, n_opponents=4, n_stabled=16, prime_frac=3/4, stable_interval=600, device='cuda'):
        self.n_envs = n_envs
        self.n_opponents = n_opponents
        self.n_stabled = n_stabled
        self.stable_interval = stable_interval
        self.device = device

        self.n_games = torch.zeros((n_opponents,), device=device)

        self.step = 0
        self.stable = {}

        prime_frac = 3/4
        if self.n_opponents:
            self.n_prime_envs = int(prime_frac*self.n_envs)
            self.n_oppo_envs = int((1 - prime_frac)*self.n_envs//self.n_opponents)
        else:
            self.n_prime_envs = self.n_envs
            self.n_oppo_envs = 0
        assert self.n_prime_envs + self.n_oppo_envs*self.n_opponents == self.n_envs

        self.stable = {i: agentfunc().evaluator.state_dict() for i in range(self.n_stabled)}

        idxs = np.random.choice(list(self.stable), (self.n_opponents,))
        start, chunk = self.n_prime_envs, self.n_oppo_envs
        self.league_eval = LeagueEvaluator(
            [slice(start + i*chunk, start + (i+1)*chunk) for i, idx in enumerate(idxs)],
            [agentfunc().evaluator for idx in idxs])

        self._update_mask(False)

    def _is_league(self, agent):
        return agent.evaluator == self.league_eval

    def _update_mask(self, is_league):
        is_prime = torch.full((self.n_envs,), True, device=self.device)
        if is_league:
            for s in self.league_eval.slices:
                is_prime[s] = False
        self.is_prime = is_prime

    def _update_stats(self, transition):
        # Update the games count
        for i, s in enumerate(self.league_eval.slices):
            self.n_games[i] += transition.terminal[s].sum()
    
    def _update_opponents(self):
        # Figure out who's been playing too long. Stagger it a bit so they don't all change at once
        threshold = torch.linspace(1.5, 2.5, self.n_opponents, device=self.device).mul(self.n_oppo_envs).int()
        (replace,) = (self.n_games >= threshold).nonzero(as_tuple=True)
        # Swap out any over the limit
        for i in replace:
            new = np.random.choice(list(self.stable))
            self.league_eval.opponents[i].load_state_dict(self.stable[new])
            log.info(f'New opponent #{new} is {self.step - new} steps old')

            self.n_games[i] = 0

    def _update_stable(self, evaluator):
        if self.step % self.stable_interval == 0:
            old = np.random.choice(list(self.stable))
            del self.stable[old]
            self.stable[self.step] = clone(evaluator.state_dict())
            log.info(f'Network #{self.step} stabled; #{old} removed')

    def update(self, agent, transition):
        # Toggle whether the network is running only the prime copy, or many.
        if self._is_league(agent):
            agent.evaluator = self.league_eval.evaluator
        else:
            self.league_eval.evaluator = agent.evaluator
            agent.evaluator = self.league_eval

        self._update_mask(self._is_league(agent))
        self._update_stats(transition)
        self._update_opponents()
        self._update_stable(agent.evaluator)

        self.step += 1

class MockWorlds(arrdict.arrdict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_envs = len(self.dummy)

class MockEvaluator(nn.Module):

    def __init__(self, skill):
        super().__init__()
        self.skill = skill

    def forward(self, worlds):
        return torch.full_like(worlds.dummy, self.skill)

    def state_dict(self):
        return self.skill

    def load_state_dict(self, sd):
        self.skill = sd

class MockAgent:

    def __init__(self, skill=0):
        self.evaluator = MockEvaluator(skill)

    def __call__(self, worlds):
        return self.evaluator(worlds)

def demo():
    worlds = MockWorlds(dummy=torch.zeros((128,)))

    league = League(MockAgent, worlds.n_envs, stable_interval=8)

    agent = MockAgent(0)
    evaluator = agent.evaluator

    results = []
    for _ in range(400):
        skills = agent(worlds)
        transitions = arrdict.arrdict(
            terminal=torch.rand_like(skills) < .25)
        league.update(agent, transitions)

        evaluator.skill += 1

        results.append(skills)
    results = torch.stack(results)

    import matplotlib.pyplot as plt
    for i in range(88, 128, 8):
        y = results[1::2, i].cpu()
        x = np.arange(len(y))
        plt.scatter(x, y, marker='.')
