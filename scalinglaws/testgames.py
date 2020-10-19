"""
What to test?
    * Does it all run? Only need one action, one player, one timestep for this. 
    * Does it correctly pick the winning move? One player, two actions, instant returns. 
    * Does it correctly pick the winning move deep in the tree? One player, two actions, a bunch of ignored actions,
      and then a payoff depending on those first two
    * Does it correctly estimate state values for different players? Need two players with different returns.
    * 
"""
import torch
from rebar import arrdict

class RandomAgent:

    def __call__(self, inputs, value=True):
        return arrdict.arrdict(
            logits=torch.log(inputs.valid.float()/inputs.valid.sum(-1, keepdims=True)))

class KnownValueRandomAgent:

    def __call__(self, inputs, value=True):
        return arrdict.arrdict(
            logits=torch.log(inputs.valid.float()/inputs.valid.sum(-1, keepdims=True)),
            v=inputs.v)

class RandomRolloutAgent:

    def __init__(self, env, n_rollouts):
        self.env = env
        self.n_rollouts = n_rollouts

    def rollout(self, inputs):
        env = self.env
        original = env.state_dict()
        chooser = inputs.seat

        live = torch.ones_like(inputs.terminal)
        reward = torch.zeros_like(inputs.terminal, dtype=torch.float)
        while True:
            if not live.any():
                break

            actions = torch.distributions.Categorical(probs=inputs.mask.float()).sample()
            same_seat = (inputs.seat == chooser).float()

            inputs = env.step(actions)

            reward += inputs.reward * live.float() * same_seat
            live = live & ~inputs.terminal

        env.load_state_dict(original)

        return reward

    def __call__(self, inputs, value=True):
        B, A = inputs.valid.shape
        return arrdict.arrdict(
            logits=torch.log(inputs.valid.float()/inputs.valid.sum(-1, keepdims=True)),
            v=torch.stack([self.rollout(inputs) for _ in range(self.n_rollouts)]).mean(0))
    
class InstantReturn:

    def __init__(self, n_envs=1, device='cuda'):
        self.device = torch.device(device)
        self.n_envs = 1
        self.n_seats = 1

    def reset(self):
        return arrdict.arrdict(
            valid=torch.ones((self.n_envs, 1), dtype=torch.bool, device=self.device),
            seats=torch.ones((self.n_envs,), dtype=torch.int, device=self.device))

    def step(self, actions):
        arc = arrdict.arrdict(
            terminal=torch.ones((self.n_envs,), dtype=torch.bool, device=self.device),
            rewards=torch.ones((self.n_envs,), dtype=torch.float, device=self.device))
        
        node = arrdict.arrdict(
            valid=torch.ones((self.n_envs, 1), dtype=torch.bool, device=self.device),
            seats=torch.ones((self.n_envs,), dtype=torch.int, device=self.device))
        
        return arc, node