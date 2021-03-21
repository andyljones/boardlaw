import numpy as np
from gym import spaces
import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
import inspect

OBS = {}
ACT = {}

class Kwargs(spaces.Dict):

    def __init__(self, **kwargs):
        super().__init__(spaces=OrderedDict(kwargs))

    def __repr__(self):
        return "Kwargs(" + ", ". join([str(k) + ":" + str(s) for k, s in self.spaces.items()]) + ")"

class Registry:

    def __init__(self):
        self.modules = {}

    def key(self, space):
        if isinstance(space, str):
            return space
        elif isinstance(space, spaces.Space):
            return type(space).__name__
        elif issubclass(space, spaces.Space):
            return space.__name__
        else:
            raise ValueError(f'Can\'t generate a registry key for "{space}"')

    def register(self, space):
        def wrapper(cls):
            self.modules[self.key(space)] = cls
            return cls
        return wrapper
    
    def __getitem__(self, space):
        return self.modules[self.key(space)]

OBS = Registry()
ACT = Registry()

@OBS.register(spaces.Box)
class ObsBox(nn.Module):

    def __init__(self, space, width):
        super().__init__()

        self.space = space

        fanin = np.prod(space.shape)
        self.layer = nn.Linear(fanin, width) 

        self.register_buffer('low', torch.as_tensor(space.low).float())
        self.register_buffer('high', torch.as_tensor(space.high).float())

    def forward(self, obs):
        tail = self.space.shape
        head = obs.shape[:-len(tail)]

        mid = (self.low + self.high)/2.
        span = (self.high - self.low)/2.

        flattened = obs.flatten(len(head)) if head else obs
        scaled = (flattened - mid)/span

        ouput = self.layer(scaled)
        shaped = ouput.reshape(*head, -1)
        return F.relu(shaped)

@OBS.register(spaces.Dict)
class ObsDict(nn.Module):
    
    def __init__(self, space, width):
        super().__init__()

        intake = width*len(space.spaces)
        self.layer = nn.Linear(intake, width)

        self.submodules = nn.ModuleDict({k: observation_module(v, width) for k, v in space.spaces.items()})

    def forward(self, obs):
        substates = [self.submodules[k](obs[k]) for k in self.submodules]
        substates = torch.cat(substates, -1)
        output = self.layer(substates)
        return F.relu(output)

@OBS.register(Kwargs)
class ObsKwargs(ObsDict):
    
    def __init__(self, space, width):
        super().__init__(space, width)
        params = [inspect.Parameter(k, inspect.Parameter.POSITIONAL_OR_KEYWORD) for k in space.spaces]
        self.signature = inspect.Signature(params)

    def forward(self, *args, **kwargs):
        arguments = self.signature.bind(*args, **kwargs).arguments
        substates = [self.submodules[k](arguments[k]) for k in self.submodules]
        substates = torch.cat(substates, -1)
        output = self.layer(substates)
        return F.relu(output)


def observation_module(space, width):
    if isinstance(space, dict):
        space = spaces.Dict(space)
    return OBS[space](space, width)

@ACT.register(spaces.Box)
class ActBox(nn.Module):

    def __init__(self, space, width):
        super().__init__()

        self.space = space

        fanout = np.prod(space.shape)
        self.layer = nn.Linear(width, fanout) 

        self.register_buffer('low', torch.as_tensor(space.low).float())
        self.register_buffer('high', torch.as_tensor(space.high).float())

    def forward(self, state):
        output = self.layer(state)

        head = output.shape[:-1]
        tail = self.space.shape
        shaped = output.reshape(*head, *tail)
        output = torch.tanh(shaped)

        mid = (self.low + self.high)/2.
        span = (self.high - self.low)/2.
        return output*span + mid

@ACT.register('value')
class ActValue(nn.Module):

    def __init__(self, space, width):
        super().__init__()
        self.layer = nn.Linear(width, 1)

    def forward(self, state):
        return self.layer(state).squeeze(-1)

def action_module(space, width):
    return ACT[space](space, width)


class Network(nn.Module):

    def __init__(self, obs_space, action_space, width=2, depth=1):
        super().__init__()
        self.obs = observation_module(obs_space, width)
        self.act = action_module(action_space, width)
        self.core = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])

        self.register_buffer('_dummy', torch.as_tensor(np.nan))

    @property
    def device(self) -> torch.device:
        return self._dummy.device

    def forward(self, *args, **kwargs):
        state = self.obs(*args, **kwargs)
        for layer in self.core:
            # Leaky ReLU so that all outputs depend on all inputs. Makes checking that you've got things 
            # wired up easier, as you can just stuff a huge input in and see which outputs twitch
            state = F.leaky_relu(layer(state))
        return self.act(state)

def network(env):
    return Network(env.observation_space, env.action_space)