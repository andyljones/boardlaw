from . import heads, lstm
from torch import nn
from rebar import recurrence, arrdict
from torch.nn import functional as F

class Residual(nn.Linear):

    def __init__(self, width):
        super().__init__(width, width)

    def forward(self, x, **kwargs):
        return x + F.relu(super().forward(x))

class Agent(nn.Module):

    def __init__(self, obs_space, action_space, width=256):
        super().__init__()
        out = heads.output(action_space, width)
        self.sampler = out.sample
        self.policy = recurrence.Sequential(
            heads.intake(obs_space, width),
            Residual(width),
            Residual(width),
            Residual(width),
            Residual(width),
            # lstm.LSTM(width),
            out)
        self.value = recurrence.Sequential(
            heads.intake(obs_space, width),
            Residual(width),
            Residual(width),
            Residual(width),
            Residual(width),
            # lstm.LSTM(width),
            heads.ValueOutput(width))

    def forward(self, inputs, sample=False, value=False, test=False):
        kwargs = {k: v for k, v in inputs.items() if k != 'obs'}
        outputs = arrdict.arrdict(
            logits=self.policy(inputs.obs, **kwargs))

        if sample or test:
            outputs['actions'] = self.sampler(outputs.logits, test)
        if value:
            outputs['value'] = self.value(inputs.obs, **kwargs)
        return outputs

class MultiAgent(nn.ModuleList):

    def __init__(self, n_agents, *args, **kwargs):
        super().__init__([Agent(*args, **kwargs) for _ in range(n_agents)])

    def forward(self, x, **kwargs):
        return arrdict.stack([a(x[:, i], **kwargs) for i, a in enumerate(self)], 1)