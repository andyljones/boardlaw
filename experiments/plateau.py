from boardlaw import mohex, analysis
from boardlaw.main import worldfunc, agentfunc
from pavlov import storage
import torch
from rebar import arrdict

@torch.no_grad()
def rollout(worlds, agents):
    from IPython import display

    wins = torch.zeros((worlds.n_seats,), device=worlds.device)
    games = torch.zeros((), device=worlds.device)
    while True:
        decisions, masks = {}, {}
        for i, agent in enumerate(agents):
            mask = worlds.seats == i
            if mask.any():
                decisions[i] = agent(worlds[mask])
                masks[i] = mask

        actions = torch.cat([d.actions for d in decisions.values()])
        for mask, decision in zip(masks.values(), decisions.values()):
            actions[mask] = decision.actions
        
        worlds, transitions = worlds.step(actions)
        wins += (transitions.rewards == 1).sum(0)
        games += transitions.terminal.sum(0)

        rate = wins[0]/games
        p = (wins[0] + 1)/(games + 2)
        var = p*(1 - p)/games
        display.clear_output(wait=True)
        print(f'{rate:.2f}±{var**.5:.2f}')

def run():
    n_envs = 8
    world = worldfunc(n_envs)
    agent = agentfunc()
    agent.evaluator = agent.evaluator
    agent.kwargs['n_nodes'] = 512

    sd = storage.load_snapshot('*perky-boxes*', 64)
    agent.load_state_dict(sd['agent'])

    mhx = mohex.MoHexAgent()
    trace = rollout(world, [agent, mhx])