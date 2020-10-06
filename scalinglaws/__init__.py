from . import hex, agents

def train():
    buffer_size = 16
    n_envs = 4
    n_agents = 1
    batch_size = 8*1024

    env = hex.Hex(n_envs)

    agent = agents.Agent(env.obs_space, env.action_space).to(env.device)

    inputs = env.reset()
    for _ in range(120):
        decisions = agent(inputs[None], sample=True).squeeze(0)
        response, inputs = env.step(decisions.actions)
