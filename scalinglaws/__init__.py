from . import hex, agents

def train():
    buffer_size = 16
    n_envs = 8*1024
    n_agents = 1
    batch_size = 8*1024

    env = hex.Hex(n_envs)

    agent = agents.Agent(env.obs_space, env.action_space)