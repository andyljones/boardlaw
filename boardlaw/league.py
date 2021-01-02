"""
* We want to inject decisions made by old agents into experience collection
* We want to inject those decisions a few steps at a time
* We want to keep a stable of agents that can best beat eachother and can 
best beat the current agent.
* Judging 

"""
class League:

    def __init__(self, n_agents=8):
        self.steps = 0
        self.stable = {}

    def submit(self, latest):
        pass

    def select(self, latest):
        pass

    def update(self, rewards, terminal):
        pass