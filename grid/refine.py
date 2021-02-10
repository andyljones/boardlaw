import torch
from pavlov import files, storage, runs
from boardlaw.mcts import MCTSAgent 
from boardlaw.hex import Hex
from boardlaw import arena

def assemble_agent(run, idx):
    network = storage.load_raw(run, 'model')
    agent = MCTSAgent(network)

    if idx is None:
        sd = storage.load_latest(run)
    else:
        sd = storage.load_snapshot(run, idx)
    agent.load_state_dict(sd['agent'])

    return agent

class Trialer:

    def __init__(self, worldfunc, max_history=128):
        self.worlds = worldfunc(4)
        self.mohex = mohex.MoHexAgent()
        self.history = deque(maxlen=max_history//self.worlds.n_envs)

    def trial(self, agent, record=True):
        size = self.worlds.boardsize
        games = database.symmetric_games(f'mohex-{size}').pipe(append, 'agent')
        wins = database.symmetric_wins(f'mohex-{size}').pipe(append, 'agent')
        for result in self.history:
            games.loc[result.names[0], result.names[1]] += result.games
            games.loc[result.names[1], result.names[0]] += result.games
            wins.loc[result.names[0], result.names[1]] += result.wins[0]
            wins.loc[result.names[1], result.names[0]] += result.wins[1]

        soln = activelo.solve(games, wins)
        μ, σ = analysis.difference(soln, 'mohex-0.00', 'agent')
        log.info(f'Agent elo is {μ:.2f}±{2*σ:.2f} based on {2*int(games.loc["agent"].sum())} games')
        if record:
            stats.mean_std('elo-mohex', μ, σ)

        imp = activelo.improvement(soln)
        imp = pd.DataFrame(imp, games.index, games.index)

        challenger = imp['agent'].idxmax()
        randomness = float(challenger.split('-')[1])
        self.mohex.random = randomness
        results = evaluator.evaluate(self.worlds, {'agent': agent, challenger: self.mohex})
        log.info(f'Agent played {challenger}, {int(results[0].wins[0] + results[1].wins[1])}-{int(results[0].wins[1] + results[1].wins[0])}')
        self.history.extend(results)

def evaluate(run, idx):
    agent = assemble_agent(run, idx)
    boardsize = agent.network.obs_space.dims[-1]
    worlds = Hex.initial(2, boardsize)

    games = arena.database.symmetric_games(f'mohex-{boardsize}').pipe(arena.mohex.append, 'agent')
    wins = arena.database.symmetric_wins(f'mohex-{boardsize}').pipe(arena.mohex.append, 'agent')
