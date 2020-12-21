import matplotlib.pyplot as plt
import numpy as np
from rebar import arrdict
import torch
import torch.distributions
import activelo

def winrate(black, white):
    return 1/(1 + np.exp(-(black - white)))

def residual_vs_mean(Σ):
    return np.diag(Σ - np.outer(Σ.mean(0), Σ.mean(0))/Σ.mean())

def resid_var(ranks, truth):
    return (((truth - truth.mean()) - (ranks - ranks.mean()))**2).sum()/((truth - truth.mean())**2).sum()

def status(soln):
    return f'{soln.σresid:.2f}σd, {soln.resid_var:.0%} resid var'

def plot(trace, truth=None, t=-1):
    soln = trace[t]
    N = len(soln.μ)

    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(18, 6)

    ax = axes[0]
    μd = soln.μ - soln.μ.mean()
    σd = residual_vs_mean(soln.Σ)**.5
    ax.errorbar(np.arange(N), μd, yerr=σd, marker='.', linestyle='', capsize=2)
    if truth is not None:
        ax.plot(truth - truth.mean())
    ax.set_title(f'μ±σ vs. mean agent')
    ax.set_xlabel(status(soln))

    ax = axes[1]
    eigs = np.linalg.eigvalsh(soln.Σ)
    ax.plot(eigs)
    ax.set_yscale('log')
    ax.set_title(f'Σ spectrum')
    ax.set_xlabel(f'condition: {eigs.max()/eigs.min():.0G}')

    ax = axes[2]
    T = len(trace) if isinstance(trace, list) else len(trace.μ)
    ax.imshow(soln.n, cmap='Greens')
    ax.set_title('games played')
    ax.set_xlabel(f'{T} rounds, {int(soln.n.sum()/(2*T))} games per')

    return fig

def simulate(truth, n_games=256, σresid_tol=.1):
    n_agents = len(truth)
    wins = torch.zeros((n_agents, n_agents))
    games = torch.zeros((n_agents, n_agents))

    trace = []
    solver = activelo.Solver(n_agents)
    ranks = torch.full((n_agents,), 0.)
    while True:
        soln = solver(games, wins)
        ranks = torch.as_tensor(soln.μ)

        black, white = activelo.suggest(soln)
        black_wins = torch.distributions.Binomial(n_games, winrate(truth[black], truth[white])).sample()
        wins[black, white] += black_wins
        wins[white, black] += n_games - black_wins
        games[black, white] += n_games
        games[white, black] += n_games

        soln['n'] = games.clone()
        soln['w'] = wins.clone()
        soln['σresid'] = residual_vs_mean(soln.Σ).mean()**.5
        soln['resid_var'] = resid_var(ranks, truth)
        trace.append(arrdict.arrdict({k: v for k, v in soln.items() if k != 'trace'}))

        plt.close()
        from IPython import display 
        display.clear_output(wait=True)
        display.display(plot(trace, truth))
        if soln.σresid < σresid_tol:
            break
        
    trace = arrdict.stack(trace)

    return trace

def linear_ranks(n_agents=10):
    return torch.linspace(1, 5, n_agents).float()

def log_ranks(n_agents=10):
    return torch.linspace(1, 50, n_agents).float().log()

def pow_ranks(n_agents=10, pow=.5):
    return torch.linspace(1, 50, n_agents).float().pow(pow)

def simulate_log_ranks():
    truth = log_ranks(10)
    trace = simulate(truth)
    plot(trace, truth)

def random_ranks(n_agents=10):
    deltas = torch.randn((n_agents,))/n_agents**.5
    totals = deltas.cumsum(0) 
    totals = totals - totals.min()
    return torch.sort(totals).values

def simulate_random_ranks():
    counts = []
    for _ in range(100):
        ranks = random_ranks(n_agents=10)
        trace = simulate(ranks)
        counts.append(len(trace.n))
    q = np.quantile(counts, .95)
    return q