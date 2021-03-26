from analysis.paper import *
from boardlaw import elos, sql


def run():
    # Quick sanity check that adding a copy of an agent doesn't inflate Elos in any way
    boardsize = 9 
    trials = sql.query('''
            select trials.* 
            from trials 
                inner join agents_details as black
                    on (trials.black_agent == black.id)
                inner join agents_details as white
                    on (trials.white_agent == white.id)
            where 
                (black.boardsize == ?) and (white.boardsize == ?) and
                (black.test_nodes == 64) and (white.test_nodes == 64)''', index_col='id', params=(int(boardsize), int(boardsize)))
    ws, gs = elos.symmetrize(trials)

    # Set up a copy of each agent
    N = ws.shape[0]
    ws2 = np.full((2*N, 2*N), np.nan)
    ws2[:N, :N] = ws
    ws2[-N:, -N:] = ws
    ws2[:N, -N:][np.diag_indices_from(ws)] = 256
    ws2[-N:, :N][np.diag_indices_from(ws)] = 256
    ws2 = pd.DataFrame(ws2)

    gs2 = np.full((2*N, 2*N), np.nan)
    gs2[:N, :N] = gs
    gs2[-N:, -N:] = gs
    gs2[:N, -N:][np.diag_indices_from(gs)] = 512
    gs2[-N:, :N][np.diag_indices_from(gs)] = 512
    gs2 = pd.DataFrame(gs2)

    first = elos.solve(ws, gs)

    second = elos.solve(ws2, gs2)
    second = pd.Series(second.values[:N], ws.index)

    pd.concat({'first': first, 'second': second}, 1).sort_values('first').plot.scatter('first', 'second')
