import numpy as np
from open_spiel.python.games import tic_tac_toe
from open_spiel.python.algorithms import mcts

class ZeroEvaluator:
    
    def prior(self, state):
        actions = state.legal_actions()
        return [(a, 1./len(actions)) for a in actions]
        
    def evaluate(self, state):
        return (0., 0.)
    
def prepare_game(actions_str):
    game = tic_tac_toe.TicTacToeGame()
    state = game.new_initial_state()
    for action_str in actions_str.split(' '):
        for action in state.legal_actions():
            if action_str == state.action_to_string(state.current_player(), action):
                state.apply_action(action)
                break
        else:
            raise ValueError("invalid action string: {}".format(action_str))
    return game, state

def board_actions(board):
    symbols = np.frombuffer((board + '\n').encode(), dtype='S1').reshape(3, 4)[:, :3]
    indices = np.indices(symbols.shape)

    xs = indices[:, symbols == b'x'].T
    os = indices[:, symbols == b'o'].T

    assert len(xs) - len(os) in {0, 1}

    actions_str = []
    for i in range(len(os)):
        actions_str.append(f'x({xs[i, 0]},{xs[i, 1]})')
        actions_str.append(f'o({os[i, 0]},{os[i, 1]})')

    if len(os) < len(xs):
        actions_str.append(f'x({xs[-1, 0]},{xs[-1, 1]})')

    return ' '.join(actions_str)
    
def run():
    board = '''
    xox
    xxo
    o..'''[1:]

    game, state = prepare_game(board_actions(board))

    bot = mcts.MCTSBot(game, 2.5, 16, ZeroEvaluator())

    root = bot.mcts_search(state)