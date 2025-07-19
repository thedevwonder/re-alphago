import random

from dlgo.agent import Agent
from dlgo.scoring import GameResult

__all__ = [
    'DepthPrunedAgent',
]

MAX = 9999999
MIN = -9999999

def best_result(game_state, max_depth, eval_fn):
    if game_state.is_over():
        if game_state.winner() == game_state.next_player:
            return MAX
        if game_state.winner() is None:
            return eval_fn(game_state)
        return MIN
    
    if max_depth == 0:
        return eval_fn(game_state)

    best_result_so_far = MIN
    for move in game_state.legal_moves():
        next_game_state = game_state.apply_move(move)
        opponent_best_result = best_result(next_game_state, max_depth=max_depth-1, eval_fn=eval_fn)
        our_best_result = -1*opponent_best_result
        if our_best_result > best_result_so_far:
            best_result_so_far = our_best_result
    return best_result_so_far
    

class DepthPrunedAgent(Agent):
    def __init__(self, max_depth, eval_fn):
        Agent.__init__(self)
        self.max_depth = max_depth
        self.eval_fn = eval_fn
  
    def select_move(self, game_state):
        # keep a best moves list
        best_moves = []
        # keep a best score
        best_score = None

        # we calculate our best result
        # we still need to maintain best moves and select random best move
        for move in game_state.legal_moves():
            our_move = game_state.apply_move(move)
            opponent_best_result = best_result(our_move, self.max_depth, self.eval_fn)
            our_best_result = -1*opponent_best_result
            if (not best_moves) or our_best_result > best_score:
                best_moves = [move]
                best_score = our_best_result
            elif our_best_result == best_score:
                best_moves.append(move)
        return random.choice(best_moves)
            